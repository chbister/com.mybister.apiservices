"""Low-level Hetzner Cloud API client for DNS zones and RR sets.

Only documented endpoints of the current Hetzner Cloud API
(``https://api.hetzner.cloud/v1``, see https://docs.hetzner.cloud/reference/cloud)
are used:

=============================================  =================================
Endpoint                                       Purpose
=============================================  =================================
``GET  /zones/{zone}``                         resolve/verify the DNS zone
``GET  /zones/{zone}/rrsets``                  list RR sets (paginated)
``GET  /zones/{zone}/rrsets/{name}/{type}``     read one RR set (404 = missing)
``POST /zones/{zone}/rrsets``                  create an RR set (async action)
``POST .../rrsets/{name}/{type}/actions/set_records``   overwrite records
``POST .../rrsets/{name}/{type}/actions/change_ttl``     change the TTL
``GET  /actions/{id}``                         poll async actions
=============================================  =================================

Authentication: ``Authorization: Bearer <token>``. The token is only ever
held in memory, is excluded from ``repr``/logs and is never written to disk.

Retry policy
------------
* Read requests (GET) are retried on ``rate_limit_exceeded``,
  ``conflict``, ``bad_gateway``, ``timeout``, HTTP 502/504 and network
  errors - mirroring the official Hetzner client.
* Write requests (POST) are only retried when the request was provably
  **not** processed: HTTP 429 / ``rate_limit_exceeded`` or a failed
  connection attempt (``httpx.ConnectError``/``ConnectTimeout``).
  Ambiguous failures (5xx, read timeouts) are *not* retried to avoid
  uncontrolled repeated writes; callers re-check the current state
  before writing instead (idempotency).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx

from .config import DEFAULT_API_BASE_URL
from .errors import (
    ActionFailedError,
    ActionTimeoutError,
    APIError,
    MissingTokenError,
    RateLimitError,
    TransportError,
    ZoneNotFoundError,
)

logger = logging.getLogger("tools.hetzner_dns.client")

# Error codes that the official Hetzner client retries for idempotent requests.
_RETRYABLE_API_CODES = frozenset({"conflict", "rate_limit_exceeded", "bad_gateway", "timeout"})
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 504})
# Write requests: only these are provably unprocessed and therefore safe to retry.
_SAFE_WRITE_API_CODES = frozenset({"rate_limit_exceeded"})
_SAFE_WRITE_STATUS_CODES = frozenset({429})
_SAFE_WRITE_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class HetznerCloudClient:
    """Small, dependency-light client for the Hetzner Cloud DNS endpoints."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        action_timeout: float = 60.0,
        action_poll_interval: float = 0.5,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token or not token.strip():
            raise MissingTokenError(
                "HETZNER_API_TOKEN ist nicht gesetzt. "
                "Bitte einen Hetzner-Cloud-API-Token exportieren (siehe docs/hetzner-dns.md)."
            )
        self._token = token.strip()
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._action_timeout = action_timeout
        self._action_poll_interval = action_poll_interval
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
                "User-Agent": "tools.hetzner_dns",
            },
            transport=transport,
        )

    def __repr__(self) -> str:  # never expose the token
        return f"HetznerCloudClient(base_url='{self._http.base_url}', token='***')"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "HetznerCloudClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # low level request handling
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        idempotent: bool = True,
    ) -> Dict[str, Any]:
        """Perform a request and return the decoded JSON body."""
        attempt = 0
        while True:
            try:
                response = self._http.request(method, path, params=params, json=json_body)
            except httpx.RequestError as exc:
                safe = idempotent or isinstance(exc, _SAFE_WRITE_TRANSPORT_ERRORS)
                if safe and attempt < self._max_retries:
                    logger.debug(
                        "Transportfehler bei %s %s (%s), Retry %d/%d",
                        method, path, type(exc).__name__, attempt + 1, self._max_retries,
                    )
                    self._sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                logger.warning("Verbindung zur Hetzner-API fehlgeschlagen: %s", type(exc).__name__)
                raise TransportError(
                    f"Verbindung zur Hetzner-API fehlgeschlagen ({type(exc).__name__})",
                    status_code=None,
                    code="transport_error",
                ) from None

            if response.status_code < 400:
                logger.debug("%s %s -> %d", method, path, response.status_code)
                return self._decode(response)

            error = self._error_from_response(response)
            if self._should_retry(error, idempotent=idempotent) and attempt < self._max_retries:
                logger.debug(
                    "%s %s -> Fehler (Retry %d/%d): %s",
                    method, path, attempt + 1, self._max_retries,
                    getattr(error, "code", None) or response.status_code,
                )
                self._sleep(self._backoff(attempt, response))
                attempt += 1
                continue
            logger.warning(
                "Hetzner-API-Fehler bei %s %s: status=%d code=%s",
                method, path, response.status_code, getattr(error, "code", None),
            )
            raise error

    @staticmethod
    def _decode(response: httpx.Response) -> Dict[str, Any]:
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError:
            raise APIError(
                "Unerwartete (nicht-JSON) Antwort von der Hetzner-API",
                status_code=response.status_code,
                code="json_error",
            ) from None
        if not isinstance(data, dict):
            raise APIError(
                "Unerwartetes Antwortformat der Hetzner-API",
                status_code=response.status_code,
                code="json_error",
            ) from None
        return data

    @staticmethod
    def _error_from_response(response: httpx.Response) -> APIError:
        code: Optional[str] = None
        message = f"Hetzner-API-Fehler (HTTP {response.status_code})"
        correlation_id = response.headers.get("X-Correlation-Id")
        try:
            payload = response.json()
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                code = err.get("code")
                message = str(err.get("message") or message)
        except ValueError:
            pass

        kwargs = {
            "status_code": response.status_code,
            "code": code,
            "correlation_id": correlation_id,
        }
        if response.status_code == 429 or code == "rate_limit_exceeded":
            return RateLimitError(
                f"Rate-Limit der Hetzner-API erreicht: {message}",
                retry_after=_retry_after_seconds(response),
                **kwargs,
            )
        return APIError(f"Hetzner-API-Fehler: {message}", **kwargs)

    @staticmethod
    def _should_retry(error: APIError, *, idempotent: bool) -> bool:
        if idempotent:
            return (
                (error.code in _RETRYABLE_API_CODES)
                or error.status_code in _RETRYABLE_STATUS_CODES
                or isinstance(error, RateLimitError)
            )
        # Writes: only provably unprocessed requests may be repeated.
        return (error.code in _SAFE_WRITE_API_CODES) or (error.status_code in _SAFE_WRITE_STATUS_CODES)

    def _backoff(self, attempt: int, response: Optional[httpx.Response] = None) -> float:
        delay = self._retry_base_delay * (2 ** attempt)
        if response is not None:
            reset = _retry_after_seconds(response)
            if reset is not None:
                delay = max(delay, reset)
        return min(delay, self._retry_max_delay)

    # ------------------------------------------------------------------ #
    # pagination
    # ------------------------------------------------------------------ #
    def _get_all(self, path: str, params: Dict[str, Any], list_key: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        page = 1
        while True:
            data = self._request("GET", path, params={**params, "page": page, "per_page": 50}, idempotent=True)
            page_items = data.get(list_key) or []
            if not isinstance(page_items, list):
                raise APIError(f"Unerwartetes Antwortformat: '{list_key}' ist keine Liste")
            items.extend(page_items)
            meta = data.get("meta") or {}
            pagination = meta.get("pagination") or {}
            next_page = pagination.get("next_page")
            if not next_page:
                break
            page = int(next_page)
        return items

    # ------------------------------------------------------------------ #
    # zones
    # ------------------------------------------------------------------ #
    def get_zone(self, id_or_name: str) -> Dict[str, Any]:
        """Return the zone for an id or name; raises ZoneNotFoundError."""
        path = f"/zones/{quote(str(id_or_name), safe='')}"
        try:
            data = self._request("GET", path, idempotent=True)
        except APIError as exc:
            if exc.code == "not_found" or exc.status_code == 404:
                raise ZoneNotFoundError(
                    f"DNS-Zone '{id_or_name}' wurde in dem Hetzner-Konto nicht gefunden "
                    "(HETZNER_DNS_ZONE / HETZNER_DNS_ZONE_ID pruefen)"
                ) from None
            raise
        zone = data.get("zone")
        if not isinstance(zone, dict):
            raise APIError("Unerwartetes Antwortformat: 'zone' fehlt")
        return zone

    # ------------------------------------------------------------------ #
    # RR sets
    # ------------------------------------------------------------------ #
    def list_rrsets(self, zone: str, *, name: Optional[str] = None, type: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if name:
            params["name"] = name
        if type:
            params["type"] = type
        return self._get_all(f"/zones/{quote(str(zone), safe='')}/rrsets", params, "rrsets")

    def get_rrset(self, zone: str, name: str, type: str) -> Optional[Dict[str, Any]]:
        """Return one RR set or ``None`` if it does not exist (404)."""
        path = (
            f"/zones/{quote(str(zone), safe='')}/rrsets"
            f"/{quote(name, safe='')}/{quote(type, safe='')}"
        )
        try:
            data = self._request("GET", path, idempotent=True)
        except APIError as exc:
            if exc.code == "not_found" or exc.status_code == 404:
                return None
            raise
        rrset = data.get("rrset")
        return rrset if isinstance(rrset, dict) else None

    def create_rrset(
        self, zone: str, name: str, type: str, *, ttl: int, records: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Create an RR set (async) and wait for the action to finish."""
        path = f"/zones/{quote(str(zone), safe='')}/rrsets"
        data = self._request(
            "POST",
            path,
            json_body={"name": name, "type": type, "ttl": ttl, "records": records},
            idempotent=False,
        )
        self.wait_for_action(data.get("action"))
        rrset = data.get("rrset")
        return rrset if isinstance(rrset, dict) else {}

    def set_rrset_records(
        self, zone: str, name: str, type: str, *, records: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Overwrite the records of an existing RR set (async action)."""
        path = (
            f"/zones/{quote(str(zone), safe='')}/rrsets"
            f"/{quote(name, safe='')}/{quote(type, safe='')}/actions/set_records"
        )
        data = self._request("POST", path, json_body={"records": records}, idempotent=False)
        return self.wait_for_action(data.get("action")) or {}

    def change_rrset_ttl(self, zone: str, name: str, type: str, *, ttl: int) -> Dict[str, Any]:
        """Change the TTL of an existing RR set (async action)."""
        path = (
            f"/zones/{quote(str(zone), safe='')}/rrsets"
            f"/{quote(name, safe='')}/{quote(type, safe='')}/actions/change_ttl"
        )
        data = self._request("POST", path, json_body={"ttl": ttl}, idempotent=False)
        return self.wait_for_action(data.get("action")) or {}

    # ------------------------------------------------------------------ #
    # asynchronous actions
    # ------------------------------------------------------------------ #
    def get_action(self, action_id: Any) -> Dict[str, Any]:
        data = self._request("GET", f"/actions/{quote(str(action_id), safe='')}", idempotent=True)
        action = data.get("action")
        if not isinstance(action, dict):
            raise APIError("Unerwartetes Antwortformat: 'action' fehlt")
        return action

    def wait_for_action(self, action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Poll an async action until it succeeds (or raise on error/timeout)."""
        if not action:
            return None
        current = action
        deadline = time.monotonic() + self._action_timeout
        while current.get("status") == "running":
            if time.monotonic() > deadline:
                raise ActionTimeoutError(
                    f"Asynchrone Aktion {current.get('id')} hat {self._action_timeout}s nicht abgeschlossen",
                    code="action_timeout",
                )
            self._sleep(self._action_poll_interval)
            current = self.get_action(current.get("id"))

        status = current.get("status")
        err = current.get("error")
        if status == "error" or (isinstance(err, dict) and err):
            err = err if isinstance(err, dict) else {}
            raise ActionFailedError(
                f"Asynchrone Aktion {current.get('id')} fehlgeschlagen: "
                f"{err.get('message') or 'unbekannter Fehler'} ({err.get('code') or 'unknown'})",
                code=str(err.get("code") or "action_error"),
            )
        return current


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    """Best effort delay from ``Retry-After`` / ``RateLimit-Reset`` headers."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    reset = response.headers.get("RateLimit-Reset")
    if reset:
        try:
            return max(0.0, float(reset) - time.time())
        except ValueError:
            pass
    return None
