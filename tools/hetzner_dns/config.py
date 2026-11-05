"""Environment based configuration for the Hetzner DNS tool.

Reads the documented variables (no secrets are stored or logged):

========================  ==========================================================
Variable                  Meaning
========================  ==========================================================
``HETZNER_API_TOKEN``     Hetzner Cloud API token (required for API access)
``HETZNER_DNS_ZONE``      DNS zone, default ``mybister.com``
``HETZNER_DNS_ZONE_ID``   Optional zone id (skips zone lookup by name)
``HETZNER_DNS_TTL``       TTL in seconds for created/updated records, default ``60``
``API_SERVICES_DOMAIN``   Service domain, default ``api.mybister.com``
``DEFAULT_DOCKER_HOST``   Default Docker host, default ``b03.mybister.com``
``API_SERVICES_REGISTRY`` Optional path to a service registry JSON file
========================  ==========================================================

Importing this module has no side effects; values are only read when
:meth:`DnsConfig.from_env` is called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .errors import ConfigError
from .validators import normalize_target_host, validate_domain, validate_zone

DEFAULT_API_BASE_URL = "https://api.hetzner.cloud/v1"


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ConfigError(f"{key} muss eine ganze Zahl sein, gefunden: '{raw}'") from None


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ConfigError(f"{key} muss eine Zahl sein, gefunden: '{raw}'") from None


@dataclass(frozen=True)
class DnsConfig:
    """Immutable runtime configuration (the token is excluded from ``repr``)."""

    zone: str
    service_domain: str
    default_docker_host: str
    ttl: int = 60
    api_token: Optional[str] = field(default=None, repr=False)
    zone_id: Optional[str] = None
    api_base_url: str = DEFAULT_API_BASE_URL
    timeout: float = 30.0
    max_retries: int = 3
    action_timeout: float = 60.0
    action_poll_interval: float = 0.5
    registry_path: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "DnsConfig":
        """Build a config from a mapping (defaults to ``os.environ``)."""
        env = os.environ if env is None else env

        zone = validate_zone(env.get("HETZNER_DNS_ZONE", "mybister.com"))
        service_domain = validate_domain(env.get("API_SERVICES_DOMAIN", "api.mybister.com"), "API_SERVICES_DOMAIN")
        default_host = validate_domain(env.get("DEFAULT_DOCKER_HOST", "b03.mybister.com"), "DEFAULT_DOCKER_HOST")

        if service_domain != zone and not service_domain.endswith("." + zone):
            raise ConfigError(f"API_SERVICES_DOMAIN '{service_domain}' liegt ausserhalb der Zone '{zone}'")
        try:
            normalize_target_host(default_host, zone=zone, service_domain=service_domain, what="DEFAULT_DOCKER_HOST")
        except Exception as exc:  # noqa: BLE001 - re-raised as config error
            raise ConfigError(str(exc)) from None

        ttl = _read_int(env, "HETZNER_DNS_TTL", 60)
        if ttl < 1:
            raise ConfigError(f"HETZNER_DNS_TTL muss groesser als 0 sein, gefunden: {ttl}")

        timeout = _read_float(env, "HETZNER_API_TIMEOUT", 30.0)
        if timeout <= 0:
            raise ConfigError(f"HETZNER_API_TIMEOUT muss groesser als 0 sein, gefunden: {timeout}")

        max_retries = _read_int(env, "HETZNER_API_MAX_RETRIES", 3)
        if max_retries < 0:
            raise ConfigError(f"HETZNER_API_MAX_RETRIES darf nicht negativ sein, gefunden: {max_retries}")

        token = (env.get("HETZNER_API_TOKEN") or "").strip() or None
        zone_id = (env.get("HETZNER_DNS_ZONE_ID") or "").strip() or None
        registry_path = (env.get("API_SERVICES_REGISTRY") or "").strip() or None

        return cls(
            zone=zone,
            service_domain=service_domain,
            default_docker_host=default_host,
            ttl=ttl,
            api_token=token,
            zone_id=zone_id,
            timeout=timeout,
            max_retries=max_retries,
            registry_path=registry_path,
        )
