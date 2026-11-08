"""Unit tests for the Hetzner DNS tool with a mocked Hetzner Cloud API.

No network access and no productive DNS changes: the HTTP layer is
replaced by ``httpx.MockTransport`` and an in-memory fake implementation
of the documented endpoints:

* ``GET  /zones/{zone}``
* ``GET  /zones/{zone}/rrsets`` (with pagination)
* ``GET  /zones/{zone}/rrsets/{name}/{type}``
* ``POST /zones/{zone}/rrsets``
* ``POST /zones/{zone}/rrsets/{name}/{type}/actions/set_records``
* ``POST /zones/{zone}/rrsets/{name}/{type}/actions/change_ttl``
* ``GET  /actions/{id}``

Run from the repository root::

    .venv/bin/python -m pytest tools/hetzner_dns/tests -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import pytest

from tools.hetzner_dns import cli
from tools.hetzner_dns.client import HetznerCloudClient
from tools.hetzner_dns.config import DnsConfig
from tools.hetzner_dns.errors import (
    APIError,
    ActionFailedError,
    MissingTokenError,
    ValidationError,
    ZoneNotFoundError,
)
from tools.hetzner_dns.manager import (
    STATUS_CONFLICT,
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_UNCHANGED,
    STATUS_UPDATED,
    HetznerDNSManager,
)
from tools.hetzner_dns.registry import ServiceDefinition, ServiceRegistry

TOKEN = "super-secret-hetzner-token-123"
ZONE = "mybister.com"
SERVICE_DOMAIN = "api.mybister.com"
DEFAULT_HOST = "b03.mybister.com"
REPO_ROOT = Path(__file__).resolve().parents[3]

SMALL_SERVICES = (
    ServiceDefinition(id="asr-tiny", description="ASR tiny"),
    ServiceDefinition(id="asr-small", description="ASR small"),
    ServiceDefinition(id="sentiment", description="Sentiment"),
)


# --------------------------------------------------------------------------- #
# fake Hetzner Cloud API
# --------------------------------------------------------------------------- #
class FakeHetznerAPI:
    """In-memory implementation of the documented Hetzner DNS endpoints.

    Like the real API it resolves every RR set name **zone-relative**: the
    zone name is appended to whatever name a request carries (``@`` is the
    apex). A client sending a full FQDN therefore creates a doubled record
    (``<fqdn>.<zone>``) - the tests rely on this to catch wrong wire names.
    """

    def __init__(
        self,
        *,
        zone: str = ZONE,
        zone_id: int = 4242,
        token: str = TOKEN,
        rrsets: Optional[List[Dict]] = None,
        page_size: Optional[int] = None,
        async_actions: bool = False,
        action_error: Optional[Tuple[str, str]] = None,
        zone_exists: bool = True,
    ) -> None:
        self.zone = zone
        self.zone_id = zone_id
        self.token = token
        self.zone_exists = zone_exists
        self.page_size = page_size
        self.async_actions = async_actions
        self.action_error = action_error
        self.rrsets: Dict[Tuple[str, str], Dict] = {}
        for rrset in rrsets or []:
            self.add_rrset(
                rrset["name"],
                rrset["type"],
                records=rrset.get("records", []),
                ttl=rrset.get("ttl", 300),
            )
        self.requests: List[Dict] = []
        self.write_requests: List[Dict] = []
        self._actions: Dict[int, Dict] = {}
        self._running: set = set()
        self._next_action = 1
        self._fail_queue: List[object] = []
        self._method_failures: List[Tuple[str, object]] = []

    # -- seeding / scripting --------------------------------------------
    def _resolve_name(self, name: str) -> str:
        """Zone-relative name resolution of the real API (zone is appended)."""
        bare = (name or "").strip().rstrip(".").lower()
        if bare in ("", "@"):
            return self.zone.lower()
        return f"{bare}.{self.zone.lower()}"

    def add_rrset(self, name: str, type: str, *, records: List[Dict], ttl: int = 300) -> None:
        self.rrsets[(name.lower(), type.upper())] = {
            "id": f"{name}/{type}",
            "name": name.lower(),
            "type": type.upper(),
            "ttl": ttl,
            "labels": {},
            "protection": {"change": False},
            "records": [dict(r) for r in records],
            "zone": self.zone_id,
        }

    def cname_target(self, fqdn: str) -> Optional[str]:
        rrset = self.rrsets.get((fqdn.lower(), "CNAME"))
        if not rrset or not rrset["records"]:
            return None
        return rrset["records"][0]["value"]

    def fail_once(self, item: object) -> None:
        """Script the next request: an exception to raise or a response tuple.

        Response tuple: ``(status_code, body_dict[, headers_dict])``.
        """
        self._fail_queue.append(item)

    def fail_next_matching(self, method: str, item: object) -> None:
        """Fail the next request with the given HTTP method (e.g. ``POST``)."""
        self._method_failures.append((method, item))

    # -- httpx transport --------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.method
        path = request.url.path
        query = {k: v for k, v in request.url.params.items()}
        body = json.loads(request.content.decode()) if request.content else None
        self.requests.append({"method": method, "path": path, "query": query, "body": body})
        if method != "GET":
            self.write_requests.append({"method": method, "path": path, "body": body})

        for index, (wanted_method, item) in enumerate(self._method_failures):
            if wanted_method == method:
                del self._method_failures[index]
                return self._deliver_failure(item)

        if self._fail_queue:
            return self._deliver_failure(self._fail_queue.pop(0))

        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return httpx.Response(
                401, json={"error": {"code": "unauthorized", "message": "invalid token provided"}}
            )

        parts = [p for p in path.split("/") if p]
        if parts and parts[0] == "v1":  # base_url prefix (https://api.hetzner.cloud/v1)
            parts = parts[1:]

        # GET /actions/{id}
        if method == "GET" and len(parts) == 2 and parts[0] == "actions":
            return self._get_action(int(parts[1]))

        # /zones/...
        if not parts or parts[0] != "zones":
            return self._error(404, "not_found", "unknown endpoint")
        if len(parts) == 1:  # /zones (not used by the tool, but harmless)
            return httpx.Response(200, json={"zones": []})
        zone_part = parts[1]
        if not self.zone_exists or zone_part not in (self.zone, str(self.zone_id)):
            return self._error(404, "not_found", f"zone {zone_part} not found")
        if len(parts) == 2:  # GET /zones/{zone}
            return httpx.Response(
                200,
                json={
                    "zone": {
                        "id": self.zone_id,
                        "name": self.zone,
                        "ttl": 3600,
                        "mode": "primary",
                        "labels": {},
                        "record_count": len(self.rrsets),
                    }
                },
            )

        # /zones/{zone}/rrsets...
        if parts[2] != "rrsets":
            return self._error(404, "not_found", "unknown endpoint")
        if len(parts) == 3:
            if method == "GET":
                return self._list_rrsets(query)
            if method == "POST":
                return self._create_rrset(body)
            return self._error(405, "unsupported_error", "method not allowed")

        name = self._resolve_name(parts[3])
        if len(parts) == 4:
            return self._error(404, "not_found", "record type required")
        rtype = parts[4].upper()
        if len(parts) == 5:
            if method == "GET":
                rrset = self.rrsets.get((name.lower(), rtype))
                if rrset is None:
                    return self._error(404, "not_found", "rrset not found")
                return httpx.Response(200, json={"rrset": rrset})
            return self._error(405, "unsupported_error", "method not allowed")
        if parts[5] != "actions" or len(parts) != 7:
            return self._error(404, "not_found", "unknown endpoint")
        action_name = parts[6]
        if method != "POST":
            return self._error(405, "unsupported_error", "method not allowed")
        rrset = self.rrsets.get((name.lower(), rtype))
        if rrset is None:
            return self._error(404, "not_found", "rrset not found")
        if action_name == "set_records":
            rrset["records"] = [dict(r) for r in body.get("records", [])]
            return httpx.Response(200, json={"action": self._new_action("set_records")})
        if action_name == "change_ttl":
            rrset["ttl"] = int(body.get("ttl"))
            return httpx.Response(200, json={"action": self._new_action("change_ttl")})
        return self._error(404, "not_found", "unknown action")

    def _list_rrsets(self, query: Dict[str, str]) -> httpx.Response:
        items = sorted(self.rrsets.values(), key=lambda r: (r["name"], r["type"]))
        name_filter = query.get("name")
        type_filter = query.get("type")
        if name_filter:
            items = [r for r in items if r["name"] == self._resolve_name(name_filter)]
        if type_filter:
            items = [r for r in items if r["type"] == type_filter.upper()]

        page = int(query.get("page", 1))
        per_page = int(query.get("per_page", 25))
        if self.page_size:
            per_page = self.page_size
        start = (page - 1) * per_page
        window = items[start : start + per_page]
        next_page = page + 1 if start + per_page < len(items) else None
        return httpx.Response(
            200,
            json={
                "rrsets": window,
                "meta": {
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "previous_page": page - 1 if page > 1 else None,
                        "next_page": next_page,
                        "last_page": ((len(items) - 1) // per_page) + 1 if items else 1,
                        "total_entries": len(items),
                    }
                },
            },
        )

    def _create_rrset(self, body: Dict) -> httpx.Response:
        name = self._resolve_name(body.get("name", ""))
        rtype = str(body.get("type", "")).upper()
        if (name, rtype) in self.rrsets:
            return self._error(422, "uniqueness_error", "rrset already exists")
        self.add_rrset(
            name,
            rtype,
            records=body.get("records", []),
            ttl=int(body.get("ttl") or 300),
        )
        return httpx.Response(
            200, json={"rrset": self.rrsets[(name, rtype)], "action": self._new_action("create_rrset")}
        )

    def _new_action(self, command: str) -> Dict:
        action_id = self._next_action
        self._next_action += 1
        status = "running" if self.async_actions else ("error" if self.action_error else "success")
        action = {
            "id": action_id,
            "command": command,
            "status": status,
            "progress": 0 if status == "running" else 100,
            "error": (
                {"code": self.action_error[0], "message": self.action_error[1]}
                if status == "error"
                else None
            ),
        }
        self._actions[action_id] = action
        if status == "running":
            self._running.add(action_id)
        return dict(action)

    def _get_action(self, action_id: int) -> httpx.Response:
        action = self._actions.get(action_id)
        if action is None:
            return self._error(404, "not_found", "action not found")
        snapshot = dict(action)
        if action_id in self._running:
            # First poll still sees "running", the action completes afterwards.
            self._running.discard(action_id)
            if self.action_error:
                action["status"] = "error"
                action["error"] = {"code": self.action_error[0], "message": self.action_error[1]}
            else:
                action["status"] = "success"
                action["progress"] = 100
        return httpx.Response(200, json={"action": snapshot})

    @staticmethod
    def _error(status: int, code: str, message: str) -> httpx.Response:
        return httpx.Response(status, json={"error": {"code": code, "message": message}})

    @staticmethod
    def _deliver_failure(item: object) -> httpx.Response:
        if isinstance(item, Exception):
            raise item
        status, payload = item[0], item[1]
        headers = item[2] if len(item) > 2 else None
        return httpx.Response(status, json=payload, headers=headers)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def make_env(**overrides: str) -> Dict[str, str]:
    env = {
        "HETZNER_API_TOKEN": TOKEN,
        "HETZNER_DNS_ZONE": ZONE,
        "API_SERVICES_DOMAIN": SERVICE_DOMAIN,
        "DEFAULT_DOCKER_HOST": DEFAULT_HOST,
        "HETZNER_DNS_TTL": "60",
    }
    env.update(overrides)
    return env


def make_client(fake: FakeHetznerAPI, **kwargs) -> HetznerCloudClient:
    kwargs.setdefault("retry_base_delay", 0.0)
    kwargs.setdefault("action_poll_interval", 0.0)
    kwargs.setdefault("action_timeout", 5.0)
    kwargs.setdefault("max_retries", 3)
    return HetznerCloudClient(
        TOKEN, transport=httpx.MockTransport(fake.handler), **kwargs
    )


def make_manager(
    fake: FakeHetznerAPI,
    *,
    env: Optional[Dict[str, str]] = None,
    registry: Optional[ServiceRegistry] = None,
) -> HetznerDNSManager:
    config = DnsConfig.from_env(env or make_env())
    registry = registry or ServiceRegistry(services=SMALL_SERVICES)
    return HetznerDNSManager(config, registry, make_client(fake))


def run_cli(args: List[str], fake: FakeHetznerAPI, monkeypatch, capsys, registry=None) -> Tuple[int, str, str]:
    manager = make_manager(fake, registry=registry)
    monkeypatch.setattr(cli, "_build_manager", lambda: manager)
    code = cli.main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_requests(fake: FakeHetznerAPI) -> List[Dict]:
    return [r for r in fake.write_requests if r["method"] in ("POST", "PUT", "DELETE")]


# --------------------------------------------------------------------------- #
# 1 + 2: missing CNAME is detected / created with --apply
# --------------------------------------------------------------------------- #
def test_missing_cname_is_detected_in_plan():
    fake = FakeHetznerAPI()
    manager = make_manager(fake)
    change = manager.plan_service_record("asr-small", "b03")
    assert change.status == STATUS_CREATED
    assert change.target == "b03.mybister.com."  # CNAME values are absolute
    assert change.applied is False
    assert write_requests(fake) == []


def test_missing_cname_is_created_with_apply():
    fake = FakeHetznerAPI()
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_CREATED
    assert change.applied is True
    # the wire value must be absolute (trailing dot)
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."
    creates = [r for r in fake.write_requests if r["path"].endswith("/rrsets")]
    assert len(creates) == 1
    assert creates[0]["body"]["ttl"] == 60
    assert creates[0]["body"]["records"] == [{"value": "b03.mybister.com."}]


# --------------------------------------------------------------------------- #
# 3: identical CNAME is skipped
# --------------------------------------------------------------------------- #
def test_identical_cname_is_unchanged_and_not_written():
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com."}], "ttl": 60}]
    )
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_UNCHANGED
    assert change.applied is False
    assert write_requests(fake) == []


def test_record_without_trailing_dot_is_updated_by_set():
    """A stored value missing the dot is a real difference and gets fixed."""
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}], "ttl": 60}]
    )
    manager = make_manager(fake)
    record = manager.get_service_record("asr-small")
    assert record.target == "b03.mybister.com"  # raw stored value stays visible
    assert record.status == "differs"

    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_UPDATED
    assert change.applied is True
    assert change.current_target == "b03.mybister.com"
    assert change.target == "b03.mybister.com."
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."
    assert write_requests(fake), "dotless value must be rewritten"


def test_record_without_trailing_dot_is_conflict_in_bulk_without_force():
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}], "ttl": 60}]
    )
    manager = make_manager(fake)
    plan = manager.plan_bulk("b03")
    by_id = {c.service_id: c for c in plan}
    assert by_id["asr-small"].status == STATUS_CONFLICT  # needs --force to normalize
    results = manager.apply_bulk(plan)
    assert {r.service_id: r for r in results}["asr-small"].status == STATUS_CONFLICT
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com"  # untouched

    forced = manager.apply_bulk(manager.plan_bulk("b03", force=True))
    assert {r.service_id: r for r in forced}["asr-small"].status == STATUS_UPDATED
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."


# --------------------------------------------------------------------------- #
# 4: diverging CNAME is updated by single "set"
# --------------------------------------------------------------------------- #
def test_diverging_cname_is_updated_by_set():
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}], "ttl": 300}]
    )
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b02", apply=True)
    assert change.status == STATUS_UPDATED
    assert change.applied is True
    assert fake.cname_target("asr-small.api.mybister.com") == "b02.mybister.com."
    assert any(r["path"].endswith("/actions/set_records") for r in fake.write_requests)
    # the set_records payload carries the absolute value
    set_records = [r for r in fake.write_requests if r["path"].endswith("/actions/set_records")]
    assert set_records[0]["body"]["records"] == [{"value": "b02.mybister.com."}]
    # TTL drift is converged to the configured value as part of the update
    assert fake.rrsets[("asr-small.api.mybister.com", "CNAME")]["ttl"] == 60


# --------------------------------------------------------------------------- #
# 4b: wire names are zone-relative (the API appends the zone name)
# --------------------------------------------------------------------------- #
def test_create_sends_zone_relative_name():
    """Regression: a full FQDN on the wire becomes <fqdn>.<zone> (doubled)."""
    fake = FakeHetznerAPI()
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_CREATED
    creates = [r for r in fake.write_requests if r["path"].endswith("/rrsets")]
    assert creates[0]["body"]["name"] == "asr-small.api"  # NOT "asr-small.api.mybister.com"
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."
    assert fake.cname_target("asr-small.api.mybister.com.mybister.com") is None


def test_update_uses_zone_relative_name_in_path():
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}], "ttl": 300}]
    )
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b02", apply=True)
    assert change.status == STATUS_UPDATED
    paths = [r["path"] for r in fake.write_requests if r["path"].endswith("/actions/set_records")]
    assert paths, "update must go through set_records"
    assert all("/rrsets/asr-small.api/CNAME/" in p for p in paths)
    assert fake.cname_target("asr-small.api.mybister.com") == "b02.mybister.com."


def test_reads_tolerate_relative_and_absolute_rrset_names():
    fake = FakeHetznerAPI()
    fake.add_rrset("asr-small.api", "CNAME", records=[{"value": "b03.mybister.com."}], ttl=60)
    manager = make_manager(fake)
    record = manager.get_service_record("asr-small")
    assert record.exists is True
    assert record.target == "b03.mybister.com."


# --------------------------------------------------------------------------- #
# 4c: registry target hosts (field "host") and "sync"
# --------------------------------------------------------------------------- #
def test_registry_hosts_are_valid_and_complete():
    registry = ServiceRegistry.load()
    hosts = {service.id: service.host for service in registry}
    assert hosts
    for service_id, host in hosts.items():
        assert host, f"{service_id} hat keinen Ziel-Host (Feld 'host')"
        assert host == host.strip().lower().rstrip("."), f"{service_id}: host nicht normalisiert"
        assert re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)*", host), f"{service_id}: ungueltiger host '{host}'"


def test_expected_host_comes_from_registry_with_fallback():
    services = (
        ServiceDefinition(id="asr-tiny", description="ASR tiny", host="b01"),
        ServiceDefinition(id="asr-small", description="ASR small", host="b02.mybister.com"),
        ServiceDefinition(id="sentiment", description="Sentiment"),  # no assignment
    )
    manager = make_manager(FakeHetznerAPI(), registry=ServiceRegistry(services=services))
    assert manager.get_service_record("asr-tiny").expected_host == "b01.mybister.com."
    assert manager.get_service_record("asr-small").expected_host == "b02.mybister.com."
    assert manager.get_service_record("sentiment").expected_host == DEFAULT_HOST + "."


def test_sync_reconciles_assigned_hosts():
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com."}], "ttl": 60}]
    )
    services = (
        ServiceDefinition(id="asr-tiny", description="ASR tiny", host="b01"),
        ServiceDefinition(id="asr-small", description="ASR small", host="b02"),
        ServiceDefinition(id="sentiment", description="Sentiment"),  # no assignment
    )
    manager = make_manager(fake, registry=ServiceRegistry(services=services))

    plan = manager.plan_sync()
    by_id = {c.service_id: c for c in plan}
    assert by_id["asr-tiny"].status == STATUS_CREATED
    assert by_id["asr-small"].status == STATUS_UPDATED  # b03 -> b02 per registry
    assert by_id["sentiment"].status == STATUS_FAILED  # missing "host" field
    assert write_requests(fake) == []  # dry plan writes nothing

    results = {c.service_id: c for c in manager.apply_bulk(plan)}
    assert results["asr-tiny"].applied is True
    assert results["asr-small"].applied is True
    assert results["sentiment"].applied is False
    assert fake.cname_target("asr-tiny.api.mybister.com") == "b01.mybister.com."
    assert fake.cname_target("asr-small.api.mybister.com") == "b02.mybister.com."


def test_cli_sync_applies_registry_hosts(capsys, monkeypatch):
    fake = FakeHetznerAPI()
    services = (ServiceDefinition(id="asr-small", description="ASR small", host="b02"),)
    code, out, err = run_cli(
        ["sync", "--apply"], fake, monkeypatch, capsys, registry=ServiceRegistry(services=services)
    )
    assert code == cli.EXIT_OK
    assert "CREATE" in out
    assert fake.cname_target("asr-small.api.mybister.com") == "b02.mybister.com."


# --------------------------------------------------------------------------- #
# 5: bulk dry run writes nothing
# --------------------------------------------------------------------------- #
def test_bulk_dry_run_changes_nothing():
    fake = FakeHetznerAPI()
    manager = make_manager(fake)
    plan = manager.plan_bulk("b03")
    assert len(plan) == len(SMALL_SERVICES)
    assert all(c.status == STATUS_CREATED for c in plan)
    assert write_requests(fake) == []
    # plan is not applied by apply_bulk-less usage either
    assert all(c.applied is False for c in plan)


# --------------------------------------------------------------------------- #
# 6: bulk creates missing CNAMEs
# --------------------------------------------------------------------------- #
def test_bulk_apply_creates_missing_cnames():
    fake = FakeHetznerAPI(
        rrsets=[
            {"name": "sentiment.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com."}], "ttl": 60}
        ]
    )
    manager = make_manager(fake)
    results = manager.apply_bulk(manager.plan_bulk("b03"))
    by_id = {r.service_id: r for r in results}
    assert by_id["asr-tiny"].status == STATUS_CREATED
    assert by_id["asr-small"].status == STATUS_CREATED
    assert by_id["sentiment"].status == STATUS_UNCHANGED
    assert fake.cname_target("asr-tiny.api.mybister.com") == "b03.mybister.com."
    # already correct and never written -> stored value stays as-is
    assert fake.cname_target("sentiment.api.mybister.com") == "b03.mybister.com."
    assert len([r for r in fake.write_requests if r["path"].endswith("/rrsets")]) == 2


# --------------------------------------------------------------------------- #
# 7: bulk does not overwrite diverging CNAMEs without --force
# --------------------------------------------------------------------------- #
def test_bulk_without_force_reports_conflict_and_does_not_write():
    fake = FakeHetznerAPI(
        rrsets=[
            {"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b01.mybister.com"}], "ttl": 60}
        ]
    )
    manager = make_manager(fake)
    plan = manager.plan_bulk("b03")
    results = manager.apply_bulk(plan)
    by_id = {r.service_id: r for r in results}
    assert by_id["asr-small"].status == STATUS_CONFLICT
    assert by_id["asr-small"].current_target == "b01.mybister.com"  # raw stored value
    assert by_id["asr-tiny"].status == STATUS_CREATED
    # the diverging record stays untouched, only missing ones are created
    assert fake.cname_target("asr-small.api.mybister.com") == "b01.mybister.com"
    assert fake.cname_target("asr-tiny.api.mybister.com") == "b03.mybister.com."
    assert not any("set_records" in r["path"] for r in fake.write_requests)


# --------------------------------------------------------------------------- #
# 8: bulk with --force updates diverging CNAMEs
# --------------------------------------------------------------------------- #
def test_bulk_with_force_updates_diverging_cnames():
    fake = FakeHetznerAPI(
        rrsets=[
            {"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b01.mybister.com"}], "ttl": 60}
        ]
    )
    manager = make_manager(fake)
    results = manager.apply_bulk(manager.plan_bulk("b03", force=True))
    by_id = {r.service_id: r for r in results}
    assert by_id["asr-small"].status == STATUS_UPDATED
    assert by_id["asr-small"].applied is True
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."


# --------------------------------------------------------------------------- #
# 9: existing A/AAAA records are never touched
# --------------------------------------------------------------------------- #
def test_foreign_record_types_are_conflicts_and_untouched():
    fake = FakeHetznerAPI(
        rrsets=[
            {"name": "asr-small.api.mybister.com", "type": "A", "records": [{"value": "203.0.113.10"}], "ttl": 300},
            {"name": "asr-tiny.api.mybister.com", "type": "AAAA", "records": [{"value": "2001:db8::1"}], "ttl": 300},
        ]
    )
    manager = make_manager(fake)

    record = manager.get_service_record("asr-small")
    assert record.status == STATUS_CONFLICT
    assert record.conflict_types == ["A"]

    single = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert single.status == STATUS_CONFLICT
    assert single.applied is False

    results = manager.apply_bulk(manager.plan_bulk("b03"))
    by_id = {r.service_id: r for r in results}
    assert by_id["asr-small"].status == STATUS_CONFLICT
    assert by_id["asr-tiny"].status == STATUS_CONFLICT

    # both foreign records still exist, untouched
    assert fake.rrsets[("asr-small.api.mybister.com", "A")]["records"] == [{"value": "203.0.113.10"}]
    assert fake.rrsets[("asr-tiny.api.mybister.com", "AAAA")]["records"] == [{"value": "2001:db8::1"}]
    # no write touched a conflicting name; only the unrelated missing service was created
    created_names = [r["body"]["name"] for r in fake.write_requests if r["path"].endswith("/rrsets")]
    assert "asr-small.api.mybister.com" not in created_names
    assert "asr-tiny.api.mybister.com" not in created_names
    assert not any("set_records" in r["path"] or "change_ttl" in r["path"] for r in fake.write_requests)


# --------------------------------------------------------------------------- #
# 10: invalid service ids are rejected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_id", ["-bad-", "has space", "a" * 70, "", "UP PER!"])
def test_invalid_service_ids_are_rejected(bad_id):
    fake = FakeHetznerAPI()
    manager = make_manager(fake)
    with pytest.raises(ValidationError):
        manager.plan_service_record(bad_id, "b03")


def test_unknown_but_syntactically_valid_service_id_is_rejected():
    fake = FakeHetznerAPI()
    manager = make_manager(fake)
    with pytest.raises(ValidationError) as excinfo:
        manager.plan_service_record("does-not-exist", "b03")
    assert "Registry" in str(excinfo.value)
    assert write_requests(fake) == []


def test_cli_rejects_invalid_service_id(monkeypatch, capsys):
    fake = FakeHetznerAPI()
    code, _out, err = run_cli(["set", "has space", "b03", "--apply"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_USAGE
    assert "Validation error" in err
    # argparse rejects a leading-dash id as an option (also exit code 2)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["set", "-bad-", "b03", "--apply"])
    assert excinfo.value.code == cli.EXIT_USAGE
    assert write_requests(fake) == []


# --------------------------------------------------------------------------- #
# 11: invalid target hosts are rejected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_host",
    [
        "evil.example.com",        # outside the zone
        "foo.api.mybister.com",    # inside the service domain
        "b03..mybister.com",       # invalid label
        "not a host",              # invalid characters
        "-b02",                    # invalid label
        "",                        # empty
    ],
)
def test_invalid_target_hosts_are_rejected(bad_host):
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}]}]
    )
    manager = make_manager(fake)
    with pytest.raises(ValidationError):
        manager.ensure_service_record("asr-small", bad_host, apply=True)
    assert write_requests(fake) == []


def test_short_host_is_expanded_to_zone(monkeypatch, capsys):
    fake = FakeHetznerAPI()
    code, out, _err = run_cli(["set", "asr-small", "b02"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    assert "Target:  b02.mybister.com." in out  # dry run shows the absolute value
    assert write_requests(fake) == []  # dry run


# --------------------------------------------------------------------------- #
# 12: missing API token fails for write operations
# --------------------------------------------------------------------------- #
def test_missing_token_fails_for_writes():
    env = {k: v for k, v in make_env().items() if k != "HETZNER_API_TOKEN"}
    with pytest.raises(MissingTokenError):
        HetznerDNSManager.from_env(env, registry=ServiceRegistry(services=SMALL_SERVICES))
    with pytest.raises(MissingTokenError):
        HetznerCloudClient("")


def test_cli_set_apply_without_token_exits_config_error(monkeypatch, capsys):
    monkeypatch.delenv("HETZNER_API_TOKEN", raising=False)
    code = cli.main(["set", "asr-small", "b02", "--apply"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_CONFIG
    assert "HETZNER_API_TOKEN" in captured.err


# --------------------------------------------------------------------------- #
# 13: API errors are reported understandably
# --------------------------------------------------------------------------- #
def test_api_error_is_reported_with_code(monkeypatch, capsys):
    fake = FakeHetznerAPI()
    fake.fail_once((500, {"error": {"code": "service_error", "message": "backend exploded"}}))
    code, _out, err = run_cli(["list"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_API
    assert "backend exploded" in err
    assert "service_error" in err
    # no retry storm on 500: exactly one request was made
    assert len(fake.requests) == 1


def test_zone_not_found_is_config_error(monkeypatch, capsys):
    fake = FakeHetznerAPI(zone_exists=False)
    code, _out, err = run_cli(["list"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_CONFIG
    assert "Zone" in err


# --------------------------------------------------------------------------- #
# 14: rate limits and transient network errors are handled
# --------------------------------------------------------------------------- #
def test_rate_limit_is_retried_for_reads():
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}]}]
    )
    fake.fail_once((429, {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}))
    manager = make_manager(fake)
    records = manager.list_service_records()
    assert len(records) == len(SMALL_SERVICES)
    assert len(fake.requests) > 1


def test_network_error_is_retried_for_reads():
    fake = FakeHetznerAPI()
    fake.fail_once(httpx.ConnectError("connection refused"))
    manager = make_manager(fake)
    manager.list_service_records()
    assert any(r["path"].endswith("/rrsets") for r in fake.requests)


def test_ambiguous_write_errors_are_not_retried():
    fake = FakeHetznerAPI()
    fake.fail_next_matching("POST", httpx.ReadTimeout("read timed out"))
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_FAILED
    assert "timed out" in change.message.lower() or "fehlgeschlagen" in change.message.lower()
    creates = [r for r in fake.write_requests if r["path"].endswith("/rrsets")]
    assert len(creates) == 1  # exactly one attempt, no uncontrolled repeat


def test_safe_write_errors_are_retried():
    fake = FakeHetznerAPI()
    # connection was never established -> provably unprocessed -> retry is safe
    fake.fail_next_matching("POST", httpx.ConnectError("connection refused"))
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_CREATED
    creates = [r for r in fake.write_requests if r["path"].endswith("/rrsets")]
    assert len(creates) == 2  # first attempt failed at connect level, second succeeded


def test_rate_limit_error_after_retries_is_a_failure():
    fake = FakeHetznerAPI()
    for _ in range(4):  # more than max_retries
        fake.fail_once((429, {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}))
    manager = make_manager(fake)
    with pytest.raises(APIError) as excinfo:
        manager.list_service_records()
    assert excinfo.value.code == "rate_limit_exceeded"


# --------------------------------------------------------------------------- #
# 15: asynchronous Hetzner actions are awaited
# --------------------------------------------------------------------------- #
def test_async_action_is_awaited():
    fake = FakeHetznerAPI(async_actions=True)
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_CREATED
    assert change.applied is True
    action_polls = [r for r in fake.requests if "/actions/" in r["path"]]
    assert action_polls, "async action must be polled"
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."


def test_async_action_error_is_reported_as_failure():
    fake = FakeHetznerAPI(async_actions=True, action_error=("resource_unavailable", "zone locked"))
    manager = make_manager(fake)
    change = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert change.status == STATUS_FAILED
    assert "zone locked" in change.message


def test_action_failure_raises_structured_error():
    fake = FakeHetznerAPI(action_error=("conflict", "another action is running"))
    client = make_client(fake)
    with pytest.raises(ActionFailedError) as excinfo:
        client.create_rrset(fake.zone, "asr-small.api.mybister.com", "CNAME", ttl=60, records=[{"value": "b03.mybister.com"}])
    assert "another action is running" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 16: JSON output is valid
# --------------------------------------------------------------------------- #
def test_json_output_is_valid(monkeypatch, capsys):
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com."}], "ttl": 60}]
    )
    code, out, _err = run_cli(["list", "--json"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert {item["service_id"] for item in payload} == set(SMALL_SERVICES[i].id for i in range(len(SMALL_SERVICES)))
    assert all("fqdn" in item and "target" in item and "ttl" in item for item in payload)

    code, out, _err = run_cli(["get", "asr-small", "--json"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload == {
        "service_id": "asr-small",
        "fqdn": "asr-small.api.mybister.com",
        "type": "CNAME",
        "target": "b03.mybister.com.",
        "ttl": 60,
        "exists": True,
        "conflict_types": [],
        "conflict_targets": [],
        "expected_host": "b03.mybister.com.",
        "status": "present",
    }

    code, out, _err = run_cli(["bulk", "--host", "b03", "--json"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert set(payload["summary"]) == {"created", "updated", "unchanged", "conflict", "failed"}
    assert len(payload["results"]) == len(SMALL_SERVICES)
    assert payload["summary"]["unchanged"] == 1  # asr-small already points to b03

    code, out, _err = run_cli(["set", "asr-tiny", "b02", "--apply", "--json"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["status"] == STATUS_CREATED
    assert payload["applied"] is True
    assert payload["dry_run"] is False
    assert payload["target"] == "b02.mybister.com."  # plan target is absolute


def test_get_json_shows_raw_value_of_dotless_record(monkeypatch, capsys):
    """The stored value is shown verbatim so a missing dot stays visible."""
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}], "ttl": 60}]
    )
    code, out, _err = run_cli(["get", "asr-small", "--json"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["target"] == "b03.mybister.com"       # raw stored value (no dot)
    assert payload["expected_host"] == "b03.mybister.com."
    assert payload["status"] == "differs"                # -> bulk: conflict, needs --force


def test_get_json_for_missing_record(monkeypatch, capsys):
    fake = FakeHetznerAPI()
    code, out, _err = run_cli(["get", "asr-small", "--json"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_NOT_FOUND
    payload = json.loads(out)
    assert payload["exists"] is False
    assert payload["target"] is None


# --------------------------------------------------------------------------- #
# 17: repeated execution is idempotent
# --------------------------------------------------------------------------- #
def test_repeated_execution_is_idempotent():
    fake = FakeHetznerAPI()
    manager = make_manager(fake)

    first = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert first.status == STATUS_CREATED
    # initialize the remaining services as well
    manager.ensure_service_record("asr-tiny", "b03", apply=True)
    manager.ensure_service_record("sentiment", "b03", apply=True)
    writes_after_first = len(write_requests(fake))
    assert writes_after_first == 3

    second = manager.ensure_service_record("asr-small", "b03", apply=True)
    assert second.status == STATUS_UNCHANGED
    assert second.applied is False
    assert len(write_requests(fake)) == writes_after_first

    # a full bulk run after initialization must not write anything either
    results = manager.apply_bulk(manager.plan_bulk("b03"))
    assert all(r.status == STATUS_UNCHANGED for r in results)
    assert len(write_requests(fake)) == writes_after_first


# --------------------------------------------------------------------------- #
# 18: secrets never appear in logs or output
# --------------------------------------------------------------------------- #
def test_token_never_appears_in_logs_or_output(monkeypatch, capsys, caplog):
    caplog.set_level("DEBUG")
    fake = FakeHetznerAPI(token="a-different-token")
    code, out, err = run_cli(["list"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_API
    assert TOKEN not in out
    assert TOKEN not in err
    assert TOKEN not in caplog.text


def test_token_is_excluded_from_repr():
    client = HetznerCloudClient(TOKEN, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    try:
        assert TOKEN not in repr(client)
        config = DnsConfig.from_env(make_env())
        assert TOKEN not in repr(config)
        assert config.api_token == TOKEN
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# 19: service registry contains no duplicate ids
# --------------------------------------------------------------------------- #
def test_registry_contains_no_duplicate_ids():
    registry = ServiceRegistry.load()
    ids = registry.ids
    assert len(ids) == len(set(ids)), f"doppelte Service-IDs: {ids}"
    assert len(ids) >= 20

    with pytest.raises(ValidationError):
        ServiceRegistry(
            services=(
                ServiceDefinition(id="dup"),
                ServiceDefinition(id="dup"),
            )
        )


# --------------------------------------------------------------------------- #
# 20: registry matches the central service configuration (docker-compose.yml)
# --------------------------------------------------------------------------- #
def _compose_routers_with_host() -> set:
    """Router names with a Traefik ``Host(...)`` rule (production routers).

    Development routers are path-based (``PathPrefix`` without ``Host``) and
    are therefore not matched here. The public ``<service>-health`` routers
    (unauthenticated ``/health`` aliases of the same service) are skipped -
    they are not deployable services of their own.
    """
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    pattern = re.compile(r"traefik\.http\.routers\.([A-Za-z0-9_-]+)\.rule=Host\(`[^`]+`\)")
    names = set()
    for line in compose.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = pattern.search(line)
        if match and not match.group(1).endswith("-health"):
            names.add(match.group(1))
    return names


def test_registry_matches_docker_compose_services():
    registry = ServiceRegistry.load()
    routers = _compose_routers_with_host()
    # callback-tester is a webhook test helper and intentionally excluded.
    managed_routers = routers - {"callback-tester"}

    unknown_in_registry = set(registry.ids) - managed_routers
    missing_in_registry = managed_routers - set(registry.ids)
    assert not unknown_in_registry, f"Registry-IDs ohne deploybaren Compose-Service: {sorted(unknown_in_registry)}"
    assert not missing_in_registry, (
        "Compose-Services ohne Registry-Eintrag (neuen Service in "
        f"tools/service_registry.json ergaenzen): {sorted(missing_in_registry)}"
    )
    assert routers - set(registry.ids) == {"callback-tester"}


def test_prod_routers_use_dedicated_service_hosts():
    """Every production router must use its own <service-id>.api.mybister.com host.

    The former shared hosts (ask.mybister.com, ask.vereinstalk.de) and the
    PathPrefix routing were removed by the HTTPS conversion; development
    routers stay path-based and carry no Host rule at all.
    """
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    pattern = re.compile(r"traefik\.http\.routers\.([A-Za-z0-9_-]+)\.rule=Host\(`([^`]+)`\)")
    checked = 0
    for line in compose.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = pattern.search(line)
        if not match:
            continue
        name, host = match.groups()
        # public <service>-health routers share their service's hostname
        service_id = name[: -len("-health")] if name.endswith("-health") else name
        assert host == f"{service_id}.api.mybister.com", (
            f"Router {name} nutzt noch den Host {host!r} statt "
            f"{service_id}.api.mybister.com"
        )
        checked += 1
    assert checked >= 21
    assert "ask.mybister.com" not in compose
    assert "ask.vereinstalk.de" not in compose


def test_registry_excludes_documented_components():
    registry = ServiceRegistry.load()
    excluded_ids = {excluded.id for excluded in registry.excluded}
    assert {"callback-tester", "ocr-worker", "apikey-auth"} <= excluded_ids
    for service_id in ("ocr-worker", "apikey-auth", "callback-tester", "gateway"):
        assert service_id not in registry.ids


# --------------------------------------------------------------------------- #
# additional safety / behaviour tests
# --------------------------------------------------------------------------- #
def test_bulk_cli_dry_run_and_apply(monkeypatch, capsys):
    fake = FakeHetznerAPI()
    code, out, _err = run_cli(["bulk", "--host", "b03"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    assert "CREATE" in out
    assert "DRY RUN - No DNS changes applied." in out
    assert write_requests(fake) == []

    code, out, _err = run_cli(["bulk", "--host", "b03", "--apply"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    assert "Applied." in out
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."


def test_bulk_cli_conflict_exit_code(monkeypatch, capsys):
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b01.mybister.com"}]}]
    )
    code, out, _err = run_cli(["bulk", "--host", "b03", "--apply"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_CONFLICT
    assert "CONFLICT" in out
    assert fake.cname_target("asr-small.api.mybister.com") == "b01.mybister.com"

    code, out, _err = run_cli(["bulk", "--host", "b03", "--apply", "--force"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    assert fake.cname_target("asr-small.api.mybister.com") == "b03.mybister.com."


def test_set_conflict_with_foreign_record_exits_conflict(monkeypatch, capsys):
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "A", "records": [{"value": "203.0.113.10"}]}]
    )
    code, out, _err = run_cli(["set", "asr-small", "b03", "--apply"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_CONFLICT
    assert "CONFLICT" in out.upper() or "Konflikt" in out
    assert write_requests(fake) == []


def test_get_missing_record_exit_code(monkeypatch, capsys):
    fake = FakeHetznerAPI()
    code, out, _err = run_cli(["get", "asr-small"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_NOT_FOUND
    assert "nicht angelegt" in out


def test_list_shows_expected_host_and_status(monkeypatch, capsys):
    fake = FakeHetznerAPI(
        rrsets=[
            {"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b01.mybister.com"}], "ttl": 60}
        ]
    )
    code, out, _err = run_cli(["list"], fake, monkeypatch, capsys)
    assert code == cli.EXIT_OK
    assert "EXPECTED" in out
    assert "b03.mybister.com." in out  # expected host is reported absolute
    assert "DIFFERS" in out
    assert "MISSING" in out


def test_pagination_is_followed():
    rrsets = [
        {"name": f"svc{i}.api.mybister.com", "type": "CNAME", "records": [{"value": "b03.mybister.com"}], "ttl": 60}
        for i in range(12)
    ]
    fake = FakeHetznerAPI(rrsets=rrsets, page_size=5)
    manager = make_manager(fake)
    rrsets_found = manager._client.list_rrsets(manager.zone)
    assert len(rrsets_found) == 12
    pages = [r["query"].get("page") for r in fake.requests if r["path"].endswith("/rrsets")]
    assert pages == ["1", "2", "3"]


def test_import_has_no_side_effects(monkeypatch):
    """Importing the package must not read env vars or open sockets."""
    import importlib

    import tools.hetzner_dns as pkg
    import tools.hetzner_dns.cli as cli_mod
    import tools.hetzner_dns.config as config_mod
    import tools.hetzner_dns.manager as manager_mod

    created = []
    original_init = httpx.Client.__init__

    def tracking_init(self, *args, **kwargs):
        created.append(1)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", tracking_init)
    monkeypatch.setattr(config_mod.os, "environ", {"HETZNER_API_TOKEN": "should-not-be-read"})
    for module in (pkg, config_mod, manager_mod, cli_mod):
        importlib.reload(module)
    assert created == []


def test_no_write_endpoints_on_dry_run(monkeypatch, capsys):
    fake = FakeHetznerAPI(
        rrsets=[{"name": "asr-small.api.mybister.com", "type": "CNAME", "records": [{"value": "b01.mybister.com"}]}]
    )
    run_cli(["set", "asr-small", "b03"], fake, monkeypatch, capsys)
    run_cli(["bulk", "--host", "b03"], fake, monkeypatch, capsys)
    assert write_requests(fake) == []
    # only GET requests were used
    assert {r["method"] for r in fake.requests} == {"GET"}
