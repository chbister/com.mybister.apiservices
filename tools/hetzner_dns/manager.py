"""Business logic: plan and apply CNAME records for API services.

This module is the reusable library part (no CLI, no subprocesses). It is
safe to import: there are no global side effects and no hidden DNS
changes. Every write happens only through the ``ensure_*``/``apply_*``
methods and only after an explicit ``apply=True``.

Typical usage::

    from tools.hetzner_dns import DnsConfig, HetznerDNSManager, ServiceRegistry

    config = DnsConfig.from_env()          # reads os.environ
    registry = ServiceRegistry.load()      # tools/service_registry.json
    manager = HetznerDNSManager.from_env() # or construct directly

    manager.list_service_records()
    manager.get_service_record("asr-small")
    manager.plan_service_record("asr-small", "b02")
    manager.ensure_service_record("asr-small", "b02", apply=True)
    manager.plan_bulk("b03")
    manager.apply_bulk(plan)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .client import HetznerCloudClient
from .config import DnsConfig
from .errors import (
    APIError,
    HetznerDNSError,
    ValidationError,
)
from .registry import ServiceDefinition, ServiceRegistry
from .validators import (
    absolute_in_zone,
    normalize_target_host,
    same_name,
    to_absolute,
    to_relative,
    validate_service_id,
)

logger = logging.getLogger("tools.hetzner_dns.manager")

# Status vocabulary shared by planning and applying.
STATUS_CREATED = "created"
STATUS_UPDATED = "updated"
STATUS_UNCHANGED = "unchanged"
STATUS_CONFLICT = "conflict"
STATUS_FAILED = "failed"

ALL_STATUSES = (STATUS_CREATED, STATUS_UPDATED, STATUS_UNCHANGED, STATUS_CONFLICT, STATUS_FAILED)

# Policy for records whose current target differs from the desired one.
POLICY_OVERWRITE = "overwrite"  # single "set": always update
POLICY_FORCE = "force"          # bulk with --force: update
POLICY_KEEP = "keep"            # bulk without --force: report conflict

CHANGE_POLICIES = (POLICY_OVERWRITE, POLICY_FORCE, POLICY_KEEP)


@dataclass
class ServiceRecord:
    """Current DNS state of one service."""

    service_id: str
    fqdn: str
    type: str = "CNAME"
    exists: bool = False
    target: Optional[str] = None
    ttl: Optional[int] = None
    conflict_types: List[str] = field(default_factory=list)
    conflict_targets: List[str] = field(default_factory=list)
    expected_host: Optional[str] = None

    @property
    def status(self) -> str:
        if self.conflict_types:
            return STATUS_CONFLICT
        if not self.exists:
            return "missing"
        # Exact comparison on purpose: a stored value without the trailing
        # dot ("b03.mybister.com") differs from the expected absolute value
        # ("b03.mybister.com.") and must be reported so it gets rewritten.
        if self.expected_host and self.target != self.expected_host:
            return "differs"
        return "present"

    def to_dict(self) -> Dict[str, object]:
        return {
            "service_id": self.service_id,
            "fqdn": self.fqdn,
            "type": self.type if self.exists or not self.conflict_types else self.conflict_types[0],
            "target": self.target,
            "ttl": self.ttl,
            "exists": self.exists,
            "conflict_types": list(self.conflict_types),
            "conflict_targets": list(self.conflict_targets),
            "expected_host": self.expected_host,
            "status": self.status,
        }


@dataclass
class ChangeResult:
    """Planned or applied change of one service CNAME."""

    service_id: str
    fqdn: str
    status: str
    target: str
    current_target: Optional[str] = None
    ttl: Optional[int] = None
    applied: bool = False
    message: str = ""
    policy: str = POLICY_OVERWRITE

    @property
    def verb(self) -> str:
        return {
            STATUS_CREATED: "CREATE",
            STATUS_UPDATED: "UPDATE",
            STATUS_UNCHANGED: "SKIP",
            STATUS_CONFLICT: "CONFLICT",
            STATUS_FAILED: "ERROR",
        }.get(self.status, self.status.upper())

    def to_dict(self) -> Dict[str, object]:
        return {
            "service_id": self.service_id,
            "fqdn": self.fqdn,
            "type": "CNAME",
            "status": self.status,
            "target": self.target,
            "current_target": self.current_target,
            "ttl": self.ttl,
            "applied": self.applied,
            "message": self.message,
            "policy": self.policy,
        }


def summarize(results: Sequence[ChangeResult]) -> Dict[str, int]:
    """Count results per status (all five statuses are always present)."""
    counts = {status: 0 for status in ALL_STATUSES}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


class HetznerDNSManager:
    """High level, idempotent CNAME management for the service registry."""

    def __init__(self, config: DnsConfig, registry: ServiceRegistry, client: HetznerCloudClient) -> None:
        self.config = config
        self.registry = registry
        self._client = client
        self._zone: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        *,
        registry: Optional[ServiceRegistry] = None,
        client: Optional[HetznerCloudClient] = None,
    ) -> "HetznerDNSManager":
        """Build a manager from environment variables (no API call yet)."""
        config = DnsConfig.from_env(env)
        if registry is None:
            registry = ServiceRegistry.load(config.registry_path)
        if client is None:
            client = HetznerCloudClient(
                config.api_token or "",
                base_url=config.api_base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
                action_timeout=config.action_timeout,
                action_poll_interval=config.action_poll_interval,
            )
        return cls(config, registry, client)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def zone(self) -> str:
        """Zone id (resolved once by name, then reused for all calls)."""
        if self._zone is None:
            identifier = self.config.zone_id or self.config.zone
            zone = self._client.get_zone(identifier)
            name = str(zone.get("name", ""))
            if self.config.zone_id is None and name != self.config.zone:
                raise ValidationError(
                    f"Zone '{name}' passt nicht zu HETZNER_DNS_ZONE '{self.config.zone}'"
                )
            self._zone = str(zone.get("id") or name)
        return self._zone

    def fqdn(self, service_id: str) -> str:
        """Return the FQDN of a service (validates the id against the registry)."""
        self.registry.get(service_id)
        return f"{validate_service_id(service_id)}.{self.config.service_domain}"

    def api_name(self, fqdn: str) -> str:
        """Zone-relative form of *fqdn* as expected by the Hetzner API.

        The API appends the zone name to every RR set name it receives,
        so ``asr-small.api.mybister.com`` goes over the wire as
        ``asr-small.api`` (zone ``mybister.com``). Sending the FQDN would
        create the doubled record ``asr-small.api.mybister.com.mybister.com``.
        """
        return to_relative(fqdn, self.config.zone)

    def _expected_host_for(self, service: ServiceDefinition) -> Optional[str]:
        """Assigned target host of *service* (registry ``host`` field).

        Falls back to ``DEFAULT_DOCKER_HOST`` for services without an
        assignment. Always absolute (trailing dot), like CNAME values.
        """
        if service.host:
            return to_absolute(self.normalize_host(service.host))
        return to_absolute(self.config.default_docker_host) if self.config.default_docker_host else None

    def normalize_host(self, host: str) -> str:
        """Normalize a short host name or FQDN to a target inside the zone."""
        return normalize_target_host(
            host,
            zone=self.config.zone,
            service_domain=self.config.service_domain,
            what="Zielhost",
        )

    def _index_by_name(self, rrsets: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
        index: Dict[str, Dict[str, Dict[str, object]]] = {}
        for rrset in rrsets:
            # Tolerate both name forms on read: the API may return names
            # relative to the zone or already fully qualified.
            name = absolute_in_zone(str(rrset.get("name", "")), self.config.zone)
            rtype = str(rrset.get("type", "")).upper()
            index.setdefault(name, {})[rtype] = rrset
        return index

    @staticmethod
    def _target_of(rrset: Optional[Dict[str, object]]) -> Optional[str]:
        if not rrset:
            return None
        records = rrset.get("records") or []
        if not records:
            return None
        first = records[0]
        value = str(first.get("value", "")).strip().lower() if isinstance(first, dict) else str(first).strip().lower()
        # The stored value is reported verbatim (only case/space normalized).
        # A missing trailing dot is a real difference that has to be fixed by
        # an explicit write, so it must stay visible here - never normalize
        # it away, or dotless records would look "unchanged" forever.
        return value or None

    @staticmethod
    def _all_targets(rrset: Dict[str, object]) -> List[str]:
        values: List[str] = []
        for record in rrset.get("records") or []:
            if isinstance(record, dict):
                value = str(record.get("value", ""))
            else:
                value = str(record)
            value = value.strip().lower()
            if value:
                values.append(value)
        return values

    def _record_from_index(self, service_id: str, index: Dict[str, Dict[str, Dict[str, object]]]) -> ServiceRecord:
        fqdn = self.fqdn(service_id)
        by_type = index.get(absolute_in_zone(fqdn, self.config.zone), {})
        cname = by_type.get("CNAME")
        conflict_types = sorted(t for t in by_type if t != "CNAME")
        conflict_targets: List[str] = []
        for rtype in conflict_types:
            conflict_targets.extend(self._all_targets(by_type[rtype]))
        return ServiceRecord(
            service_id=validate_service_id(service_id),
            fqdn=fqdn,
            exists=cname is not None,
            target=self._target_of(cname),
            ttl=int(cname.get("ttl")) if cname is not None and cname.get("ttl") is not None else None,
            conflict_types=conflict_types,
            conflict_targets=conflict_targets,
            expected_host=self._expected_host_for(self.registry.get(service_id)),
        )

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def list_service_records(self) -> List[ServiceRecord]:
        """Current state of every service in the registry (1-2 API calls)."""
        rrsets = self._client.list_rrsets(self.zone)
        index = self._index_by_name(rrsets)
        return [self._record_from_index(service.id, index) for service in self.registry]

    def get_service_record(self, service_id: str) -> ServiceRecord:
        """Current state of a single service (``exists=False`` if not created yet)."""
        self.fqdn(service_id)  # validates the id
        # Fetch the whole (small) zone and match locally: no name goes over
        # the wire on reads, so they stay correct regardless of how the API
        # resolves name filters - names are zone-relative everywhere else.
        rrsets = self._client.list_rrsets(self.zone)
        index = self._index_by_name(rrsets)
        return self._record_from_index(service_id, index)

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def _plan(
        self,
        service_id: str,
        host: str,
        *,
        policy: str,
        record: Optional[ServiceRecord] = None,
    ) -> ChangeResult:
        if policy not in CHANGE_POLICIES:
            raise ValidationError(f"Unbekannte Ueberschreib-Richtlinie '{policy}'")
        normalized_id = validate_service_id(service_id)
        fqdn = self.fqdn(normalized_id)
        # The desired CNAME value is absolute (trailing dot), e.g. b03.mybister.com.
        desired = to_absolute(self.normalize_host(host))
        if same_name(desired, fqdn):
            raise ValidationError(
                f"Zielhost darf nicht dem eigenen DNS-Namen '{fqdn}' entsprechen"
            )
        if record is None:
            record = self.get_service_record(normalized_id)

        if record.conflict_types:
            return ChangeResult(
                service_id=normalized_id,
                fqdn=fqdn,
                status=STATUS_CONFLICT,
                target=desired,
                current_target=record.target,
                ttl=self.config.ttl,
                message=(
                    f"Fuer '{fqdn}' existiert bereits ein Record vom Typ "
                    f"{', '.join(record.conflict_types)} - dieser wird nicht veraendert"
                ),
                policy=policy,
            )
        if not record.exists:
            return ChangeResult(
                service_id=normalized_id,
                fqdn=fqdn,
                status=STATUS_CREATED,
                target=desired,
                current_target=None,
                ttl=self.config.ttl,
                message="CNAME fehlt und wird erstellt",
                policy=policy,
            )
        # Exact comparison: the stored value must match the desired absolute
        # value character by character - a missing trailing dot counts as a
        # divergence here and is only fixed with an explicit write.
        current = (record.target or "").lower()
        if current == desired:
            return ChangeResult(
                service_id=normalized_id,
                fqdn=fqdn,
                status=STATUS_UNCHANGED,
                target=desired,
                current_target=current,
                ttl=record.ttl,
                message="CNAME ist bereits korrekt",
                policy=policy,
            )
        if policy == POLICY_KEEP:
            return ChangeResult(
                service_id=normalized_id,
                fqdn=fqdn,
                status=STATUS_CONFLICT,
                target=desired,
                current_target=current,
                ttl=record.ttl,
                message=(
                    f"CNAME zeigt aktuell auf '{current}' statt '{desired}' - "
                    "nicht ueberschrieben (Bulk ohne --force)"
                ),
                policy=policy,
            )
        return ChangeResult(
            service_id=normalized_id,
            fqdn=fqdn,
            status=STATUS_UPDATED,
            target=desired,
            current_target=current,
            ttl=self.config.ttl,
            message=f"CNAME wird von '{current}' auf '{desired}' umgestellt",
            policy=policy,
        )

    def plan_service_record(self, service_id: str, host: str, *, force: bool = False) -> ChangeResult:
        """Plan a single service change (no write)."""
        policy = POLICY_FORCE if force else POLICY_OVERWRITE
        return self._plan(service_id, host, policy=policy)

    def plan_bulk(self, host: str, *, force: bool = False) -> List[ChangeResult]:
        """Plan changes for all registry services (no write).

        Without ``force`` existing CNAMEs with a different target are
        reported as ``conflict`` and are never overwritten.
        """
        normalized_host = self.normalize_host(host)
        policy = POLICY_FORCE if force else POLICY_KEEP
        rrsets = self._client.list_rrsets(self.zone)
        index = self._index_by_name(rrsets)
        plan: List[ChangeResult] = []
        for service in self.registry:
            record = self._record_from_index(service.id, index)
            try:
                plan.append(self._plan(service.id, normalized_host, policy=policy, record=record))
            except HetznerDNSError as exc:
                plan.append(
                    ChangeResult(
                        service_id=service.id,
                        fqdn=f"{service.id}.{self.config.service_domain}",
                        status=STATUS_FAILED,
                        target=to_absolute(normalized_host),
                        message=str(exc),
                        policy=policy,
                    )
                )
        return plan

    def plan_sync(self, *, force: bool = False) -> List[ChangeResult]:
        """Plan the reconcile of every service with its registry ``host``.

        Each service is compared against the target host assigned in the
        registry (the deployment distribution): missing CNAMEs are planned
        as ``created``, diverging ones as ``updated``. Services without an
        assignment are reported as ``failed`` - add a ``host`` field for
        them. With ``apply`` on :meth:`apply_bulk` the plan is written.
        """
        policy = POLICY_FORCE if force else POLICY_OVERWRITE
        rrsets = self._client.list_rrsets(self.zone)
        index = self._index_by_name(rrsets)
        plan: List[ChangeResult] = []
        for service in self.registry:
            record = self._record_from_index(service.id, index)
            if not service.host:
                plan.append(
                    ChangeResult(
                        service_id=service.id,
                        fqdn=self.fqdn(service.id),
                        status=STATUS_FAILED,
                        target="",
                        current_target=record.target,
                        ttl=record.ttl,
                        applied=False,
                        message=(
                            f"Kein Ziel-Host fuer '{service.id}' in der Registry "
                            "(Feld 'host' ergaenzen)"
                        ),
                        policy=policy,
                    )
                )
                continue
            try:
                plan.append(self._plan(service.id, service.host, policy=policy, record=record))
            except HetznerDNSError as exc:
                plan.append(
                    ChangeResult(
                        service_id=service.id,
                        fqdn=self.fqdn(service.id),
                        status=STATUS_FAILED,
                        target="",
                        applied=False,
                        message=str(exc),
                        policy=policy,
                    )
                )
        return plan

    # ------------------------------------------------------------------ #
    # applying
    # ------------------------------------------------------------------ #
    def ensure_service_record(
        self, service_id: str, host: str, *, apply: bool = True, force: bool = False
    ) -> ChangeResult:
        """Plan (and optionally apply) one service CNAME.

        With ``apply=False`` nothing is written. With ``apply=True`` the
        current state is re-checked immediately before the write to avoid
        clobbering concurrent changes (best effort - see docs/hetzner-dns.md
        for the remaining race conditions).
        """
        policy = POLICY_FORCE if force else POLICY_OVERWRITE
        plan = self._plan(service_id, host, policy=policy)
        if not apply:
            return plan

        if plan.status in (STATUS_UNCHANGED, STATUS_CONFLICT, STATUS_FAILED):
            return plan

        # Re-check the current state right before writing.
        fresh = self.get_service_record(plan.service_id)
        plan = self._plan(plan.service_id, host, policy=policy, record=fresh)
        if plan.status in (STATUS_UNCHANGED, STATUS_CONFLICT, STATUS_FAILED):
            logger.info("%s: Zustand geaendert, Ergebnis: %s", plan.fqdn, plan.status)
            return plan

        try:
            return self._write(plan, fresh)
        except APIError as exc:
            logger.warning("%s: Schreibfehler: %s", plan.fqdn, exc)
            return ChangeResult(
                service_id=plan.service_id,
                fqdn=plan.fqdn,
                status=STATUS_FAILED,
                target=plan.target,
                current_target=plan.current_target,
                ttl=plan.ttl,
                applied=False,
                message=str(exc),
                policy=plan.policy,
            )

    def _write(self, plan: ChangeResult, current: ServiceRecord) -> ChangeResult:
        zone = self.zone
        fqdn = plan.fqdn
        if plan.status == STATUS_CREATED:
            logger.info("%s: erstelle CNAME -> %s", fqdn, plan.target)
            # The wire name must be zone-relative ("asr-small.api", see
            # api_name()); the wire value must be absolute ("b03.mybister.com.").
            rrset = self._client.create_rrset(
                zone,
                self.api_name(fqdn),
                "CNAME",
                ttl=self.config.ttl,
                records=[{"value": to_absolute(plan.target)}],
            )
            target = self._target_of(rrset) or plan.target
            ttl = int(rrset.get("ttl")) if rrset.get("ttl") is not None else self.config.ttl
            return ChangeResult(
                service_id=plan.service_id,
                fqdn=fqdn,
                status=STATUS_CREATED,
                target=target,
                current_target=None,
                ttl=ttl,
                applied=True,
                message=f"CNAME erstellt: {fqdn} -> {target}",
                policy=plan.policy,
            )

        if plan.status == STATUS_UPDATED:
            logger.info("%s: aktualisiere CNAME %s -> %s", fqdn, plan.current_target, plan.target)
            # The wire name must be zone-relative; the wire value absolute.
            self._client.set_rrset_records(
                zone, self.api_name(fqdn), "CNAME", records=[{"value": to_absolute(plan.target)}]
            )
            ttl = current.ttl
            if self.config.ttl and ttl != self.config.ttl:
                self._client.change_rrset_ttl(zone, self.api_name(fqdn), "CNAME", ttl=self.config.ttl)
                ttl = self.config.ttl
            return ChangeResult(
                service_id=plan.service_id,
                fqdn=fqdn,
                status=STATUS_UPDATED,
                target=plan.target,
                current_target=plan.current_target,
                ttl=ttl,
                applied=True,
                message=f"CNAME aktualisiert: {plan.current_target} -> {plan.target}",
                policy=plan.policy,
            )

        return plan  # pragma: no cover - guarded by caller

    def apply_bulk(self, plan: Sequence[ChangeResult]) -> List[ChangeResult]:
        """Apply a plan created by :meth:`plan_bulk` (idempotent, per-item).

        Items with status ``unchanged``/``conflict`` are passed through
        unchanged; failures of single items do not abort the bulk run.
        """
        results: List[ChangeResult] = []
        for change in plan:
            if change.status != STATUS_CREATED and change.status != STATUS_UPDATED:
                results.append(change)
                continue
            try:
                results.append(
                    self.ensure_service_record(
                        change.service_id,
                        change.target,
                        apply=True,
                        force=change.policy == POLICY_FORCE,
                    )
                )
            except HetznerDNSError as exc:
                logger.warning("%s: Bulk-Fehler: %s", change.fqdn, exc)
                results.append(
                    ChangeResult(
                        service_id=change.service_id,
                        fqdn=change.fqdn,
                        status=STATUS_FAILED,
                        target=change.target,
                        current_target=change.current_target,
                        ttl=change.ttl,
                        applied=False,
                        message=str(exc),
                        policy=change.policy,
                    )
                )
        return results
