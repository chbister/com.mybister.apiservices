"""Validation helpers for DNS names, service ids and target hosts."""

from __future__ import annotations

import re
from typing import Optional

from .errors import ValidationError

# A single DNS label: 1-63 chars, lower-case alphanumerics and hyphens,
# no leading/trailing hyphen.
_LABEL_RE = re.compile(r"^(?!-)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

_MAX_NAME_LEN = 253


def is_valid_label(value: str) -> bool:
    """Return ``True`` if *value* is a syntactically valid DNS label."""
    return bool(value) and bool(_LABEL_RE.match(value))


def validate_domain(name: str, what: str = "Domain") -> str:
    """Validate and normalize a fully qualified DNS name (no trailing dot)."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{what} darf nicht leer sein")
    normalized = name.strip().lower().rstrip(".")
    if len(normalized) > _MAX_NAME_LEN:
        raise ValidationError(f"{what} '{name}' ist zu lang (max. {_MAX_NAME_LEN} Zeichen)")
    labels = normalized.split(".")
    if len(labels) < 2:
        raise ValidationError(f"{what} '{name}' muss vollstaendig qualifiziert sein (z. B. b03.mybister.com)")
    for label in labels:
        if not is_valid_label(label):
            raise ValidationError(f"{what} '{name}' enthaelt ungueltiges Label '{label}'")
    return normalized


def validate_service_id(service_id: str) -> str:
    """Validate a service id (a single DNS label, e.g. ``asr-small``)."""
    if not isinstance(service_id, str) or not service_id.strip():
        raise ValidationError("Service-ID darf nicht leer sein")
    normalized = service_id.strip().lower()
    if not is_valid_label(normalized):
        raise ValidationError(
            f"Ungueltige Service-ID '{service_id}' "
            "(nur a-z, 0-9 und Bindestriche, max. 63 Zeichen, kein Bindestrich am Anfang/Ende)"
        )
    return normalized


def validate_zone(zone: str, what: str = "HETZNER_DNS_ZONE") -> str:
    """Validate the DNS zone name."""
    return validate_domain(zone, what)


def to_absolute(name: str) -> str:
    """Return the root-anchored form of a DNS name: a trailing dot.

    CNAME values in the Hetzner Cloud API (and in DNS wire format in
    general) must be absolute, e.g. ``b03.mybister.com.``. Values without
    a trailing dot are accepted on read for tolerance, but everything
    this tool writes or reports as a CNAME value is absolute.
    """
    value = (name or "").strip()
    if not value:
        return value
    return value if value.endswith(".") else value + "."


def same_name(left: Optional[str], right: Optional[str]) -> bool:
    """Compare two DNS names ignoring case and an optional trailing dot."""
    def bare(value: Optional[str]) -> str:
        return (value or "").strip().rstrip(".").lower()

    return bare(left) == bare(right)


def normalize_target_host(host: str, *, zone: str, service_domain: str, what: str = "Zielhost") -> str:
    """Normalize a target host to an FQDN inside *zone*.

    Accepts a short host name (``b02`` -> ``b02.<zone>``) or a full FQDN.
    Rejects anything outside the configured zone and anything inside the
    service domain (a CNAME pointing into the service domain could create
    a chain/loop).
    """
    if not isinstance(host, str) or not host.strip():
        raise ValidationError(f"{what} darf nicht leer sein")
    value = host.strip().lower().rstrip(".")
    if "." not in value:
        value = f"{value}.{zone}"
    fqdn = validate_domain(value, what)
    if fqdn != zone and not fqdn.endswith("." + zone):
        raise ValidationError(f"{what} '{host}' liegt ausserhalb der DNS-Zone '{zone}'")
    if fqdn == service_domain or fqdn.endswith("." + service_domain):
        raise ValidationError(
            f"{what} '{host}' liegt innerhalb der Service-Domain '{service_domain}' "
            "(CNAME-Ziele duerfen nicht auf die Service-Domain zeigen)"
        )
    return fqdn
