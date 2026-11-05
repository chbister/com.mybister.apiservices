"""Hetzner Cloud DNS management for API service CNAME records.

Manages records of the form::

    <service-id>.api.mybister.com  CNAME  <host>.mybister.com

against the Hetzner Cloud API (``https://api.hetzner.cloud/v1``).

Library usage (no CLI/subprocess needed)::

    from tools.hetzner_dns import HetznerDNSManager

    manager = HetznerDNSManager.from_env()
    manager.list_service_records()
    manager.ensure_service_record("asr-small", "b02", apply=True)

Command line usage (from the repository root)::

    python -m tools.hetzner_dns list
    python -m tools.hetzner_dns get asr-small
    python -m tools.hetzner_dns set asr-small b02 --apply
    python -m tools.hetzner_dns bulk --host b03 --apply

Importing this package has no side effects: no environment variables are
read and no network requests are made until a manager/client is built.
"""

from .config import DnsConfig
from .errors import (
    ActionFailedError,
    ActionTimeoutError,
    APIError,
    ConfigError,
    HetznerDNSError,
    MissingTokenError,
    NotFoundError,
    RateLimitError,
    TransportError,
    ValidationError,
    ZoneNotFoundError,
)
from .manager import (
    ALL_STATUSES,
    CHANGE_POLICIES,
    ChangeResult,
    HetznerDNSManager,
    ServiceRecord,
    summarize,
)
from .registry import ServiceDefinition, ServiceRegistry

__all__ = [
    "ALL_STATUSES",
    "CHANGE_POLICIES",
    "ActionFailedError",
    "ActionTimeoutError",
    "APIError",
    "ChangeResult",
    "ConfigError",
    "DnsConfig",
    "HetznerDNSManager",
    "HetznerDNSError",
    "MissingTokenError",
    "NotFoundError",
    "RateLimitError",
    "ServiceDefinition",
    "ServiceRecord",
    "ServiceRegistry",
    "TransportError",
    "ValidationError",
    "ZoneNotFoundError",
    "summarize",
]
