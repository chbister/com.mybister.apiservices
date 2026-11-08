#!/usr/bin/env python3
"""Compose service names for one deployment host (standard library only).

The deployment workflow uses this to start exactly the containers of a
host. The placement comes from the registry ``tools/service_registry.json``
(single source of truth): every ``services`` entry (DNS-managed) and every
``components`` entry (deployable, no DNS) assigned to the host, where
``host: "*"`` means "runs on every host".

No third-party dependencies on purpose - the GitHub runners have no
``.venv``/``httpx`` and only need ``python3``.

Usage (from the repository root)::

    python3 tools/deploy_targets.py b03            # names for b03 (space separated)
    python3 tools/deploy_targets.py b03 --json     # ... as JSON array
    python3 tools/deploy_targets.py b03 --others   # names NOT on b03 (prune candidates)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.hetzner_dns.registry import ServiceRegistry  # noqa: E402  (path bootstrap)


def _host_matches(assignment: str, wanted: str) -> bool:
    """``*`` matches everything; short names match their FQDNs and vice versa."""
    if assignment == "*":
        return True
    a = (assignment or "").strip().lower().rstrip(".")
    w = (wanted or "").strip().lower().rstrip(".")
    return bool(a) and bool(w) and (a == w or w.startswith(a + ".") or a.startswith(w + "."))


def _compose_name(entry) -> str:
    return entry.compose_service or entry.id


def compose_services_for(
    host: str, *, registry: Optional[ServiceRegistry] = None
) -> List[str]:
    """Compose service names assigned to *host* (registry order)."""
    registry = registry or ServiceRegistry.load()
    entries = list(registry.services) + list(registry.components)
    return [_compose_name(entry) for entry in entries if _host_matches(entry.host, host)]


def other_services(host: str, *, registry: Optional[ServiceRegistry] = None) -> List[str]:
    """Compose service names NOT assigned to *host* (prune candidates)."""
    registry = registry or ServiceRegistry.load()
    entries = list(registry.services) + list(registry.components)
    return [_compose_name(entry) for entry in entries if not _host_matches(entry.host, host)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deploy_targets",
        description="Compose-Services eines Deploy-Hosts aus der Service-Registry",
    )
    parser.add_argument("host", help="Ziel-Host, z. B. b03 oder b03.mybister.com")
    parser.add_argument(
        "--others",
        action="store_true",
        help="nicht zugewiesene Services ausgeben (Prune-Kandidaten) statt der eigenen",
    )
    parser.add_argument("--json", action="store_true", help="JSON-Array statt Leerzeichenliste")
    args = parser.parse_args(argv)

    registry = ServiceRegistry.load()
    wanted = args.host.strip()
    if not wanted:
        print("deploy_targets: host darf nicht leer sein", file=sys.stderr)
        return 2

    names = other_services(wanted, registry=registry) if args.others else compose_services_for(
        wanted, registry=registry
    )
    entries = list(registry.services) + list(registry.components)
    explicitly_assigned = any(
        entry.host and entry.host != "*" and _host_matches(entry.host, wanted) for entry in entries
    )
    if not explicitly_assigned and not args.others:
        known = sorted({entry.host for entry in entries if entry.host and entry.host != "*"})
        print(
            f"deploy_targets: keine Services fuer Host '{wanted}' zugewiesen "
            f"(bekannt: {', '.join(known)})",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(names))
    else:
        print(" ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
