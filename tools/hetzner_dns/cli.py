"""Command line interface for the Hetzner DNS tool.

Run from the repository root::

    python -m tools.hetzner_dns list
    python -m tools.hetzner_dns get asr-small
    python -m tools.hetzner_dns set asr-small b02 --apply
    python -m tools.hetzner_dns bulk --host b03 --apply

Exit codes (for GitHub Actions):

====  =========================================================
Code  Meaning
====  =========================================================
0     Success
2     Usage/validation error (bad service id, host, arguments)
3     Configuration error (missing token, bad/missing zone)
4     API/network/action failure
5     Conflict (foreign record type or diverging bulk target)
6     DNS record not found (``get`` only)
====  =========================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .errors import (
    APIError,
    ConfigError,
    HetznerDNSError,
    ValidationError,
)
from .manager import (
    STATUS_CONFLICT,
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_UNCHANGED,
    STATUS_UPDATED,
    ChangeResult,
    HetznerDNSManager,
    summarize,
)
from .registry import ServiceRegistry

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_API = 4
EXIT_CONFLICT = 5
EXIT_NOT_FOUND = 6

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

STATUS_EXIT_CODES = {
    STATUS_CREATED: EXIT_OK,
    STATUS_UPDATED: EXIT_OK,
    STATUS_UNCHANGED: EXIT_OK,
    STATUS_CONFLICT: EXIT_CONFLICT,
    STATUS_FAILED: EXIT_API,
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine readable JSON output")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging (stderr)")

    parser = argparse.ArgumentParser(
        prog="python -m tools.hetzner_dns",
        description="Verwaltung der CNAME-Records der API-Services in der Hetzner-Cloud-DNS-Zone.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser(
        "list", parents=[common], help="aktuellen Stand aller Service-CNAMEs anzeigen"
    )
    p_list.set_defaults(func=cmd_list)

    p_get = subparsers.add_parser(
        "get", parents=[common], help="einen Service abfragen"
    )
    p_get.add_argument("service_id", help="Service-ID, z. B. asr-small")
    p_get.set_defaults(func=cmd_get)

    p_set = subparsers.add_parser(
        "set", parents=[common], help="einen Service-CNAME anlegen/aktualisieren (Standard: Dry-Run)"
    )
    p_set.add_argument("service_id", help="Service-ID, z. B. asr-small")
    p_set.add_argument("host", help="Ziel-Host, z. B. b02 oder b02.mybister.com")
    p_set.add_argument("--apply", action="store_true", help="Aenderung tatsaechlich schreiben")
    p_set.set_defaults(func=cmd_set)

    p_bulk = subparsers.add_parser(
        "bulk", parents=[common], help="alle Service-CNAMEs auf einen Host planen/aktualisieren"
    )
    p_bulk.add_argument("--host", required=True, help="Ziel-Host, z. B. b03 oder b03.mybister.com")
    p_bulk.add_argument("--apply", action="store_true", help="Aenderungen tatsaechlich schreiben")
    p_bulk.add_argument(
        "--force",
        action="store_true",
        help="abweichende vorhandene CNAMEs ueberschreiben (standardmaessig: Konflikt, kein Write)",
    )
    p_bulk.set_defaults(func=cmd_bulk)

    p_services = subparsers.add_parser(
        "services", parents=[common], help="Service-Registry anzeigen (ohne API-Zugriff)"
    )
    p_services.set_defaults(func=cmd_services)

    return parser


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _print_change(change: ChangeResult, width: int = 0) -> None:
    fqdn = change.fqdn.ljust(width) if width else change.fqdn
    print(f"{change.verb:<8} {fqdn} -> {change.target}")
    if change.message and change.status in (STATUS_CONFLICT, STATUS_FAILED):
        print(f"{'':<8} {change.message}")


def _print_summary(results: Sequence[ChangeResult]) -> None:
    counts = summarize(results)
    print("\nSummary:")
    print(f"Create: {counts[STATUS_CREATED]}")
    print(f"Update: {counts[STATUS_UPDATED]}")
    print(f"Unchanged: {counts[STATUS_UNCHANGED]}")
    print(f"Conflicts: {counts[STATUS_CONFLICT]}")
    print(f"Errors: {counts[STATUS_FAILED]}")


def _json_dump(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _exit_for_results(results: Sequence[ChangeResult]) -> int:
    counts = summarize(results)
    if counts[STATUS_FAILED]:
        return EXIT_API
    if counts[STATUS_CONFLICT]:
        return EXIT_CONFLICT
    return EXIT_OK


def _build_manager() -> HetznerDNSManager:
    """Build the manager from the environment (patched in tests)."""
    return HetznerDNSManager.from_env()


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> int:
    manager = _build_manager()
    records = manager.list_service_records()
    if args.json:
        _json_dump([record.to_dict() for record in records])
        return EXIT_OK

    rows = []
    for record in records:
        rows.append(
            [
                record.service_id,
                record.fqdn,
                record.target or "-",
                str(record.ttl) if record.ttl is not None else "-",
                record.expected_host or "-",
                record.status.upper(),
            ]
        )
    _print_table(["SERVICE", "DNS NAME", "TARGET", "TTL", "EXPECTED", "STATUS"], rows)
    return EXIT_OK


def cmd_get(args: argparse.Namespace) -> int:
    manager = _build_manager()
    record = manager.get_service_record(args.service_id)
    if record.conflict_types:
        exit_code = EXIT_CONFLICT
    elif not record.exists:
        exit_code = EXIT_NOT_FOUND
    else:
        exit_code = EXIT_OK
    if args.json:
        _json_dump(record.to_dict())
        return exit_code

    print(f"Service: {record.service_id}")
    print(f"DNS:     {record.fqdn}")
    if record.conflict_types:
        print(f"Type:    {', '.join(record.conflict_types)} (Konflikt - kein CNAME moeglich)")
        if record.conflict_targets:
            print(f"Target:  {', '.join(record.conflict_targets)}")
        print("Action:  Keine Aenderung - fremden Record nicht anfassen")
        return EXIT_CONFLICT
    if not record.exists:
        print("Type:    CNAME")
        print("Target:  (nicht angelegt)")
        print("TTL:     -")
        print(f"Action:  anlegen mit: python -m tools.hetzner_dns set {record.service_id} <host> --apply")
        return EXIT_NOT_FOUND
    print("Type:    CNAME")
    print(f"Target:  {record.target}")
    print(f"TTL:     {record.ttl if record.ttl is not None else '-'}")
    return EXIT_OK


def cmd_set(args: argparse.Namespace) -> int:
    manager = _build_manager()
    change = manager.ensure_service_record(args.service_id, args.host, apply=args.apply)

    if args.json:
        payload = change.to_dict()
        payload["dry_run"] = not args.apply
        _json_dump(payload)
        return STATUS_EXIT_CODES[change.status]

    print(f"Service: {change.service_id}")
    print(f"DNS:     {change.fqdn}")
    print("Type:    CNAME")
    if change.current_target and change.current_target != change.target:
        print(f"Target:  {change.current_target} -> {change.target}")
    else:
        print(f"Target:  {change.target}")
    if change.ttl is not None:
        print(f"TTL:     {change.ttl}")
    print(f"Status:  {change.status.upper()}{' (geschrieben)' if change.applied else ''}")
    if change.message:
        print(f"Note:    {change.message}")
    if not args.apply:
        print("\nDRY RUN - No DNS changes applied. Use --apply to write.")
    return STATUS_EXIT_CODES[change.status]


def cmd_bulk(args: argparse.Namespace) -> int:
    manager = _build_manager()
    plan = manager.plan_bulk(args.host, force=args.force)
    results = manager.apply_bulk(plan) if args.apply else list(plan)

    if args.json:
        _json_dump(
            {
                "dry_run": not args.apply,
                "force": args.force,
                "host": args.host,
                "summary": summarize(results),
                "results": [change.to_dict() for change in results],
            }
        )
        return _exit_for_results(results)

    for change in results:
        _print_change(change, width=max(len(c.fqdn) for c in results))
    _print_summary(results)
    if not args.apply:
        print("\nDRY RUN - No DNS changes applied.")
    else:
        print("\nApplied.")
    return _exit_for_results(results)


def cmd_services(args: argparse.Namespace) -> int:
    registry_path = (os.environ.get("API_SERVICES_REGISTRY") or "").strip() or None
    registry = ServiceRegistry.load(Path(registry_path) if registry_path else None)
    if args.json:
        _json_dump(
            {
                "services": [service.to_dict() for service in registry],
                "excluded": [
                    {"id": excluded.id, "reason": excluded.reason} for excluded in registry.excluded
                ],
            }
        )
        return EXIT_OK
    service_domain = (os.environ.get("API_SERVICES_DOMAIN") or "api.mybister.com").strip()
    rows = [
        [service.id, f"{service.id}.{service_domain}", service.path, service.description]
        for service in registry
    ]
    _print_table(["SERVICE", "DNS NAME", "PATH", "DESCRIPTION"], rows)
    if registry.excluded:
        print("\nNicht verwaltet (kein CNAME):")
        for excluded in registry.excluded:
            print(f"  - {excluded.id}: {excluded.reason}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format=LOG_FORMAT,
        stream=sys.stderr,
    )

    try:
        return args.func(args)
    except ValidationError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except APIError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return EXIT_API
    except HetznerDNSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
