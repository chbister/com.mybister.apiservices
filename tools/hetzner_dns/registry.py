"""Central, declarative service registry.

The registry is a single JSON file (``tools/service_registry.json``) that
lists every independently deployable API service. It is the single source
of truth for the DNS tool and is intended to be reused later by the
deployment/resource scheduler.

Importing this module has no side effects; the file is only read when
:meth:`ServiceRegistry.load` is called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .errors import ConfigError, ValidationError
from .validators import validate_service_id

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "service_registry.json"


@dataclass(frozen=True)
class ServiceDefinition:
    """A single deployable service."""

    id: str
    description: str = ""
    compose_service: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "description": self.description,
            "compose_service": self.compose_service,
        }


@dataclass(frozen=True)
class ExcludedService:
    """A component that is deliberately *not* managed via DNS."""

    id: str
    reason: str = ""


@dataclass(frozen=True)
class ServiceRegistry:
    """Immutable list of deployable services (ordered)."""

    services: Sequence[ServiceDefinition] = field(default_factory=tuple)
    excluded: Sequence[ExcludedService] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: List[str] = []
        for service in self.services:
            normalized = validate_service_id(service.id)
            if normalized != service.id:
                raise ValidationError(f"Service-ID '{service.id}' muss kleingeschrieben sein")
            if service.id in seen:
                raise ValidationError(f"Service-ID '{service.id}' kommt mehrfach in der Service-Registry vor")
            seen.append(service.id)
        object.__setattr__(self, "_index", {s.id: s for s in self.services})

    # -- container style helpers ------------------------------------------
    @property
    def ids(self) -> List[str]:
        return [service.id for service in self.services]

    def get(self, service_id: str) -> ServiceDefinition:
        """Return a service definition or raise :class:`ValidationError`."""
        normalized = validate_service_id(service_id)
        index: Dict[str, ServiceDefinition] = getattr(self, "_index")
        try:
            return index[normalized]
        except KeyError:
            raise ValidationError(
                f"Unbekannte Service-ID '{service_id}'. "
                f"Registry: {', '.join(self.ids)}"
            ) from None

    def __contains__(self, service_id: object) -> bool:
        if not isinstance(service_id, str):
            return False
        index: Dict[str, ServiceDefinition] = getattr(self, "_index")
        return service_id.strip().lower() in index

    def __len__(self) -> int:
        return len(self.services)

    def __iter__(self):
        return iter(self.services)

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ServiceRegistry":
        """Load the registry from a JSON file (defaults to the repo file)."""
        registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
        if not registry_path.is_file():
            raise ConfigError(f"Service-Registry nicht gefunden: {registry_path}")
        try:
            raw = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Service-Registry {registry_path} ist kein gueltiges JSON: {exc}") from None
        if not isinstance(raw, dict):
            raise ConfigError(f"Service-Registry {registry_path} muss ein JSON-Objekt sein")
        if raw.get("version") != 1:
            raise ConfigError(f"Service-Registry {registry_path}: unbekannte version '{raw.get('version')}'")

        raw_services = raw.get("services")
        if not isinstance(raw_services, list) or not raw_services:
            raise ConfigError(f"Service-Registry {registry_path}: 'services' darf nicht leer sein")

        services: List[ServiceDefinition] = []
        for entry in raw_services:
            if not isinstance(entry, dict) or not entry.get("id"):
                raise ConfigError(f"Service-Registry {registry_path}: Eintrag ohne 'id': {entry!r}")
            services.append(
                ServiceDefinition(
                    id=str(entry["id"]),
                    description=str(entry.get("description", "")),
                    compose_service=str(entry.get("compose_service", "")),
                )
            )

        excluded: List[ExcludedService] = []
        for entry in raw.get("excluded", []) or []:
            if isinstance(entry, dict) and entry.get("id"):
                excluded.append(ExcludedService(id=str(entry["id"]), reason=str(entry.get("reason", ""))))

        return cls(services=tuple(services), excluded=tuple(excluded))
