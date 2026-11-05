"""Structured exceptions for the Hetzner DNS tool.

All errors derive from :class:`HetznerDNSError`. Error messages never
contain the API token or any other secret.
"""

from __future__ import annotations

from typing import Optional


class HetznerDNSError(Exception):
    """Base class for every error raised by this package."""


class ValidationError(HetznerDNSError):
    """An argument (service id, host name, FQDN, ...) is invalid."""


class ConfigError(HetznerDNSError):
    """The environment/configuration is missing or inconsistent."""


class MissingTokenError(ConfigError):
    """``HETZNER_API_TOKEN`` is not set (or empty)."""


class NotFoundError(HetznerDNSError):
    """A requested object does not exist."""


class ZoneNotFoundError(ConfigError, NotFoundError):
    """The configured DNS zone does not exist in the Hetzner account."""


class APIError(HetznerDNSError):
    """The Hetzner Cloud API answered with an error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.correlation_id = correlation_id

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts = [self.message]
        if self.code:
            parts.append(f"code={self.code}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.correlation_id:
            parts.append(f"correlation_id={self.correlation_id}")
        return " | ".join(parts)


class RateLimitError(APIError):
    """The API rate limit was exceeded (``rate_limit_exceeded``)."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TransportError(APIError):
    """Network-level failure while talking to the API."""


class ActionFailedError(APIError):
    """An asynchronous Hetzner action finished with status ``error``."""


class ActionTimeoutError(APIError):
    """An asynchronous Hetzner action did not finish in time."""
