"""Tests for ``tools/deploy_targets.py`` (deployment placement per host).

The helper is standard-library only (the GitHub runners have no
``httpx``); these tests exercise both the functions and the CLI entry
point against the real registry.
"""

from __future__ import annotations

import json

from tools import deploy_targets
from tools.hetzner_dns.registry import ComponentDefinition, ServiceDefinition, ServiceRegistry


def _small_registry() -> ServiceRegistry:
    return ServiceRegistry(
        services=(
            ServiceDefinition(id="sentiment", description="s", compose_service="sentiment", host="b01"),
            ServiceDefinition(id="ocr", description="o", compose_service="ocr", host="b03"),
        ),
        components=(
            ComponentDefinition(id="apikey-auth", compose_service="apikey-auth", host="*"),
            ComponentDefinition(id="ocr-worker", compose_service="ocr-worker", host="b03"),
        ),
    )


def test_compose_services_for_host_include_components():
    names = deploy_targets.compose_services_for("b03", registry=_small_registry())
    assert names == ["ocr", "apikey-auth", "ocr-worker"]


def test_fqdn_input_matches_short_assignment():
    names = deploy_targets.compose_services_for("b03.mybister.com", registry=_small_registry())
    assert "ocr" in names and "ocr-worker" in names
    assert deploy_targets.compose_services_for("b01", registry=_small_registry()) == [
        "sentiment",
        "apikey-auth",
    ]


def test_wildcard_component_is_on_every_host():
    registry = _small_registry()
    for host in ("b01", "b02", "b03"):
        assert "apikey-auth" in deploy_targets.compose_services_for(host, registry=registry)


def test_other_services_are_the_complement():
    registry = _small_registry()
    mine = set(deploy_targets.compose_services_for("b01", registry=registry))
    others = set(deploy_targets.other_services("b01", registry=registry))
    assert not (mine & others)
    assert others == {"ocr", "ocr-worker"}
    assert "apikey-auth" not in others  # "*" runs everywhere


def test_real_registry_covers_all_hosts():
    registry = ServiceRegistry.load()
    for host in ("b01", "b02", "b03"):
        names = deploy_targets.compose_services_for(host, registry=registry)
        assert "apikey-auth" in names
    b03 = deploy_targets.compose_services_for("b03", registry=registry)
    assert "ocr" in b03 and "ocr-worker" in b03 and "ocrpdf-lb" in b03
    # ocrpdf-app: die Node-Replicas duessen beim Deploy nicht fehlen (Pull + Recreate)
    assert "ocrpdf" in b03
    assert "sentiment" not in b03


def test_cli_output_and_unknown_host(capsys):
    code = deploy_targets.main(["b03"])
    captured = capsys.readouterr()
    assert code == 0
    assert "ocr-worker" in captured.out

    code = deploy_targets.main(["b99"])
    captured = capsys.readouterr()
    assert code == 1
    assert "keine Services" in captured.err

    code = deploy_targets.main(["b02", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    names = json.loads(captured.out)
    assert isinstance(names, list) and "callback-tester" in names
