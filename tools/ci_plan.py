#!/usr/bin/env python3
"""Plan CI image builds: what must be built, what can be pulled (stdlib only).

The Integration-Tests workflow must provide the complete dev image set on
one runner. Instead of building every image from scratch on every run, this
tool decides per distinct image:

* the build context changed in this PR/push -> the image must be built
  (one job per image, pushed to GHCR with tag ``ci-<sha>``),
* everything else -> pull ``ghcr.io/<owner>/<name>:latest`` (built on main).

Input is the JSON of
``docker compose --profile dev --profile build --profile builder config
--format json`` plus the changed file list and the event name. Output goes
to ``$GITHUB_OUTPUT`` (``matrix=``, ``images=``, ``has_builds=``) when that
file is set, plus a human-readable summary on stdout.

The dev image tag (``<name>:local``) and the GHCR image
(``ghcr.io/<owner>/<name>:latest``, built by the ``*-image-builder``
services) are matched by their base name.
"""

import argparse
import json
import os
import sys

# Changes in these files affect every image or the planning itself.
GLOBAL_TRIGGERS = (
    "docker-compose.yml",
    "docker-compose.ci.yml",
    ".github/workflows/ci.yml",
    "tools/ci_plan.py",
)
# Shared code directories (parity with build-and-push-docker-images.yml).
SHARED_PATHS = ("libs", "packages", "shared", "common")


def base_name(image: str) -> str:
    """'ghcr.io/chbister/emotion:latest' -> 'emotion', 'emotion:local' -> 'emotion'."""
    name = image.rsplit("/", 1)[-1]
    return name.rsplit(":", 1)[0] if ":" in name else name


def ghcr_repo(image: str) -> str:
    """'ghcr.io/chbister/emotion:latest' -> 'ghcr.io/chbister/emotion'."""
    name = image.rsplit("/", 1)[-1]
    return image.rsplit(":", 1)[0] if ":" in name else image


def plan(compose: dict, changed: list, event: str, root: str) -> dict:
    services = compose.get("services", {})

    # GHCR repository per image base name (from the *-image-builder services).
    ghcr_by_name = {}
    for svc in services.values():
        image = svc.get("image") or ""
        if image.startswith("ghcr.io/"):
            ghcr_by_name[base_name(image)] = ghcr_repo(image)

    # Distinct dev images: group services with a build config by image tag.
    groups = {}
    for name, svc in services.items():
        profiles = svc.get("profiles") or []
        build = svc.get("build")
        if build is None or "dev" not in profiles:
            continue
        tag = svc.get("image") or name
        group = groups.setdefault(tag, {"services": [], "build": build})
        group["services"].append(name)

    build_all = event == "workflow_dispatch" or any(
        f in GLOBAL_TRIGGERS for f in changed
    ) or any(f.split("/")[0] in SHARED_PATHS for f in changed)

    matrix, images = [], []
    for tag, group in sorted(groups.items()):
        name = base_name(tag)
        ghcr = ghcr_by_name.get(name)
        if ghcr is None:
            sys.exit(f"error: no GHCR image for dev tag {tag!r} "
                     f"(missing {name}-image-builder?)")
        build = group["build"]
        context = build.get("context") if isinstance(build, dict) else build
        rel_context = os.path.relpath(context, root).replace(os.sep, "/")
        # compose resolves build.dockerfile relative to the build context
        # (or absolute); the build job needs it relative to the project root.
        df = build.get("dockerfile") if isinstance(build, dict) else None
        if df is None:
            dockerfile = f"{rel_context}/Dockerfile"
        elif os.path.isabs(df):
            dockerfile = os.path.relpath(df, root).replace(os.sep, "/")
        else:
            dockerfile = f"{rel_context}/{df}"
        target = (build.get("target") if isinstance(build, dict) else None) or ""

        if build_all:
            reason = f"Global trigger (event: {event} or shared/CI file changed)"
        elif any(f == rel_context or f.startswith(rel_context + "/") for f in changed):
            reason = f"Changes detected in service context: {rel_context}"
        else:
            reason = ""

        images.append({
            "name": name,
            "ghcr": ghcr,
            "changed": bool(reason),
            "primary_service": group["services"][0],
        })
        if reason:
            matrix.append({
                "name": name,
                "ghcr": ghcr,
                "primary_service": group["services"][0],
                "services": ", ".join(group["services"]),
                "rel_context": rel_context,
                "dockerfile": dockerfile,
                "target": target,
                "reason": reason,
            })

    return {"matrix": matrix, "images": images}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-json",
                        help="output of 'docker compose config --format json' "
                             "(default: stdin)")
    parser.add_argument("--changed-files-json", required=True,
                        help='JSON array of changed file paths, e.g. ["a/b.py"]')
    parser.add_argument("--event", default="pull_request")
    parser.add_argument("--root", default=os.getcwd(),
                        help="project root for relative paths (default: cwd)")
    args = parser.parse_args()

    if args.compose_json:
        with open(args.compose_json) as fh:
            compose = json.load(fh)
    else:
        compose = json.load(sys.stdin)
    changed = json.loads(args.changed_files_json)

    result = plan(compose, changed, args.event, args.root)
    matrix, images = result["matrix"], result["images"]

    print(f"Changed files: {len(changed)}")
    for entry in images:
        state = "BUILD  " if entry["changed"] else "pull   "
        print(f"  {state} {entry['name']:20s} ({entry['ghcr']})")
    print(f"Builds: {len(matrix)} of {len(images)} images")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as fh:
            fh.write(f"matrix={json.dumps({'include': matrix}, separators=(',', ':'))}\n")
            fh.write(f"images={json.dumps(images, separators=(',', ':'))}\n")
            fh.write(f"has_builds={'true' if matrix else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
