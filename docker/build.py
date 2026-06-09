#!/usr/bin/env python3
"""Build and push Miles Docker images.

Pick a build with --cuda / --arch. The (cuda, arch) -> build-args / tag-suffix
table below is the single source of truth; docker/Dockerfile's header defers
to it. An empty build_args means "use the Dockerfile's
own ARG defaults" (the primary 130/x86 combo).

Usage:
    python docker/build.py --cuda 130 --arch x86 --push
    python docker/build.py --cuda 129 --arch x86 --tag latest
    python docker/build.py --cuda 130 --arch x86 --test --dry-run
"""

import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import typer

IMAGE = "radixark/miles"
DOCKERFILE_DEFAULT = "docker/Dockerfile"

# Single source of truth, keyed by (cuda, arch). build_args override the
# Dockerfile's ARG defaults; {} means "use the defaults" (130/x86). tag_suffix
# encodes the CUDA variant only (cu13 = none, cu12 = -cu12); arch never appears
# in the tag. aarch64 / multi-arch is intentionally not wired here yet.
BUILD_CONFIGS: dict[tuple[str, str], dict] = {
    ("130", "x86"): {
        "tag_suffix": "",
        "build_args": {},
    },
    ("129", "x86"): {
        "tag_suffix": "-cu12",
        "build_args": {
            "ENABLE_CUDA_13": "0",
            "SGLANG_IMAGE_TAG": "v0.5.12-cu129",
            "WHEELS_TAG": "cu129-x86_64-v0.5.12",
        },
    },
}


class Cuda(str, Enum):
    cu129 = "129"
    cu130 = "130"


class Arch(str, Enum):
    x86 = "x86"
    aarch64 = "aarch64"


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def dockerfile_args(dockerfile: str) -> set[str]:
    """Names declared via `ARG <NAME>` in the Dockerfile.

    Used to fail fast when a build config emits a build-arg the Dockerfile
    does not accept -- exactly the drift (e.g. a dead ENABLE_SGLANG_PATCH)
    this table exists to prevent.
    """
    args: set[str] = set()
    for line in Path(dockerfile).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("ARG "):
            args.add(stripped[len("ARG ") :].split("=", 1)[0].strip())
    return args


def resolve_tags(image: str, tag: str, suffix: str, test: bool) -> list[str]:
    """Tag is prefix + suffix: prefix is `tag` (dev/latest/literal), suffix is
    the CUDA variant. Base `dev` also gets a timestamped sibling for the daily
    build's history + prune (`^dev-[0-9]{12}$`). `--test` appends `-test` and
    publishes a single, non-timestamped tag -- a manual, overwritable image."""
    base = f"{image}:{tag}{suffix}"
    if test:
        return [f"{base}-test"]
    if tag == "dev":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        return [base, f"{base}-{timestamp}"]
    return [base]


def build_and_push(cuda: str, arch: str, tag: str, test: bool, dockerfile: str, push: bool, dry_run: bool) -> None:
    config = BUILD_CONFIGS.get((cuda, arch))
    if config is None:
        supported = ", ".join(f"(cuda={c}, arch={a})" for c, a in BUILD_CONFIGS)
        raise typer.BadParameter(f"no build config for (cuda={cuda}, arch={arch}); supported: {supported}")

    build_args = config["build_args"]
    undeclared = set(build_args) - dockerfile_args(dockerfile)
    if undeclared:
        raise typer.BadParameter(f"build config emits args not declared in {dockerfile}: {sorted(undeclared)}")

    tags = resolve_tags(IMAGE, tag, config["tag_suffix"], test)

    cmd = ["docker", "buildx", "build", "-f", dockerfile]
    if push:
        cmd += ["--push"]

    # Proxy args (pass through if set in environment, check both cases)
    for arg_name in ["HTTP_PROXY", "HTTPS_PROXY"]:
        value = os.environ.get(arg_name.lower()) or os.environ.get(arg_name)
        if value:
            cmd += ["--build-arg", f"{arg_name}={value}"]
    cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

    for key, value in build_args.items():
        cmd += ["--build-arg", f"{key}={value}"]

    for t in tags:
        cmd += ["-t", t]
    cmd += ["."]

    print(f"\n=== Building {' '.join(tags)} ===", flush=True)
    run(cmd, dry_run)


def main(
    cuda: Cuda = typer.Option(..., help="CUDA build: 129 (12.9) or 130 (13.0)."),  # noqa: B008
    arch: Arch = typer.Option(..., help="Target arch: x86 or aarch64."),  # noqa: B008
    tag: str = typer.Option(  # noqa: B008
        "dev",
        help="Tag base. 'dev' publishes a rolling + timestamped tag; any other value publishes that tag verbatim.",
    ),
    test: bool = typer.Option(  # noqa: B008
        False,
        help="Append a -test suffix (e.g. dev-test); a single, non-timestamped throwaway tag.",
    ),
    dockerfile: str = typer.Option(DOCKERFILE_DEFAULT, help="Path to the Dockerfile."),  # noqa: B008
    push: bool = typer.Option(False, help="Push images to registry after building."),  # noqa: B008
    dry_run: bool = typer.Option(False, help="Print commands without executing them."),  # noqa: B008
) -> None:
    build_and_push(cuda.value, arch.value, tag, test, dockerfile, push=push, dry_run=dry_run)


if __name__ == "__main__":
    typer.run(main)
