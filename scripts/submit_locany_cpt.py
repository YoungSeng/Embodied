#!/usr/bin/env python3
"""Render and submit an A800 CPT job to a selected YG resource group."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "locany_cpt_resources.json"
BASE_YAMLS = {
    "smoke": PROJECT_ROOT / "locany_cpt_v4_a100x4_smoke_merlin.yaml",
    "formal": PROJECT_ROOT / "locany_cpt_v4_a100x4_formal_merlin.yaml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit LocateAnything CPT A800x4 to the YG default or "
            "AIAI_locate scheduling resource group"
        )
    )
    parser.add_argument("--mode", choices=tuple(BASE_YAMLS), default="smoke")
    parser.add_argument(
        "--cluster",
        default="yg",
        help="CPT scheduling profile: yg or aiai_locate (default: yg)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-yaml", type=Path, default=None)
    parser.add_argument("--output-yaml", type=Path, default=None)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--mlx-bin", default="mlx")
    runtime_group = parser.add_mutually_exclusive_group()
    runtime_group.add_argument(
        "--install-system-runtime-deps",
        dest="install_system_runtime_deps",
        action="store_true",
        help="Allow the CPT launcher to apt-install libgl1/libglib2.0-0 when required",
    )
    runtime_group.add_argument(
        "--no-install-system-runtime-deps",
        dest="install_system_runtime_deps",
        action="store_false",
        help="Fail runtime preflight instead of installing missing system libraries",
    )
    parser.set_defaults(install_system_runtime_deps=True)
    return parser.parse_args(argv)


def load_resource(config_path: Path, cluster: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported CPT resource config: {config_path}")
    a800 = config["a800"]
    resources = a800["resource_groups"]
    if cluster not in resources:
        raise ValueError(
            f"Unknown --cluster {cluster!r}; choose one of {sorted(resources)}"
        )
    resource = dict(resources[cluster])
    resource["cluster_id"] = int(a800["base_cluster_id"])
    resource["base_group_id"] = int(a800["base_group_id"])
    resource["group_id"] = int(resource["group_id"])
    return resource


def _replace_once(pattern: str, replacement: str, text: str, label: str) -> str:
    rendered, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} in base YAML; found {count}")
    return rendered


def render_job(
    base_text: str,
    *,
    cluster: str,
    resource: dict[str, Any],
    install_system_runtime_deps: bool = True,
) -> str:
    expected_cluster_id = int(resource["cluster_id"])
    expected_base_group_id = int(resource["base_group_id"])
    if f"clusterId: {expected_cluster_id}" not in base_text:
        raise ValueError(
            f"Base YAML is not an A800 YG CPT job (cluster {expected_cluster_id})"
        )

    rendered = _replace_once(
        rf"^(\s{{8}}-\s+){expected_base_group_id}\s*$",
        rf"\g<1>{int(resource['group_id'])}",
        base_text,
        "A800 groupIds entry",
    )

    # The checked-in A800 YAML intentionally has no queueName. Strip a stale
    # rendered queue if a user supplies --base-yaml, then add the selected one.
    rendered = re.sub(r"^\s{10}queueName:.*\n", "", rendered, flags=re.MULTILINE)
    queue_name = str(resource.get("queue_name", ""))
    if queue_name:
        rendered = _replace_once(
            r"^(\s{10}gpuv:\s*A800_SXM_40GB\s*)$",
            rf"\g<1>\n          queueName: {queue_name}",
            rendered,
            "A800 gpuv entry",
        )

    deps_value = "1" if install_system_runtime_deps else "0"
    deps_pattern = r'^\s{4}INSTALL_SYSTEM_RUNTIME_DEPS:\s*"[01]"\s*$'
    if re.search(deps_pattern, rendered, flags=re.MULTILINE):
        rendered = re.sub(
            deps_pattern,
            f'    INSTALL_SYSTEM_RUNTIME_DEPS: "{deps_value}"',
            rendered,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        rendered = _replace_once(
            r"^(\s{4}CUDA_DEVICES:.*)$",
            rf'\g<1>\n    INSTALL_SYSTEM_RUNTIME_DEPS: "{deps_value}"',
            rendered,
            "CUDA_DEVICES env entry",
        )

    if cluster != "yg":
        job_suffix = re.sub(r"[^a-z0-9-]+", "-", cluster.lower()).strip("-")
        rendered = _replace_once(
            r"^(\s{2}name:\s*')([^']+)('\s*)$",
            rf"\g<1>\g<2>-{job_suffix}\g<3>",
            rendered,
            "job name",
        )
        rendered = _replace_once(
            r"^(caption:\s*')([^']+)('\s*)$",
            rf"\g<1>\g<2> [{cluster}]\g<3>",
            rendered,
            "caption",
        )
    return rendered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    resource = load_resource(config_path, args.cluster)
    base_yaml = (args.base_yaml or BASE_YAMLS[args.mode]).expanduser().resolve()
    rendered = render_job(
        base_yaml.read_text(encoding="utf-8"),
        cluster=args.cluster,
        resource=resource,
        install_system_runtime_deps=args.install_system_runtime_deps,
    )

    if args.output_yaml is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_yaml = (
            PROJECT_ROOT
            / "jobs"
            / "rendered"
            / f"locany_cpt_a800x4_{args.mode}_{args.cluster}_{stamp}.yaml"
        )
    else:
        output_yaml = args.output_yaml.expanduser().resolve()
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(rendered, encoding="utf-8")

    print("===== LocateAnything CPT submission =====")
    print(f"mode                         : {args.mode}")
    print(f"cluster_selector             : {args.cluster}")
    print(f"resource_display_name        : {resource['display_name']}")
    print(f"cluster_id                   : {resource['cluster_id']}")
    print(f"resource_group_id            : {resource['group_id']}")
    print(f"resource_queue_name          : {resource.get('queue_name') or '<default>'}")
    print(
        "install_system_runtime_deps: "
        f"{int(args.install_system_runtime_deps)}"
    )
    print(f"rendered_yaml                : {output_yaml}")
    if args.render_only:
        print("[RENDER ONLY] mlx was not invoked")
        return 0

    command = [args.mlx_bin, "job", "submitv2", "--path", str(output_yaml)]
    print("submit_command               :", " ".join(command))
    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Cannot find {args.mlx_bin!r}; use --render-only outside a Merlin host. "
            f"Rendered YAML: {output_yaml}"
        ) from exc
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
