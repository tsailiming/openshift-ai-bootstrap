#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["pyyaml"]
# ///
"""Enable NeMo Guardrails IPP plugins on the live MaaS ConfigMap.

Edits custom-ipp-config.yaml in place (idempotent), applies it, then
restarts payload-processing and waits for rollout.

Requires: oc (logged in). Run with: uv run enable-ipp-nemo-plugins.py ...
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse, urlunparse

DEFAULT_NS = "openshift-ingress"
DEFAULT_CM = "payload-processing-plugins"
DEFAULT_DEPLOY = "payload-processing"
DEFAULT_PATH = "/v1/guardrail/checks"
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_ROLLOUT_TIMEOUT = "180s"

REQUEST_TYPE = "nemo-request-guard"
REQUEST_NAME = "nemo-input"
RESPONSE_TYPE = "nemo-response-guard"
RESPONSE_NAME = "nemo-output"


def die(msg: str) -> None:
    sys.exit(f"error: {msg}")


def require_yaml():
    try:
        import yaml
    except ImportError:
        die("PyYAML is required. Run this script with: uv run enable-ipp-nemo-plugins.py")
    return yaml


def oc(*args: str, capture: bool = True) -> str:
    cmd = ["oc", *args]
    result = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        die(f"oc {' '.join(args)} failed: {err}")
    return result.stdout if capture else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enable NeMo Guardrails plugins on the running IPP ConfigMap."
    )
    parser.add_argument(
        "--request",
        action="store_true",
        help="Enable nemo-request-guard (input rails). Default if neither flag is set.",
    )
    parser.add_argument(
        "--response",
        action="store_true",
        help="Enable nemo-response-guard (output rails).",
    )
    parser.add_argument(
        "--nemo-guardrail-check-url",
        required=True,
        metavar="URL",
        help="NeMo Guardrails checks URL (scheme optional; host without port uses https). "
        f"If the URL has no path, {DEFAULT_PATH} is appended.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the patched ConfigMap YAML and exit without applying or restarting.",
    )
    args = parser.parse_args()
    if not args.request and not args.response:
        args.request = True
    return args


def normalize_guardrail_check_url(raw: str) -> str:
    """Return the full nemoURL value for the IPP plugin parameters."""
    raw = raw.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname:
        die(f"invalid --nemo-guardrail-check-url: {raw}")

    path = parsed.path.rstrip("/")
    if not path:
        path = DEFAULT_PATH

    if parsed.port is not None:
        scheme = parsed.scheme or "https"
        netloc = f"{parsed.hostname}:{parsed.port}"
    else:
        scheme = "https"
        netloc = parsed.hostname

    return urlunparse((scheme, netloc, path, "", "", ""))


def find_plugin_index(plugins: list[Any], plugin_type: str) -> int:
    for i, plugin in enumerate(plugins):
        if isinstance(plugin, dict) and plugin.get("type") == plugin_type:
            return i
    return -1


def upsert_plugin(
    plugins: list[Any],
    plugin_type: str,
    name: str,
    parameters: dict[str, Any],
    *,
    before_type: str | None = None,
    after_type: str | None = None,
) -> None:
    params = dict(parameters)
    idx = find_plugin_index(plugins, plugin_type)
    if idx >= 0:
        plugins[idx]["name"] = name
        plugins[idx]["parameters"] = params
        return
    insert_at = len(plugins)
    if before_type:
        before = find_plugin_index(plugins, before_type)
        if before >= 0:
            insert_at = before
    elif after_type:
        after = find_plugin_index(plugins, after_type)
        if after >= 0:
            insert_at = after + 1
    plugins.insert(
        insert_at,
        {"type": plugin_type, "name": name, "parameters": params},
    )


def plugin_ref_names(refs: list[Any]) -> list[str]:
    names = []
    for item in refs:
        if isinstance(item, dict) and "pluginRef" in item:
            names.append(item["pluginRef"])
    return names


def upsert_ref(refs: list[Any], name: str, *, before: str | None = None) -> None:
    if name in plugin_ref_names(refs):
        return
    new = {"pluginRef": name}
    if before:
        for i, item in enumerate(refs):
            if isinstance(item, dict) and item.get("pluginRef") == before:
                refs.insert(i, new)
                return
    refs.append(new)


def default_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    profiles = cfg.setdefault("profiles", [])
    for profile in profiles:
        if isinstance(profile, dict) and profile.get("name") == "default":
            return profile
    if profiles and isinstance(profiles[0], dict):
        return profiles[0]
    profile: dict[str, Any] = {"name": "default", "plugins": {}}
    profiles.append(profile)
    return profile


def patch_config(
    cfg: dict[str, Any],
    *,
    url: str,
    timeout_seconds: int,
    enable_request: bool,
    enable_response: bool,
) -> dict[str, Any]:
    plugins = cfg.setdefault("plugins", [])
    if not isinstance(plugins, list):
        die("custom-ipp-config.yaml plugins is not a list")

    if enable_request:
        upsert_plugin(
            plugins,
            REQUEST_TYPE,
            REQUEST_NAME,
            {"nemoURL": url, "timeoutSeconds": timeout_seconds},
            before_type="api-translation",
        )

    if enable_response:
        upsert_plugin(
            plugins,
            RESPONSE_TYPE,
            RESPONSE_NAME,
            {"nemoURL": url, "timeoutSeconds": timeout_seconds},
            after_type="apikey-injection",
        )

    profile = default_profile(cfg)
    chain = profile.setdefault("plugins", {})
    request_refs = chain.setdefault("request", [])
    response_refs = chain.setdefault("response", [])
    if not isinstance(request_refs, list):
        die("profiles.plugins.request is not a list")
    if not isinstance(response_refs, list):
        die("profiles.plugins.response is not a list")

    if enable_request:
        upsert_ref(request_refs, REQUEST_NAME, before="api-translation")
    if enable_response:
        upsert_ref(response_refs, RESPONSE_NAME)

    return cfg


def dump_yaml(cfg: dict[str, Any]) -> str:
    yaml = require_yaml()

    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data: object) -> bool:
            return True

    dumped = yaml.dump(
        cfg,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        Dumper=NoAliasDumper,
    )
    if not dumped.endswith("\n"):
        dumped += "\n"
    return dumped


def main() -> None:
    args = parse_args()
    if shutil.which("oc") is None:
        die("oc not found in PATH")
    nemo_url = normalize_guardrail_check_url(args.nemo_guardrail_check_url)
    print(
        f"request={'on' if args.request else 'off'} "
        f"response={'on' if args.response else 'off'} "
        f"timeoutSeconds={DEFAULT_TIMEOUT_SECONDS}"
        f"{' dry-run' if args.dry_run else ''}"
    )
    print(f"nemoURL={nemo_url}")

    oc("whoami")
    raw = oc("get", "cm", DEFAULT_CM, "-n", DEFAULT_NS, "-o", "json")
    cm = json.loads(raw)
    inner = (cm.get("data") or {}).get("custom-ipp-config.yaml")
    if not inner:
        die(f"ConfigMap {DEFAULT_NS}/{DEFAULT_CM} has no data.custom-ipp-config.yaml")

    yaml = require_yaml()
    loaded = yaml.safe_load(inner)
    if not isinstance(loaded, dict):
        die("custom-ipp-config.yaml did not parse as a mapping")
    cfg = copy.deepcopy(loaded)

    patched = patch_config(
        cfg,
        url=nemo_url,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        enable_request=args.request,
        enable_response=args.response,
    )
    new_yaml = dump_yaml(patched)
    print("--- patched custom-ipp-config.yaml ---")
    print(new_yaml, end="")
    print("--------------------------------------")

    if args.dry_run:
        print("dry-run: ConfigMap not patched, deployment not restarted")
        return

    patch = {"data": {"custom-ipp-config.yaml": new_yaml}}
    oc("patch", "cm", DEFAULT_CM, "-n", DEFAULT_NS, "--type=merge", "-p", json.dumps(patch))

    print(f"Restarting {DEFAULT_DEPLOY} in {DEFAULT_NS}...")
    oc("rollout", "restart", f"deploy/{DEFAULT_DEPLOY}", "-n", DEFAULT_NS)
    oc(
        "rollout",
        "status",
        f"deploy/{DEFAULT_DEPLOY}",
        "-n",
        DEFAULT_NS,
        f"--timeout={DEFAULT_ROLLOUT_TIMEOUT}",
    )

    print(f"Done. IPP is using {nemo_url} (timeoutSeconds={DEFAULT_TIMEOUT_SECONDS})")
    print(
        oc(
            "get",
            "cm",
            DEFAULT_CM,
            "-n",
            DEFAULT_NS,
            "-o",
            "jsonpath={.data.custom-ipp-config\\.yaml}",
        )
    )


if __name__ == "__main__":
    main()
