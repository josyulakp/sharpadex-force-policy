"""Load the YAML config and apply `dotted.key=value` overrides from the command line."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"unknown config section '{k}' in override '{dotted}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key '{dotted}'")
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got '{ov}'")
        _set(cfg, key, yaml.safe_load(raw))   # yaml parsing gives ints/floats/bools/lists for free
    return cfg


def parse_cli(argv: list[str]) -> dict:
    """`--config path` plus any number of key=value overrides."""
    path, overrides, it = None, [], iter(argv)
    for a in it:
        if a == "--config":
            path = next(it)
        else:
            overrides.append(a)
    return load_config(path, overrides)
