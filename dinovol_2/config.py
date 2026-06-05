from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DATASET_SECTIONS = ("dataset", "val_dataset", "monitor_dataset")


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"configuration references unset environment variable {name!r}")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(value: str | os.PathLike[str], *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path


def _load_datasets_file(section: Mapping[str, Any], *, relative_to: Path) -> list[dict[str, Any]]:
    datasets_file = section.get("datasets_file")
    if datasets_file is None:
        return list(section.get("datasets", []))

    datasets_path = _resolve_path(str(datasets_file), relative_to=relative_to)
    payload = _expand_env(_load_json(datasets_path))
    if isinstance(payload, dict):
        payload = payload.get("datasets")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"datasets_file {datasets_path} must contain a non-empty dataset list.")
    return [dict(item) for item in payload]


def _resolve_dataset_sections(config: dict[str, Any], *, relative_to: Path) -> None:
    for section_name in _DATASET_SECTIONS:
        section = config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping when provided.")
        datasets = _load_datasets_file(section, relative_to=relative_to)
        if not datasets:
            raise ValueError(f"{section_name}.datasets must not be empty.")
        section["datasets"] = datasets
        section.pop("datasets_file", None)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    payload = _expand_env(_load_json(path))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration {path} must contain a JSON object.")
    config = dict(payload)
    _resolve_dataset_sections(config, relative_to=path.parent)
    return config
