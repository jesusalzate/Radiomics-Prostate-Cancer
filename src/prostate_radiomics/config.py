from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config; return an empty dict when omitted."""

    if path is None:
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        payload = _load_minimal_yaml_mapping(Path(path))
    else:
        with Path(path).open("r", encoding="utf-8") as file_handle:
            payload = yaml.safe_load(file_handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return payload


def config_arguments(payload: dict[str, Any], section: str | None = None) -> dict[str, Any]:
    """Return flat argument defaults from a config payload."""

    if section and isinstance(payload.get(section), dict):
        section_payload = payload[section]
        if isinstance(section_payload.get("arguments"), dict):
            return section_payload["arguments"]
        return section_payload
    if isinstance(payload.get("arguments"), dict):
        return payload["arguments"]
    return payload


def arguments_to_cli_items(arguments: dict[str, Any]) -> list[str]:
    """Convert a mapping into argparse-style command-line items."""

    items: list[str] = []
    for key, value in arguments.items():
        if value is None or key in {"command", "description"}:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                items.append(flag)
            continue
        if isinstance(value, list):
            items.append(flag)
            items.extend(str(item) for item in value)
            continue
        items.extend([flag, str(value)])
    return items


def _load_minimal_yaml_mapping(path: Path) -> dict[str, Any]:
    """Parse the simple YAML subset used by this repo's configs.

    This fallback keeps `--dry-run` usable in bare Python environments. Full
    YAML support is still provided by PyYAML when project dependencies are
    installed.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any, Any, str | None]] = [(-1, root, None, None)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        text = raw_line.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if text.startswith("- "):
            if not isinstance(container, list):
                current_indent, current_container, parent, key = stack[-1]
                if isinstance(current_container, dict) and not current_container and parent is not None and key is not None:
                    container = []
                    parent[key] = container
                    stack[-1] = (current_indent, container, parent, key)
                else:
                    raise ValueError(f"Unsupported YAML list at {path}:{line_number}")
            container.append(_parse_scalar(text[2:].strip()))
            continue

        key, separator, value = text.partition(":")
        if not separator:
            raise ValueError(f"Unsupported YAML line at {path}:{line_number}: {raw_line}")
        key = key.strip()
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            container[key] = child
            stack.append((indent, child, container, key))
        else:
            container[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
