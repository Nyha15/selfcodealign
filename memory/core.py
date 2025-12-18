"""
Core / in-context memory layer (Letta-inspired).

Design constraints:
- Persistent across sessions (disk-backed JSON in `.memory/blocks/`)
- Always loaded at agent startup (via `load_all_core_memory()`)
- No caching / no in-memory state (read-modify-write every operation)
- Simple + demo-friendly (human-readable JSON, minimal code)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


# Fixed set of Letta-style core memory blocks.
_CORE_BLOCK_FILES: Dict[str, str] = {
    "project_context": "project_context.json",
    "current_work": "current_work.json",
    "workflow_state": "workflow_state.json",
    "learned_patterns": "learned_patterns.json",
}


def _repo_root() -> Path:
    # `memory/core.py` -> repo root is the parent of the `memory/` package.
    return Path(__file__).resolve().parents[1]


def _blocks_dir() -> Path:
    return _repo_root() / ".memory" / "blocks"


def _utc_now_iso() -> str:
    # ISO-8601 with 'Z' suffix to clearly indicate UTC.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_block_name(block: str) -> str:
    if block not in _CORE_BLOCK_FILES:
        valid = ", ".join(sorted(_CORE_BLOCK_FILES.keys()))
        raise ValueError(f"Unknown memory block '{block}'. Valid blocks: {valid}")
    return block


def _block_path(block: str) -> Path:
    block = _validate_block_name(block)
    return _blocks_dir() / _CORE_BLOCK_FILES[block]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Core memory block file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Core memory block must be a JSON object: {path}")
    return data


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    # Always keep JSON files human-readable for interview/demo use.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


_PathToken = Union[str, int]


def _parse_field_path(field: str) -> List[_PathToken]:
    """
    Parse a simple dotted / bracketed path like:
      - "phase"
      - "preferences.default_slack_channel"
      - "workflows[0].observed_sequence"
    into tokens: ["workflows", 0, "observed_sequence"].
    """
    if not field or not isinstance(field, str):
        raise ValueError("field must be a non-empty string")

    tokens: List[_PathToken] = []
    i = 0
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            tokens.append("".join(buf))
            buf = []

    while i < len(field):
        ch = field[i]
        if ch == ".":
            flush_buf()
            i += 1
            continue
        if ch == "[":
            flush_buf()
            j = field.find("]", i + 1)
            if j == -1:
                raise ValueError(f"Unclosed '[' in field path: {field!r}")
            idx_str = field[i + 1 : j].strip()
            if not idx_str.isdigit():
                raise ValueError(f"Only integer indices are supported in field paths: {field!r}")
            tokens.append(int(idx_str))
            i = j + 1
            continue
        buf.append(ch)
        i += 1

    flush_buf()
    return tokens


def _get_parent_and_key(root: Dict[str, Any], field: str) -> Tuple[Any, _PathToken]:
    """
    Resolve a field path to (parent_container, final_key_or_index).
    """
    tokens = _parse_field_path(field)
    if not tokens:
        raise ValueError("field path resolved to empty tokens")

    cur: Any = root
    for tok in tokens[:-1]:
        if isinstance(tok, str):
            if not isinstance(cur, dict) or tok not in cur:
                raise KeyError(f"Path segment '{tok}' not found for field path {field!r}")
            cur = cur[tok]
        else:
            if not isinstance(cur, list):
                raise TypeError(f"Encountered list index {tok} but current value is not a list for {field!r}")
            if tok < 0 or tok >= len(cur):
                raise IndexError(f"List index {tok} out of range for {field!r}")
            cur = cur[tok]

    return cur, tokens[-1]

def load_all_core_memory() -> Dict[str, Dict[str, Any]]:
    """
    Loads all core memory blocks from disk.

    Intended usage: called at agent startup to hydrate always-in-context memory.
    """
    blocks: Dict[str, Dict[str, Any]] = {}
    for block_name in _CORE_BLOCK_FILES:
        blocks[block_name] = _read_json(_block_path(block_name))
    return blocks


def memory_replace(block: str, field: str, new_value: Any) -> None:
    """
    Precise update of an existing field (Letta-style memory_replace).

    - `block`: one of the fixed core blocks
    - `field`: top-level JSON field to replace
    - `new_value`: value to set
    """
    path = _block_path(block)
    data = _read_json(path)

    parent, key = _get_parent_and_key(data, field)
    if isinstance(key, str):
        if not isinstance(parent, dict) or key not in parent:
            raise KeyError(f"Field '{field}' not found in block '{block}'.")
        parent[key] = new_value
    else:
        if not isinstance(parent, list):
            raise TypeError(f"Field '{field}' resolves to an index but parent is not a list.")
        if key < 0 or key >= len(parent):
            raise IndexError(f"Index {key} out of range for field '{field}'.")
        parent[key] = new_value
    data["last_updated"] = _utc_now_iso()
    _atomic_write_json(path, data)


def memory_insert(block: str, field: str, value: Any) -> None:
    """
    Append-style update for list fields (Letta-style memory_insert).

    - `block`: one of the fixed core blocks
    - `field`: top-level JSON field expected to be a list
    - `value`: value to append to the list
    """
    path = _block_path(block)
    data = _read_json(path)

    parent, key = _get_parent_and_key(data, field)
    if isinstance(key, str):
        if not isinstance(parent, dict) or key not in parent:
            raise KeyError(f"Field '{field}' not found in block '{block}'.")
        current = parent[key]
    else:
        if not isinstance(parent, list):
            raise TypeError(f"Field '{field}' resolves to an index but parent is not a list.")
        if key < 0 or key >= len(parent):
            raise IndexError(f"Index {key} out of range for field '{field}'.")
        current = parent[key]

    if not isinstance(current, list):
        raise TypeError(
            f"Field '{field}' in block '{block}' must be a list for memory_insert; got {type(current).__name__}."
        )

    current.append(value)
    data["last_updated"] = _utc_now_iso()
    _atomic_write_json(path, data)


