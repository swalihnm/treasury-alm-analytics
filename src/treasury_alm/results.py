"""Single writer for results/summary.json: every quoted number must pass through here."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import RESULTS


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def update_summary(section: str, payload: dict[str, Any]) -> None:
    """Replace `section` of results/summary.json with `payload` (file created if missing)."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "summary.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = _clean(payload)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_summary() -> dict[str, Any]:
    path = RESULTS / "summary.json"
    return dict(json.loads(path.read_text())) if path.exists() else {}
