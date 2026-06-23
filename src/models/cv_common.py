"""Shared utilities for cross-validation CLI scripts."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rich.console import Console

    _RICH_AVAILABLE = True
except ImportError:
    Console = None  # type: ignore[misc, assignment]
    _RICH_AVAILABLE = False

from src.models.hbb.variants import parse_csv_list

_CONSOLE = (
    Console(highlight=False)
    if _RICH_AVAILABLE and os.getenv("RCV_PLAIN") != "1"
    else None
)


class NullProgress:
    """No-op progress stand-in when Rich is unavailable."""

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def add_task(self, *args: object, **kwargs: object) -> int:
        return 0

    def update(self, *args: object, **kwargs: object) -> None:
        return None

    def advance(self, *args: object, **kwargs: object) -> None:
        return None


def cv_console() -> Console | None:
    return _CONSOLE


def cv_log(message: str = "", **kwargs: Any) -> None:
    if _CONSOLE is not None:
        _CONSOLE.print(message, **kwargs)
    else:
        print(message)


def fmt_float(value: float | None, ndigits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{ndigits}f}"


def format_exc(exc: BaseException) -> str:
    msg = str(exc).strip()
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return f"{type(exc).__name__} (no message)"


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def extract_sampler_diagnostics(
    model: Any, max_treedepth: int
) -> dict[str, float]:
    out = {"n_divergences": np.nan, "pct_max_treedepth": np.nan}
    try:
        ss = model.trace.sample_stats
        if "diverging" in ss:
            out["n_divergences"] = float(ss["diverging"].sum().values)
        if "tree_depth" in ss:
            td = ss["tree_depth"].values
            out["pct_max_treedepth"] = float(100.0 * (td >= max_treedepth).mean())
    except Exception:
        pass
    return out


__all__ = [
    "NullProgress",
    "cv_console",
    "cv_log",
    "extract_sampler_diagnostics",
    "fmt_float",
    "format_exc",
    "now_tag",
    "parse_csv_list",
    "write_json",
]
