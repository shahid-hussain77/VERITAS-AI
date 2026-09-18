"""Analysis history — store + retrieve past runs."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

from veritas.config import WORKSPACE_DIR


HISTORY_DIR = WORKSPACE_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_run(result: dict, name: str = "") -> str:
    """Save a run result. Returns run_id."""
    run_id = f"run_{int(time.time() * 1000)}"
    path = HISTORY_DIR / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "name": name or result.get("document", {}).get("title", "untitled"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return run_id


def list_runs(limit: int = 50) -> list[dict]:
    """List recent runs."""
    files = sorted(HISTORY_DIR.glob("run_*.json"), reverse=True)[:limit]
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            report = data.get("result", {}).get("report", {})
            out.append({
                "run_id": data.get("run_id"),
                "name": data.get("name"),
                "saved_at": data.get("saved_at"),
                "document": report.get("document_name", "?"),
                "findings": report.get("findings_count", 0),
                "confidence": report.get("score", {}).get("overall_confidence", "?"),
            })
        except Exception:
            continue
    return out


def load_run(run_id: str) -> Optional[dict]:
    """Load a run by ID."""
    path = HISTORY_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_run(run_id: str) -> bool:
    """Delete a run."""
    path = HISTORY_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


__all__ = ["save_run", "list_runs", "load_run", "delete_run"]