# w2c_mlflow_logging.py
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Runtime dependency injection (to avoid circular imports)
# ---------------------------------------------------------------------

_GET_CHILD_ARTIFACT_DIR: Optional[Callable[[str], Path]] = None
_GET_CHILD_CHECKPOINT_DIR: Optional[Callable[[str], Path]] = None
_LOAD_MLFLOW_IDS: Optional[Callable[[], Dict[str, Any]]] = None
_SAVE_MLFLOW_IDS: Optional[Callable[[Dict[str, Any]], None]] = None
_UTC_NOW_ISO: Optional[Callable[[], str]] = None


def bind_runtime_dependencies(
    *,
    get_child_artifact_dir: Callable[[str], Path],
    get_child_checkpoint_dir: Callable[[str], Path],
    load_mlflow_ids: Callable[[], Dict[str, Any]],
    save_mlflow_ids: Callable[[Dict[str, Any]], None],
    utc_now_iso: Callable[[], str],
) -> None:
    """
    Bind required runtime functions from the main script.

    This module intentionally does NOT import your main script to avoid circular imports.
    You MUST call this once before using init_logging_context() or mlflow_child_run().
    """
    global _GET_CHILD_ARTIFACT_DIR, _GET_CHILD_CHECKPOINT_DIR, _LOAD_MLFLOW_IDS, _SAVE_MLFLOW_IDS, _UTC_NOW_ISO
    _GET_CHILD_ARTIFACT_DIR = get_child_artifact_dir
    _GET_CHILD_CHECKPOINT_DIR = get_child_checkpoint_dir
    _LOAD_MLFLOW_IDS = load_mlflow_ids
    _SAVE_MLFLOW_IDS = save_mlflow_ids
    _UTC_NOW_ISO = utc_now_iso


def _require_bound(fn: Optional[Callable[..., Any]], name: str) -> Callable[..., Any]:
    if fn is None:
        raise RuntimeError(
            f"{name} is not bound. Call bind_runtime_dependencies(...) before using w2c_mlflow_logging."
        )
    return fn


# ---------------------------------------------------------------------
# Local artifact context (session-scoped) + helpers
# ---------------------------------------------------------------------

@dataclass
class LoggingContext:
    artifact_dir: Path


_GLOBAL_LOG_CTX: Optional[LoggingContext] = None
_CURRENT_LLM_CALLS_REL: Optional[str] = None  # e.g. "llm_calls/mcq_logprob.jsonl"


def init_logging_context(exp_name: str = "parent") -> Path:
    """
    Create a stable local artifact directory (session-scoped) for an experiment name.
    """
    global _GLOBAL_LOG_CTX

    get_child_artifact_dir = _require_bound(_GET_CHILD_ARTIFACT_DIR, "_GET_CHILD_ARTIFACT_DIR")
    artifact_dir = get_child_artifact_dir(exp_name)

    for sub in ["inputs", "outputs", "llm_calls", "metrics"]:
        (artifact_dir / sub).mkdir(parents=True, exist_ok=True)

    _GLOBAL_LOG_CTX = LoggingContext(artifact_dir=artifact_dir)
    return artifact_dir


def _require_log_ctx() -> LoggingContext:
    """
    Return the active LoggingContext.
    Must be called after init_logging_context() (directly or via mlflow_child_run()).
    """
    if _GLOBAL_LOG_CTX is None:
        raise RuntimeError("Logging context is not initialized. Call init_logging_context() first.")
    return _GLOBAL_LOG_CTX


def log_json_artifact(obj: Any, rel_path: str) -> None:
    """
    Write a JSON artifact locally under the current artifact_dir.
    This does NOT upload to MLflow (upload happens in flush_* functions).
    """
    if _GLOBAL_LOG_CTX is None:
        return
    path = _GLOBAL_LOG_CTX.artifact_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def log_jsonl_artifact(list_of_obj: List[Any], rel_path: str) -> None:
    """
    Compatibility helper: writes a LIST as pretty JSON (even if the filename ends with .jsonl).
    For true JSONL append, use append_jsonl_artifact().
    """
    if _GLOBAL_LOG_CTX is None:
        return
    path = _GLOBAL_LOG_CTX.artifact_dir / rel_path
    if path.suffix == ".jsonl":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list_of_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl_artifact(obj: Any, rel_path: str) -> None:
    """
    Append a single JSON object to a JSONL file locally (one object per line).
    """
    if _GLOBAL_LOG_CTX is None:
        return
    path = _GLOBAL_LOG_CTX.artifact_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def set_current_llm_calls_rel(rel_path: Optional[str]) -> None:
    """
    Set the current JSONL relative path used by log_llm_call_incremental().
    """
    global _CURRENT_LLM_CALLS_REL
    _CURRENT_LLM_CALLS_REL = rel_path


def log_llm_call_incremental(call_log: Dict[str, Any]) -> None:
    """
    Append one LLM call record to the current llm_calls stream (if configured).
    """
    if _GLOBAL_LOG_CTX is None or not _CURRENT_LLM_CALLS_REL:
        return
    append_jsonl_artifact(call_log, _CURRENT_LLM_CALLS_REL)


# ---------------------------------------------------------------------
# MLflow upload helpers (best-effort)
# ---------------------------------------------------------------------

def _safe_log_artifacts(local_path: Path, artifact_path: str) -> None:
    """
    Best-effort MLflow artifact upload; never raises.
    """
    try:
        if local_path.exists():
            if local_path.is_dir():
                mlflow.log_artifacts(str(local_path), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    except Exception:
        logger.exception("Failed to log artifacts '%s' -> '%s'", local_path, artifact_path)


def flush_llm_calls(exp_name: str) -> None:
    """
    Upload llm_calls/<exp_name>.jsonl to MLflow.
    """
    ctx = _require_log_ctx()
    local_file = ctx.artifact_dir / f"llm_calls/{exp_name}.jsonl"
    _safe_log_artifacts(local_file, "llm_calls")


def flush_outputs(exp_dir_name: str) -> None:
    """
    Upload outputs/<exp_dir_name>/ directory to MLflow.
    """
    ctx = _require_log_ctx()
    _safe_log_artifacts(ctx.artifact_dir / "outputs" / exp_dir_name, f"outputs/{exp_dir_name}")


def flush_inputs_parent() -> None:
    """
    Upload inputs/ directory to MLflow (parent scope).
    """
    ctx = _require_log_ctx()
    _safe_log_artifacts(ctx.artifact_dir / "inputs", "inputs")


def flush_checkpoints(exp_name: str) -> None:
    """
    Upload checkpoints/<exp_name>/ directory to MLflow.
    """
    get_child_checkpoint_dir = _require_bound(_GET_CHILD_CHECKPOINT_DIR, "_GET_CHILD_CHECKPOINT_DIR")
    ckpt_dir = get_child_checkpoint_dir(exp_name)
    _safe_log_artifacts(ckpt_dir, f"checkpoints/{exp_name}")


# ---------------------------------------------------------------------
# MLflow child run context manager (with resume)
# ---------------------------------------------------------------------

@contextmanager
def mlflow_child_run(exp_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Nested MLflow run with:
      - session-scoped local artifacts under artifacts_local/<exp_name>
      - llm_calls jsonl stream setup
      - resume support via mlflow_ids.json child_run_ids[exp_name]
      - artifact uploads on exit (llm_calls, outputs, checkpoints)
    """
    load_ids = _require_bound(_LOAD_MLFLOW_IDS, "_LOAD_MLFLOW_IDS")
    save_ids = _require_bound(_SAVE_MLFLOW_IDS, "_SAVE_MLFLOW_IDS")
    utc_now_iso = _require_bound(_UTC_NOW_ISO, "_UTC_NOW_ISO")
    get_child_artifact_dir = _require_bound(_GET_CHILD_ARTIFACT_DIR, "_GET_CHILD_ARTIFACT_DIR")

    global _GLOBAL_LOG_CTX

    artifact_dir = get_child_artifact_dir(exp_name)
    for sub in ["inputs", "outputs", "llm_calls", "metrics"]:
        (artifact_dir / sub).mkdir(parents=True, exist_ok=True)
    _GLOBAL_LOG_CTX = LoggingContext(artifact_dir=artifact_dir)

    llm_calls_rel = f"llm_calls/{exp_name}.jsonl"
    set_current_llm_calls_rel(llm_calls_rel)

    local_llm_calls = artifact_dir / llm_calls_rel
    local_llm_calls.parent.mkdir(parents=True, exist_ok=True)
    if not local_llm_calls.exists():
        local_llm_calls.write_text("", encoding="utf-8")

    ids = load_ids()
    child_run_ids = ids.get("child_run_ids")
    if not isinstance(child_run_ids, dict):
        child_run_ids = {}
        ids["child_run_ids"] = child_run_ids

    child_run_id = child_run_ids.get(exp_name)
    resumed = False
    run_ctx = None

    if isinstance(child_run_id, str) and child_run_id.strip():
        try:
            run_ctx = mlflow.start_run(run_id=child_run_id, nested=True)
            resumed = True
            logger.info("[MLFLOW] Resumed child run exp=%s run_id=%s", exp_name, child_run_id)
        except Exception:
            logger.warning("[MLFLOW] Failed to resume child run exp=%s; creating a new run.", exp_name)
            run_ctx = None
            resumed = False

    if run_ctx is None:
        run_ctx = mlflow.start_run(run_name=exp_name, nested=True)
        active = mlflow.active_run()
        if active is not None:
            child_run_ids[exp_name] = active.info.run_id
            ids["child_run_ids"] = child_run_ids
            save_ids(ids)
            logger.info("[MLFLOW] Created child run exp=%s run_id=%s", exp_name, active.info.run_id)

    with run_ctx:
        try:
            try:
                mlflow.set_tag("child_run_resumed", str(resumed).lower())
                mlflow.set_tag("child_exp_name", exp_name)
                mlflow.set_tag(("resumed_at_utc" if resumed else "created_at_utc"), utc_now_iso())
            except Exception:
                logger.exception("Failed to set child run tags for exp=%s", exp_name)

            if tags:
                try:
                    mlflow.set_tags(tags)
                except Exception:
                    logger.exception("Failed to set tags for exp=%s", exp_name)

            yield

        finally:
            try:
                flush_llm_calls(exp_name)
            except Exception:
                logger.exception("Failed to flush llm_calls for exp=%s", exp_name)

            try:
                flush_outputs(exp_name)
            except Exception:
                logger.exception("Failed to flush outputs for exp=%s", exp_name)

            try:
                flush_checkpoints(exp_name)
            except Exception:
                logger.exception("Failed to flush checkpoints for exp=%s", exp_name)

            set_current_llm_calls_rel(None)
