from __future__ import annotations

# with harmony prompt added for gpt-oss: (13/01/26)
# WITH API SEED and no reasoning gpt for mcq string-based 11 Jannuary
# 8 January added audit fallbacks in the checkpoints folders
# with checkpoints and stability with new metrics
## SISTEMATO LIMITE MCQ
# version 4 30 dicembre

import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from w2c_config import load_config, config_for_fingerprint, config_pretty

from w2c_mlflow_logging import (
    bind_runtime_dependencies,
    init_logging_context,
    log_json_artifact,
    log_jsonl_artifact,
    append_jsonl_artifact,
    log_llm_call_incremental,
    flush_inputs_parent,
    mlflow_child_run,
)

import w2c_prompts as prompts
import w2c_templates as templates


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =============================================================================
# ENV / CONFIG (TOML + .env via w2c_config)
# =============================================================================

# Repo root = parent of code_divided/ (fallback to cwd if running in notebook without __file__)
if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    REPO_ROOT = Path.cwd().resolve()

# Prefer local config.toml, fallback to config.example.toml
_CFG_PATH = REPO_ROOT / "config.toml"
if not _CFG_PATH.exists():
    _CFG_PATH = REPO_ROOT / "config.example.toml"

CFG: Dict[str, Any] = load_config(str(_CFG_PATH))

# Expose config objects to notebook runner / session fingerprinting
W2C_CONFIG: Dict[str, Any] = CFG
W2C_CONFIG_FOR_FINGERPRINT: Dict[str, Any] = config_for_fingerprint(CFG)
W2C_CONFIG_PATH: str = str(_CFG_PATH.resolve())

# Provider: JRC
TOKEN_JRC = CFG["providers"]["jrc"]["token"]
BASE_URL = str(CFG["providers"]["jrc"]["base_url"]).rstrip("/")

HEADERS = {
    "Authorization": f"Bearer {TOKEN_JRC}" if TOKEN_JRC else "",
    "Content-Type": "application/json",
}

# HTTP / retries
MAX_RETRIES = int(CFG["http"]["max_retries"])
RETRY_SLEEP_SECONDS = float(CFG["http"]["retry_sleep_seconds"])
BASE_DELAY_SECONDS = float(CFG["http"]["base_delay_seconds"])
HTTP_TIMEOUT_SECONDS = int(CFG["http"].get("timeout_seconds", 60))

# Reasoning
REASONING_MODELS = {str(m) for m in CFG["models"].get("reasoning_models", [])}
REASONING_EFFORT = CFG["models"].get("reasoning_effort", "low")  # "" or None to disable

# Debug flags
DEBUG_MCQ_TOKEN_MISMATCH = bool(CFG["debug"].get("debug_mcq_token_mismatch", False))
DEBUG_MCQ_TOKEN_MISMATCH_MAX = int(CFG["debug"].get("debug_mcq_token_mismatch_max", 10**9))
# Debug helper for MCQ logprob scoring only:
# Purpose: diagnose prompt/choice boundary drift caused by tokenizer/prefix mismatches, delimiter handling,
# or whitespace/template changes that can make logprob attribution incorrect (e.g., scoring includes/excludes
# unintended tokens and shifts the chosen option).
# When enabled, it runs an additional per-choice tokenization comparison (old char-offset slicing vs
# LM-eval-style token-split using the prompt token count / LCP fallback) and logs warnings on mismatches.
# This does NOT change the scoring logic or the selected label directly, but it increases /completions echo calls
# (more latency/API load), which can indirectly influence runs via retries/rate-limits/timeouts and fallbacks.
# Not used by MCQ string-based or LLM-as-judge.

PRINT_TEMPLATE_FAMILY_ONCE = bool(CFG["debug"].get("print_template_family_once", True))
_PRINTED_TEMPLATE_FAMILY = False

# =============================================================================
# HTTP CLIENT (RETRIES)
# =============================================================================

# =============================================================================
# MULTI-PROVIDER CONFIG (JRC + GEMINI)
# =============================================================================

TOKEN_GEMINI = CFG["providers"]["gemini"]["token"]
GEMINI_BASE_URL = str(CFG["providers"]["gemini"]["base_url"]).rstrip("/")

GEMINI_MODEL_PREFIXES = tuple(
    p.strip().lower()
    for p in CFG["providers"]["gemini"].get("model_prefixes", ["gemini-"])
    if isinstance(p, str) and p.strip()
)

# API seed is part of config now (used by chat/completions + /completions)
API_SEED = int(CFG["run"]["api_seed"])


def _is_gemini_model(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return any(m.startswith(p) for p in GEMINI_MODEL_PREFIXES)


def _provider_for_model(model_name: str) -> str:
    return "gemini" if _is_gemini_model(model_name) else "jrc"


def _headers_for_provider(provider: str) -> Dict[str, str]:
    if provider == "gemini":
        if not TOKEN_GEMINI:
            raise RuntimeError(
                "Model Gemini richiesto ma manca TOKEN_GEMINI / GEMINI_API_KEY."
            )
        return {
            "Authorization": f"Bearer {TOKEN_GEMINI}",
            "Content-Type": "application/json",
        }

    # default: JRC
    if not TOKEN_JRC:
        raise RuntimeError(
            "Model JRC richiesto ma manca TOKEN_JRC / OPENAI_API_KEY."
        )
    return HEADERS


def _base_url_for_provider(provider: str) -> str:
    if provider == "gemini":
        return GEMINI_BASE_URL
    return BASE_URL


_SESSION = requests.Session()


def _post_with_retries(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: int = HTTP_TIMEOUT_SECONDS,
    provider: Optional[str] = None,
) -> requests.Response:
    """
    Send a POST request with retry/backoff on timeouts, 429, and 5xx responses.
    Supports provider-specific headers (e.g., JRC vs Gemini OpenAI-compatible endpoint).

    Backward-compatible usage:
        _post_with_retries(url, payload)

    New usage:
        _post_with_retries(url, payload, headers=..., provider="gemini")
    """
    last_resp: Optional[requests.Response] = None
    last_exc: Optional[Exception] = None

    req_headers = headers or HEADERS
    provider_name = provider or "default"

    def _retry_sleep_for_response(resp: requests.Response) -> float:
        """
        Prefer Retry-After if present and parseable, otherwise use RETRY_SLEEP_SECONDS.
        """
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                # Usually integer seconds
                v = float(retry_after.strip())
                if v > 0:
                    return v
            except Exception:
                pass
        return RETRY_SLEEP_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.post(
                url,
                headers=req_headers,
                json=payload,
                timeout=timeout_seconds,
            )
            last_exc = None
        except requests.exceptions.Timeout as e:
            last_exc = e
            logging.error(
                "[HTTP][%s] Timeout on attempt %d/%d; sleeping %.1f seconds. url=%s",
                provider_name,
                attempt,
                MAX_RETRIES,
                RETRY_SLEEP_SECONDS,
                url,
            )
            last_resp = None
            time.sleep(RETRY_SLEEP_SECONDS)
            continue
        except requests.exceptions.RequestException as e:
            last_exc = e
            logging.error(
                "[HTTP][%s] Request error on attempt %d/%d: %s; sleeping %.1f seconds. url=%s",
                provider_name,
                attempt,
                MAX_RETRIES,
                e,
                RETRY_SLEEP_SECONDS,
                url,
            )
            last_resp = None
            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        last_resp = resp

        # Retry policy: 429 + 5xx (+ 408 gateway/request timeout if a proxy returns it)
        retryable = (resp.status_code in {408, 429}) or (500 <= resp.status_code < 600)
        if not retryable:
            if BASE_DELAY_SECONDS > 0:
                time.sleep(BASE_DELAY_SECONDS)
            return resp

        sleep_s = _retry_sleep_for_response(resp)
        logging.warning(
            "[HTTP][%s] HTTP %d (retryable) on attempt %d/%d; sleeping %.1f seconds. url=%s",
            provider_name,
            resp.status_code,
            attempt,
            MAX_RETRIES,
            sleep_s,
            url,
        )
        time.sleep(sleep_s)

    if BASE_DELAY_SECONDS > 0:
        time.sleep(BASE_DELAY_SECONDS)

    if last_resp is not None:
        return last_resp

    if last_exc is not None:
        raise RuntimeError(f"All retries failed with request error/timeout. Last error: {last_exc}")

    raise RuntimeError("All retries failed with timeout / request errors.")


# =============================================================================
# CHECKPOINTING (PERSISTENT, UUID-LEVEL, APPEND-ONLY JSONL)
# =============================================================================

def _atomic_append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_jsonl_as_map(path: Path, key_field: str = "uuid") -> Dict[str, Dict[str, Any]]:
    """
    Loads a JSONL file into a dict keyed by key_field.
    If duplicated keys exist, last one wins (append-only logs are okay).
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = obj.get(key_field)
            if isinstance(k, str) and k:
                out[k] = obj
    return out


# =============================================================================
# MULTI-SESSION RUN MANAGER (PER RUN_KEY)
#   - più valutazioni/config sotto la stessa cartella runs/<RUN_KEY>/
#   - sessione selezionata via fingerprint deterministica della config
# =============================================================================

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

SESSION_DIR: Optional[Path] = None
SESSION_FINGERPRINT: Optional[str] = None


def _utc_now_iso() -> str:
    """
    Return current UTC time as ISO-8601 string (seconds precision).
    Used in manifests, markers, and MLflow tags for traceability.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_json_dumps(obj: Any) -> str:
    """
    Serialize an object into canonical JSON (sorted keys, compact separators).
    Used to compute deterministic config fingerprints across executions.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_config_fingerprint(config: Dict[str, Any], algo: str = "sha256", n: int = 16) -> str:
    """
    Fingerprint deterministica (stabile) della config.
    - Usa JSON canonico (sort_keys=True, separators compatti)
    - Hash sha256 di default
    - Ritorna i primi n caratteri (16 consigliato).
    """
    payload = _stable_json_dumps(config).encode("utf-8")
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()[:n]


def _mlflow_ids_path() -> Path:
    """
    Path to mlflow_ids.json inside the active SESSION_DIR.
    Stores parent/child run_ids to enable resume across restarts.
    """
    return _require_session_dir() / "mlflow_ids.json"


def _load_mlflow_ids() -> Dict[str, Any]:
    """
    Load mlflow_ids.json for the current session (or return defaults if missing).
    Used to resume parent/child MLflow runs deterministically.
    """
    p = _mlflow_ids_path()
    if not p.exists():
        return {"parent_run_id": None, "child_run_ids": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_mlflow_ids(obj: Dict[str, Any]) -> None:
    """
    Atomically write mlflow_ids.json to the current session directory.
    Call whenever parent/child run IDs are created to make resume crash-safe.
    """
    _write_json_atomic(_mlflow_ids_path(), obj)


@dataclass
class SessionInfo:
    fingerprint: str
    session_dir: Path
    manifest_path: Path
    config: Dict[str, Any]
    created_at: str
    updated_at: str
    schema_version: int = 1


def _sessions_root(run_dir: Path) -> Path:
    """
    Return the sessions root directory (run_dir/sessions), creating it if missing.
    Each session is a deterministic fingerprint of the evaluation config.
    """
    root = run_dir / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_sessions(run_dir: Path) -> List[SessionInfo]:
    """
    List existing sessions under run_dir/sessions by reading their manifest.json files.
    Useful for auditing or selecting prior runs/configurations.
    """
    out: List[SessionInfo] = []
    root = _sessions_root(run_dir)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        try:
            m = _read_json(manifest)
            out.append(
                SessionInfo(
                    fingerprint=m["fingerprint"],
                    session_dir=d,
                    manifest_path=manifest,
                    config=m.get("config", {}),
                    created_at=m.get("created_at", ""),
                    updated_at=m.get("updated_at", ""),
                    schema_version=int(m.get("schema_version", 1)),
                )
            )
        except Exception:
            logging.exception("Bad manifest: %s", manifest)
    return out


def _maybe_migrate_legacy_layout(run_dir: Path) -> None:
    """
    Migration:
    Se esiste run_dir/checkpoints/<metodo> o run_dir/artifacts_local/<metodo> (layout legacy),
    create a 'legacy' session if it is note there already.
    """
    legacy_checkpoints = run_dir / "checkpoints"
    legacy_artifacts = run_dir / "artifacts_local"
    if not legacy_checkpoints.exists() and not legacy_artifacts.exists():
        return

    root = _sessions_root(run_dir)
    legacy_dir = root / "legacy"
    manifest = legacy_dir / "manifest.json"

    if manifest.exists():
        return

    legacy_dir.mkdir(parents=True, exist_ok=True)

    m = {
        "schema_version": 1,
        "fingerprint": "legacy",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "config": {"note": "legacy layout (pre multi-session)"},
        "legacy_paths": {
            "checkpoints": str(legacy_checkpoints) if legacy_checkpoints.exists() else None,
            "artifacts_local": str(legacy_artifacts) if legacy_artifacts.exists() else None,
        },
    }
    _write_json_atomic(manifest, m)
    logging.info("[SESSIONS] Legacy layout detected. Created legacy manifest at: %s", manifest)


def load_or_create_session(run_dir: Path, config: Dict[str, Any]) -> SessionInfo:
    """
    Se esiste la sessione per fingerprint(config) => resume.
    Altrimenti crea nuova sessione.
    """
    _maybe_migrate_legacy_layout(run_dir)

    fp = compute_config_fingerprint(config)
    root = _sessions_root(run_dir)
    session_dir = root / fp
    manifest = session_dir / "manifest.json"

    now = _utc_now_iso()

    if manifest.exists():
        m = _read_json(manifest)
        m["updated_at"] = now
        _write_json_atomic(manifest, m)
        return SessionInfo(
            fingerprint=m["fingerprint"],
            session_dir=session_dir,
            manifest_path=manifest,
            config=m.get("config", {}),
            created_at=m.get("created_at", ""),
            updated_at=m.get("updated_at", ""),
            schema_version=int(m.get("schema_version", 1)),
        )

    session_dir.mkdir(parents=True, exist_ok=True)

    m = {
        "schema_version": 1,
        "fingerprint": fp,
        "created_at": now,
        "updated_at": now,
        "config": config,
    }
    _write_json_atomic(manifest, m)

    # pre-crea struttura
    for sub in ["checkpoints", "artifacts_local"]:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)

    logging.info("[SESSIONS] Created new session: %s", session_dir)
    return SessionInfo(
        fingerprint=fp,
        session_dir=session_dir,
        manifest_path=manifest,
        config=config,
        created_at=now,
        updated_at=now,
        schema_version=1,
    )


def _require_session_dir() -> Path:
    """
    Return the active SESSION_DIR bound to the current config fingerprint.
    Must be set via init_or_resume_session() before session-scoped operations.
    """
    if SESSION_DIR is None:
        raise RuntimeError("SESSION_DIR is None. Call init_or_resume_session() first.")
    return SESSION_DIR


def init_or_resume_session(run_dir: Path, config: Dict[str, Any]) -> SessionInfo:
    """
    Setta globals SESSION_DIR/SESSION_FINGERPRINT e ritorna SessionInfo.
    """
    global SESSION_DIR, SESSION_FINGERPRINT
    s = load_or_create_session(run_dir, config=config)
    SESSION_DIR = s.session_dir
    SESSION_FINGERPRINT = s.fingerprint
    logging.info("[SESSIONS] Using session fingerprint=%s dir=%s", s.fingerprint, s.session_dir)
    return s


# -----------------------------
# METHOD COMPLETION MARKERS
# -----------------------------

def _method_done_marker(exp_name: str) -> Path:
    """
    Return the path to the completion marker file for a pipeline in this session.
    Marker presence is used to skip already-completed pipelines safely.
    """
    return _require_session_dir() / "checkpoints" / exp_name / "_DONE.json"


def mark_method_done(exp_name: str, payload: Dict[str, Any]) -> None:
    """
    Crea marker di completamento per exp_name. Idempotente (sovrascrive atomicamente).
    """
    marker = _method_done_marker(exp_name)
    obj = {
        "exp_name": exp_name,
        "done_at": _utc_now_iso(),
        "session_fingerprint": SESSION_FINGERPRINT,
        **payload,
    }
    _write_json_atomic(marker, obj)


def is_method_done(exp_name: str) -> bool:
    """
    Check whether a pipeline completion marker exists for exp_name in this session.
    Used to implement idempotent skip/resume behavior.
    """
    return _method_done_marker(exp_name).exists()


def method_status_summary(exp_name: str, ckpt_main_file: Optional[Path], expected_n: Optional[int] = None) -> Dict[str, Any]:
    """
    Ritorna stato sintetico:
      - done: marker presente
      - n_records: righe checkpoint (se jsonl)
      - expected_n: target (opzionale)
      - percent: se expected_n noto
    """
    done = is_method_done(exp_name)
    n_records = None
    if ckpt_main_file is not None and ckpt_main_file.exists():
        try:
            with ckpt_main_file.open("r", encoding="utf-8") as f:
                n_records = sum(1 for _ in f if _.strip())
        except Exception:
            n_records = None

    pct = None
    if expected_n and isinstance(n_records, int):
        pct = float(n_records) / float(expected_n) if expected_n else None

    return {
        "exp_name": exp_name,
        "done": done,
        "checkpoint_file": str(ckpt_main_file) if ckpt_main_file else None,
        "n_records": n_records,
        "expected_n": expected_n,
        "progress": pct,
    }


def should_run_method(exp_name: str) -> bool:
    """
    Policy:
      - se marker done esiste => skip
      - altrimenti => run/resume
    """
    return not is_method_done(exp_name)


def get_child_checkpoint_dir(exp_name: str) -> Path:
    """
    Directory for persistent checkpoints for a child run/pipeline.
    NOW: scoped to SESSION_DIR, so multiple configs can share the same RUN_KEY safely.
    """
    base = _require_session_dir() / "checkpoints" / exp_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def append_audit_event(exp_name: str, event: Dict[str, Any]) -> None:
    """
    Append an audit event to checkpoints/<exp_name>/audit_fallbacks.jsonl.
    Never raises. Session-scoped (uses get_child_checkpoint_dir).
    """
    try:
        ckpt_dir = get_child_checkpoint_dir(exp_name)
        path = ckpt_dir / "audit_fallbacks.jsonl"

        base = {
            "ts_utc": _utc_now_iso(),
            "run_key": RUN_KEY,
            "session_fingerprint": SESSION_FINGERPRINT,
            "exp_name": exp_name,
            "api_seed": API_SEED,
        }
        rec = {**base, **(event or {})}
        _atomic_append_jsonl(path, rec)
    except Exception:
        logging.exception("[AUDIT] Failed to append audit event for exp=%s", exp_name)


# =============================================================================
# FALLBACK SUMMARY HELPERS (for metrics.json enrichment)
# =============================================================================

FORCED_FALLBACK_TYPES = {
    # hard fallbacks that directly force a label/output
    "judge_json_parse_failed_second_fallback_to_cannot_answer",
    "all_logprobs_-inf_string_fallback",
    "invalid_label_coercion_to_cannot_answer",
    "missing_prediction_uuid",
}


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file (one JSON object per line). Returns [] if missing.
    Never raises (best-effort).
    """
    out: List[Dict[str, Any]] = []
    try:
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        logging.exception("Failed to read JSONL: %s", path)
    return out


def summarize_audit_fallbacks(exp_name: str) -> Dict[str, Any]:
    """
    Summarize checkpoints/<exp_name>/audit_fallbacks.jsonl:
      - total events
      - unique uuids affected
      - event counts by fallback_type / stage / severity
      - unique-uuid counts by fallback_type
      - forced-fallback totals (subset of fallback types)
    """
    ckpt_dir = get_child_checkpoint_dir(exp_name)
    path = ckpt_dir / "audit_fallbacks.jsonl"

    events = _read_jsonl_records(path)

    by_type = Counter()
    by_stage = Counter()
    by_severity = Counter()

    uuids_all: set[str] = set()
    uuids_by_type: Dict[str, set[str]] = defaultdict(set)

    forced_uuids: set[str] = set()
    forced_events = 0

    for e in events:
        ft = e.get("fallback_type") or "UNKNOWN"
        st = e.get("stage") or "UNKNOWN"
        sv = e.get("severity") or "UNKNOWN"
        u = e.get("uuid")

        by_type[str(ft)] += 1
        by_stage[str(st)] += 1
        by_severity[str(sv)] += 1

        if isinstance(u, str) and u:
            uuids_all.add(u)
            uuids_by_type[str(ft)].add(u)

            if str(ft) in FORCED_FALLBACK_TYPES:
                forced_uuids.add(u)

        if str(ft) in FORCED_FALLBACK_TYPES:
            forced_events += 1

    # stable, JSON-friendly output
    return {
        "audit_path": str(path),
        "n_events_total": int(len(events)),
        "n_uuids_with_any_event": int(len(uuids_all)),
        "events_by_fallback_type": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "events_by_stage": dict(sorted(by_stage.items(), key=lambda kv: (-kv[1], kv[0]))),
        "events_by_severity": dict(sorted(by_severity.items(), key=lambda kv: (-kv[1], kv[0]))),
        "uuids_by_fallback_type": dict(
            sorted(((k, len(v)) for k, v in uuids_by_type.items()), key=lambda kv: (-kv[1], kv[0]))
        ),
        "n_forced_fallback_events": int(forced_events),
        "n_forced_fallback_uuids": int(len(forced_uuids)),
        "forced_fallback_types": sorted(FORCED_FALLBACK_TYPES),
    }


def recompute_mcq_logprob_mode_counts_from_ckpt() -> Dict[str, int]:
    """
    Recompute mode_counts for mcq_logprob across ALL uuids (important on RESUME),
    reading checkpoints/mcq_logprob/mcq_logprob_predictions.jsonl.
    """
    ckpt_dir = get_child_checkpoint_dir("mcq_logprob")
    pred_ckpt = ckpt_dir / "mcq_logprob_predictions.jsonl"
    existing = _load_jsonl_as_map(pred_ckpt, key_field="uuid")

    c = Counter()
    for rec in existing.values():
        mode = rec.get("mode")
        if isinstance(mode, str) and mode:
            c[mode] += 1
        else:
            c["UNKNOWN"] += 1

    return {
        "llama_scoring": int(c.get("llama_scoring", 0)),
        "string_fallback": int(c.get("string_fallback", 0)),
        "UNKNOWN": int(c.get("UNKNOWN", 0)),
    }


def summarize_mcq_logprob_string_fallback_details() -> Dict[str, int]:
    """
    Extra counters for mcq_logprob: within string_fallback cases, how many had pred_idx == -1, etc.
    """
    ckpt_dir = get_child_checkpoint_dir("mcq_logprob")
    pred_ckpt = ckpt_dir / "mcq_logprob_predictions.jsonl"
    existing = _load_jsonl_as_map(pred_ckpt, key_field="uuid")

    n_string_fallback = 0
    n_string_fallback_pred_idx_minus1 = 0

    for rec in existing.values():
        if rec.get("mode") != "string_fallback":
            continue
        n_string_fallback += 1
        v = rec.get("string_fallback_pred_idx")
        try:
            if int(v) == -1:
                n_string_fallback_pred_idx_minus1 += 1
        except Exception:
            pass

    return {
        "n_string_fallback": int(n_string_fallback),
        "n_string_fallback_pred_idx_minus1": int(n_string_fallback_pred_idx_minus1),
    }


def get_child_artifact_dir(exp_name: str) -> Path:
    """
    Directory for local artifacts to be later uploaded to MLflow for a child run.
    NOW: scoped to SESSION_DIR.
    """
    base = _require_session_dir() / "artifacts_local" / exp_name
    base.mkdir(parents=True, exist_ok=True)
    return base


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class When2CallExample:
    uuid: str
    source: str
    source_id: str
    question: str
    orig_question: Optional[str]
    correct_answer: str
    answers: Dict[str, str]
    target_tool: Optional[str]
    tools_raw: List[Any]
    orig_tools: List[Any]
    held_out_param: Optional[str]

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "When2CallExample":
        return When2CallExample(
            uuid=obj["uuid"],
            source=obj.get("source", ""),
            source_id=obj.get("source_id", ""),
            question=obj["question"],
            orig_question=obj.get("orig_question"),
            correct_answer=obj["correct_answer"],
            answers=obj.get("answers", {}),
            target_tool=obj.get("target_tool"),
            tools_raw=obj.get("tools", []),
            orig_tools=obj.get("orig_tools", []),
            held_out_param=obj.get("held_out_param"),
        )


@dataclass
class ModelResponse:
    uuid: str
    raw_text: str


@dataclass
class JudgeDecision:
    uuid: str
    predicted_label: str
    judge_raw: str
    judge_rationale: Optional[str] = None

    # repair/fallback audit flags (persist in checkpoints)
    judge_parse_failed_first: bool = False
    judge_parse_failed_second: bool = False
    judge_used_retry: bool = False
    judge_fallback_to_cannot_answer: bool = False
    judge_exception_first: Optional[str] = None
    judge_exception_second: Optional[str] = None


@dataclass
class EvaluationMetrics:
    accuracy: float
    macro_f1: float
    macro_f1_no_direct: float
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: Dict[str, Dict[str, int]]
    tool_hallucination_rate: Optional[float] = None
    answer_hallucination_rate: Optional[float] = None
    parameter_hallucination_rate: Optional[float] = None
    n_total: int = 0
    n_missing_uuid: int = 0
    n_invalid_label: int = 0
    n_converted_to_cannot_answer: int = 0


# =============================================================================
# DATA LOADING / SUBSAMPLE
# =============================================================================

def load_when2call_jsonl(path: str) -> List[When2CallExample]:
    """
    Load a When2Call JSONL dataset into typed When2CallExample objects.
    Validates that gold labels are in BEHAVIOR_LABELS and returns all examples.
    """
    logging.info("Loading When2Call from %s", path)
    examples: List[When2CallExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ex = When2CallExample.from_json(obj)
            if ex.correct_answer not in templates.BEHAVIOR_LABELS:
                raise ValueError(f"Unexpected correct_answer '{ex.correct_answer}' (uuid={ex.uuid})")
            examples.append(ex)
    logging.info("Loaded %d examples", len(examples))
    return examples


def stratified_subsample_by_label(
    examples: List[When2CallExample],
    n_per_label: int,
    seed: int = 42,
) -> List[When2CallExample]:
    """
    Build a balanced subset by sampling up to n_per_label per gold label.
    Uses a fixed RNG seed for reproducibility and shuffles the final sample.
    """
    rng = random.Random(seed)
    by_label: Dict[str, List[When2CallExample]] = defaultdict(list)
    for ex in examples:
        by_label[ex.correct_answer].append(ex)

    subsampled: List[When2CallExample] = []
    for label in templates.BEHAVIOR_LABELS:
        group = by_label.get(label, [])
        if not group:
            logging.warning("No examples for label '%s'.", label)
            continue
        rng.shuffle(group)
        if len(group) < n_per_label:
            logging.warning(
                "Only %d examples for label '%s' (< %d). Using all of them.",
                len(group),
                label,
                n_per_label,
            )
            subsampled.extend(group)
        else:
            subsampled.extend(group[:n_per_label])

    rng.shuffle(subsampled)
    logging.info(
        "Stratified subsample: %d examples (target %d per %d labels)",
        len(subsampled),
        n_per_label,
        len(templates.BEHAVIOR_LABELS),
    )
    return subsampled


# =============================================================================
# LLM API WRAPPERS
# =============================================================================

def raw_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call /chat/completions with robust parsing across provider response shapes.
    Routes automatically to JRC or Gemini (OpenAI-compatible) based on model name prefix.

    Notes:
      - Gemini support here is for chat/completions only (LLM-as-judge, MCQ string-based).
      - /completions echo+logprobs is NOT handled by Gemini in this script path.
      - Keeps MCQ digit extraction logic to avoid reasoning/final split issues.
    """
    provider = _provider_for_model(model)
    base_url = _base_url_for_provider(provider)
    url = f"{base_url}/chat/completions"
    req_headers = _headers_for_provider(provider)

    # Copy messages defensively (we may rewrite roles on retry)
    msgs: List[Dict[str, str]] = [dict(m) for m in (messages or [])]

    # Provider-specific payload shaping:
    # - JRC: keep full OpenAI/JRC parameters + seed + penalties + optional reasoning
    # - Gemini(OpenAI-compat): keep payload minimal to avoid incompatibilities
    payload: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider != "gemini":
        payload.update(
            {
                "seed": API_SEED,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            }
        )

        if REASONING_EFFORT and model in REASONING_MODELS:
            # Disable reasoning for strict-output MCQ to avoid "reasoning/final" split issues
            if (meta or {}).get("pipeline") != "mcq_classifier":
                payload["reasoning"] = {"effort": REASONING_EFFORT}
    else:
        # Gemini OpenAI-compatible: keep a conservative payload.
        # top_p is often supported, but omitting it reduces incompatibility surface.
        # seed/reasoning/penalties often vary by adapter and can trigger 400.
        pass

    def _extract_first_mcq_digit(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        m = re.search(r"\b([0-3])\b", s)
        return m.group(1) if m else None

    def _is_role_rejection(resp: requests.Response) -> bool:
        if resp.ok:
            return False
        txt = (resp.text or "")[:2000].lower()
        return (
            ("developer" in txt and "role" in txt)
            or ("unsupported" in txt and "developer" in txt)
            or ("invalid" in txt and "role" in txt and "developer" in txt)
        )

    t0 = time.perf_counter()
    resp = _post_with_retries(
        url,
        payload,
        headers=req_headers,
        provider=provider,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Retry once if gateway/provider rejects role='developer'
    if (not resp.ok) and any(m.get("role") == "developer" for m in (msgs or [])) and _is_role_rejection(resp):
        logging.warning(
            "[CHAT][%s] Gateway/provider rejected developer role; retrying with merged system message.",
            provider,
        )
        merged_messages = templates._merge_developer_into_system_messages(msgs)
        payload_retry = dict(payload)
        payload_retry["messages"] = merged_messages

        t1 = time.perf_counter()
        resp = _post_with_retries(
            url,
            payload_retry,
            headers=req_headers,
            provider=provider,
            timeout_seconds=HTTP_TIMEOUT_SECONDS,
        )
        latency_ms = (time.perf_counter() - t1) * 1000.0
        payload = payload_retry  # keep final payload for logging consistency

    # Optional Gemini compatibility retry (rare): if "max_tokens" is rejected, try "max_completion_tokens"
    if (not resp.ok) and provider == "gemini":
        body_preview = (resp.text or "")[:2000].lower()
        if ("max_tokens" in body_preview and ("unsupported" in body_preview or "unknown" in body_preview or "invalid" in body_preview)):
            logging.warning(
                "[CHAT][gemini] 'max_tokens' rejected by endpoint; retrying with 'max_completion_tokens'."
            )
            payload_retry2 = dict(payload)
            payload_retry2.pop("max_tokens", None)
            payload_retry2["max_completion_tokens"] = max_tokens

            t2 = time.perf_counter()
            resp = _post_with_retries(
                url,
                payload_retry2,
                headers=req_headers,
                provider=provider,
                timeout_seconds=HTTP_TIMEOUT_SECONDS,
            )
            latency_ms = (time.perf_counter() - t2) * 1000.0
            payload = payload_retry2

    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    content = ""
    mcq_digit: Optional[str] = None
    pipeline = (meta or {}).get("pipeline")

    try:
        candidate_texts: List[str] = []

        if isinstance(data, dict):
            # -----------------------------------------------------------------
            # OpenAI-compatible shapes (JRC / Gemini OpenAI-compatible / adapters)
            # -----------------------------------------------------------------
            choices = data.get("choices") or []
            if choices and isinstance(choices, list):
                choice0 = choices[0] if len(choices) > 0 else None
                if isinstance(choice0, dict):
                    message = choice0.get("message")

                    # OpenAI-like: choices[0].message
                    if isinstance(message, dict):
                        raw_c = message.get("content")

                        if isinstance(raw_c, str):
                            content = raw_c
                            if raw_c:
                                candidate_texts.append(raw_c)

                        elif isinstance(raw_c, list):
                            parts: List[str] = []
                            for part in raw_c:
                                if not isinstance(part, dict):
                                    continue
                                if isinstance(part.get("text"), str):
                                    parts.append(part["text"])
                                elif isinstance(part.get("content"), str):
                                    parts.append(part["content"])
                                elif isinstance(part.get("value"), str):
                                    parts.append(part["value"])
                            if parts:
                                joined = "".join(parts)
                                content = joined
                                candidate_texts.append(joined)

                        # Collect other possibly-populated fields (Qwen adapters, custom gateways)
                        for k in ("reasoning", "reasoning_content", "final", "output_text", "text"):
                            v = message.get(k)
                            if isinstance(v, str) and v.strip():
                                candidate_texts.append(v)

                        # Tool calls sometimes exist with empty content; we still log them
                        tool_calls = message.get("tool_calls")
                        if isinstance(tool_calls, list) and tool_calls:
                            try:
                                candidate_texts.append(json.dumps(tool_calls, ensure_ascii=False))
                            except Exception:
                                candidate_texts.append(str(tool_calls))

                    # Some providers: choices[0].text
                    if isinstance(choice0.get("text"), str) and choice0["text"].strip():
                        candidate_texts.append(choice0["text"])

                    # Some providers: choices[0].delta.content
                    delta = choice0.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"].strip():
                        candidate_texts.append(delta["content"])

            # -----------------------------------------------------------------
            # Gemini native-ish fallback (if user points to non-OpenAI endpoint by mistake)
            # candidates[0].content.parts[].text
            # -----------------------------------------------------------------
            candidates = data.get("candidates")
            if isinstance(candidates, list) and candidates:
                c0 = candidates[0]
                if isinstance(c0, dict):
                    c0_content = c0.get("content")
                    if isinstance(c0_content, dict):
                        parts = c0_content.get("parts")
                        if isinstance(parts, list):
                            native_parts: List[str] = []
                            for p in parts:
                                if isinstance(p, dict) and isinstance(p.get("text"), str):
                                    native_parts.append(p["text"])
                            if native_parts:
                                native_joined = "".join(native_parts)
                                candidate_texts.append(native_joined)
                                if not content:
                                    content = native_joined

            # Top-level fallbacks
            for k in ("output_text", "text", "content"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    candidate_texts.append(v)
                    if not content:
                        content = v

        # Normalize primary content
        content = str(content or "").strip()

        # MCQ classifier special handling:
        # - DO NOT return reasoning/thoughts.
        # - If content is empty, extract digit 0-3 from any available field and return only that digit.
        if pipeline == "mcq_classifier":
            if content:
                mcq_digit = _extract_first_mcq_digit(content)

            if mcq_digit is None:
                for t in candidate_texts:
                    mcq_digit = _extract_first_mcq_digit(t)
                    if mcq_digit is not None:
                        break

            if mcq_digit is not None:
                content = mcq_digit
            else:
                content = content.strip()

    except Exception as e:
        logging.exception("Error extracting 'content' from chat response: %s", e)
        content = ""

    call_log = {
        "kind": "chat_completion",
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "meta": meta or {},
        "request": {
            "url": url,
            "messages": payload.get("messages", msgs),
            # helpful for debugging provider incompatibilities without dumping everything
            "payload_keys": sorted(list(payload.keys())),
        },
        "response": data,
        "parsed_content": content,
        "status_code": resp.status_code,
        "latency_ms": latency_ms,
    }
    log_llm_call_incremental(call_log)

    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
    return content


# =============================================================================
# /COMPLETIONS echo TOKENIZATION + SCORING
# =============================================================================

_PROMPT_ECHO_CACHE: Dict[Tuple[str, str], Dict[str, List[Any]]] = {}
# key = (model_name, prompt_stripped)


def _fetch_echo_tokens_from_server(
    *,
    model_name: str,
    text: str,
) -> Tuple[List[str], List[Optional[float]], List[int]]:
    """
    /completions echo=True: ritorna tokens/logprobs/offsets SOLO per l'echo del text
    (esclude il token generato obbligatorio via max_tokens=1) filtrando o < len(text).
    """
    url = f"{BASE_URL}/completions"
    payload = {
        "model": model_name,
        "prompt": text,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 1,
        # "echo": True,
        "seed": API_SEED,
    }

    resp = _post_with_retries(url, payload)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    lp = data["choices"][0]["logprobs"]
    tokens = lp["tokens"]
    token_logprobs = lp["token_logprobs"]
    text_offsets = lp["text_offset"]

    L = len(text)
    kept_tokens: List[str] = []
    kept_lps: List[Optional[float]] = []
    kept_offsets: List[int] = []

    for t, p, o in zip(tokens, token_logprobs, text_offsets):
        if o is None:
            continue
        if o < L:
            kept_tokens.append(t)
            kept_lps.append(p)
            kept_offsets.append(o)

    return kept_tokens, kept_lps, kept_offsets


def _get_prompt_tokens_cached(*, model_name: str, prompt: str) -> Tuple[List[str], List[int]]:
    """
    Cache prompt echo-tokenization to avoid repeated /completions echo calls.
    Returns (tokens, offsets) for stable prompt/choice splitting during scoring.
    """
    key = (model_name, prompt)
    cached = _PROMPT_ECHO_CACHE.get(key)
    if cached is not None:
        return cached["tokens"], cached["offsets"]

    toks, _lps, offs = _fetch_echo_tokens_from_server(model_name=model_name, text=prompt)
    _PROMPT_ECHO_CACHE[key] = {"tokens": toks, "offsets": offs}
    return toks, offs


def _longest_common_prefix_len(a: List[str], b: List[str]) -> int:
    """
    Compute the token-level longest common prefix length between two token lists.
    Used as a fallback split point when prompt tokenization diverges from full_text.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def score_choice_with_completions_lmeval_token_split(
    prompt: str,
    choice: str,
    model_name: str,
    target_delimiter: str = templates.TARGET_DELIMITER,
) -> Tuple[float, int, bool, int, int]:
    """
    LM-Eval-like scoring (robusto): split in token-space usando prompt token count,
    con fallback LCP se la prefix tokenization diverge.

    Returns:
      (total_logprob, num_choice_tokens, used_lcp_split, prompt_tok_count, lcp)
    """
    prompt = templates._strip_trailing_spaces_like_lmeval(prompt)
    scored_suffix = target_delimiter + choice
    full_text = prompt + scored_suffix

    prompt_tokens, _ = _get_prompt_tokens_cached(model_name=model_name, prompt=prompt)
    prompt_tok_count = len(prompt_tokens)

    full_tokens, full_lps, _ = _fetch_echo_tokens_from_server(model_name=model_name, text=full_text)

    used_lcp = False
    lcp = prompt_tok_count

    if full_tokens[:prompt_tok_count] != prompt_tokens:
        lcp = _longest_common_prefix_len(prompt_tokens, full_tokens)
        used_lcp = True
        logging.warning(
            "[MCQ-LOGPROB] Prefix token mismatch (model=%s). prompt_tok_count=%d lcp=%d. Falling back to LCP split.",
            model_name,
            prompt_tok_count,
            lcp,
        )
        split = lcp
    else:
        split = prompt_tok_count

    choice_lps = full_lps[split:]

    total_logprob = 0.0
    num_choice_tokens = 0
    for lp in choice_lps:
        if lp is None:
            continue
        total_logprob += lp
        num_choice_tokens += 1

    if num_choice_tokens == 0:
        logging.warning(
            "[MCQ-LOGPROB] zero counted choice tokens (model=%s) split=%d prompt_tok=%d full_tok=%d choice_repr=%r",
            model_name,
            split,
            prompt_tok_count,
            len(full_tokens),
            choice[:120],
        )
        return float("-inf"), 0, used_lcp, prompt_tok_count, lcp

    return total_logprob, num_choice_tokens, used_lcp, prompt_tok_count, lcp


def debug_compare_old_vs_new_choice_tokens(
    *,
    model_name: str,
    prompt: str,
    choice: str,
    target_delimiter: str = templates.TARGET_DELIMITER,
    uuid: str = "",
    choice_idx: int = -1,
    max_token_snippet: int = 5,
) -> None:
    """
    Debug non-invasivo: confronta selezione tokens via char-range vs token-split.
    Logga solo se mismatch.
    """
    prompt_stripped = templates._strip_trailing_spaces_like_lmeval(prompt)
    scored_suffix = target_delimiter + choice
    full_text = prompt_stripped + scored_suffix

    full_tokens, full_lps, full_offsets = _fetch_echo_tokens_from_server(model_name=model_name, text=full_text)
    prompt_tokens, _ = _get_prompt_tokens_cached(model_name=model_name, prompt=prompt_stripped)

    prompt_tok_count = len(prompt_tokens)
    lcp = _longest_common_prefix_len(prompt_tokens, full_tokens)
    split = prompt_tok_count if full_tokens[:prompt_tok_count] == prompt_tokens else lcp

    new_choice_tokens = full_tokens[split:]
    new_choice_lps = full_lps[split:]
    new_tokens_filtered = [t for t, lp in zip(new_choice_tokens, new_choice_lps) if lp is not None]
    num_choice_tokens_new = len(new_tokens_filtered)

    start = len(prompt_stripped)
    end = len(prompt_stripped) + len(scored_suffix)

    old_choice_tokens: List[str] = []
    for t, lp, off in zip(full_tokens, full_lps, full_offsets):
        if lp is None or off is None:
            continue
        if start <= off < end:
            old_choice_tokens.append(t)

    prefix_mismatch = (lcp != prompt_tok_count)
    count_mismatch = (len(old_choice_tokens) != num_choice_tokens_new)
    tokens_mismatch = (old_choice_tokens != new_tokens_filtered)

    if not (prefix_mismatch or count_mismatch or tokens_mismatch):
        return

    def _snip(xs: List[str]) -> Tuple[List[str], List[str]]:
        if len(xs) <= max_token_snippet:
            return xs, []
        return xs[:max_token_snippet], xs[-max_token_snippet:]

    old_head, old_tail = _snip(old_choice_tokens)
    new_head, new_tail = _snip(new_tokens_filtered)

    logging.warning(
        "[MCQ-DEBUG] uuid=%s choice_idx=%d prefix_mismatch=%s count_mismatch=%s tokens_mismatch=%s | "
        "prompt_tok_count=%d lcp=%d split=%d | old_num=%d new_num=%d | "
        "old_head=%s old_tail=%s | new_head=%s new_tail=%s",
        uuid,
        choice_idx,
        prefix_mismatch,
        count_mismatch,
        tokens_mismatch,
        prompt_tok_count,
        lcp,
        split,
        len(old_choice_tokens),
        num_choice_tokens_new,
        old_head,
        old_tail,
        new_head,
        new_tail,
    )


# =============================================================================
# LLM-AS-JUDGE PIPELINE
# =============================================================================

def call_target_model_freeform(
    example: When2CallExample,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> ModelResponse:
    """
    Query the target model for LLM-as-judge with the NVIDIA-style prompt ALWAYS.

    Rationale:
      - Restores the pre-Harmony behavior for the LLM-as-judge target phase.
      - Ensures gpt-oss-* target models use the same NVIDIA system+user prompt
        (tools in <tool>...</tool> blocks) as other models.

    NOTE:
      - This does NOT change MCQ / MCQ-logprob prompt-family behavior.
      - Only the llm_judge target freeform call is affected.
    """
    # ALWAYS NVIDIA for LLM-as-judge target calls
    system_prompt, user_prompt, _choices, _gold_idx, _answer_names = templates.build_prompts_nvidia(example)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = raw_chat_completion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        meta={
            "pipeline": "llm_judge_target",
            "uuid": example.uuid,
            "source": example.source,
            "source_id": example.source_id,
            "template_family": "nvidia",
        },
    )
    return ModelResponse(uuid=example.uuid, raw_text=text)


def call_judge_model(
    example: When2CallExample,
    model_response: ModelResponse,
    judge_model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> JudgeDecision:
    """
    Run the judge model and enforce valid JSON output via a repair retry if needed.

    NEW:
      - Emits repair/fallback flags (stored in JudgeDecision + checkpoint)
      - Writes audit events into checkpoints/llm_judge/audit_fallbacks.jsonl
    """
    judge_prompt = templates.build_judge_prompt(example, model_response)
    base_meta = {
        "pipeline": "llm_judge_judge",
        "uuid": example.uuid,
        "source": example.source,
        "source_id": example.source_id,
    }

    messages = [{"role": "user", "content": judge_prompt}]
    raw = raw_chat_completion(
        model=judge_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        meta=base_meta,
    )

    parse_failed_first = False
    parse_failed_second = False
    used_retry = False
    fallback_to_cannot = False
    exc1: Optional[str] = None
    exc2: Optional[str] = None

    try:
        label, rationale = templates.parse_judge_json(raw)
    except Exception as e_first:
        parse_failed_first = True
        exc1 = str(e_first)
        used_retry = True

        logging.warning(
            "[LLM-JUDGE] First JSON parse failed for uuid=%s: %s. Retrying with rewrite prompt.",
            example.uuid,
            e_first,
        )

        # Audit event (first parse failure)
        append_audit_event(
            "llm_judge",
            {
                "uuid": example.uuid,
                "pipeline": "llm_judge",
                "stage": "judge_parsing",
                "fallback_type": "judge_json_parse_failed_first",
                "severity": "warning",
                "details": {
                    "judge_model": judge_model,
                    "exception": exc1,
                    "raw_preview": (raw or "")[:200],
                },
            },
        )

        messages_retry = list(messages)
        messages_retry.append({"role": "assistant", "content": raw})
        messages_retry.append(
            {
                "role": "user",
                "content": (
                    "Please re-write your response to be shorter and make sure "
                    "it's a valid json in the prescribed format."
                ),
            }
        )

        raw_retry = raw_chat_completion(
            model=judge_model,
            messages=messages_retry,
            temperature=temperature,
            max_tokens=max_tokens,
            meta={**base_meta, "retry": 1},
        )

        try:
            label, rationale = templates.parse_judge_json(raw_retry)
            raw = raw_retry
        except Exception as e_second:
            parse_failed_second = True
            exc2 = str(e_second)
            fallback_to_cannot = True

            logging.error(
                "[LLM-JUDGE] Second JSON parse failed for uuid=%s: %s. Falling back to 'cannot_answer'.",
                example.uuid,
                e_second,
            )

            # Audit event (second parse failure -> fallback)
            append_audit_event(
                "llm_judge",
                {
                    "uuid": example.uuid,
                    "pipeline": "llm_judge",
                    "stage": "judge_parsing",
                    "fallback_type": "judge_json_parse_failed_second_fallback_to_cannot_answer",
                    "severity": "error",
                    "details": {
                        "judge_model": judge_model,
                        "exception_first": exc1,
                        "exception_second": exc2,
                        "raw_first_preview": (raw or "")[:200],
                        "raw_second_preview": (raw_retry or "")[:200],
                    },
                },
            )

            label, rationale = "cannot_answer", None
            # keep raw as raw_retry (more informative for checkpoint)
            raw = raw_retry

    return JudgeDecision(
        uuid=example.uuid,
        predicted_label=label,
        judge_raw=raw,
        judge_rationale=rationale,
        judge_parse_failed_first=parse_failed_first,
        judge_parse_failed_second=parse_failed_second,
        judge_used_retry=used_retry,
        judge_fallback_to_cannot_answer=fallback_to_cannot,
        judge_exception_first=exc1,
        judge_exception_second=exc2,
    )


def run_llm_as_judge_evaluation(
    examples: List[When2CallExample],
    target_model: str,
    judge_model: str,
    model_temperature: float = 0.0,
    judge_temperature: float = 0.0,
) -> Tuple[List[ModelResponse], List[JudgeDecision], EvaluationMetrics]:
    """
    End-to-end LLM-as-judge evaluation with checkpointed resume for target and judge phases.
    Produces per-UUID predictions, persists JSONL checkpoints, and returns aggregate metrics.

    NEW:
      - Judge checkpoint includes repair/fallback flags
      - Metrics emits audit JSONL (missing uuid + coercion) into checkpoints/llm_judge/audit_fallbacks.jsonl
    """
    ckpt_dir = get_child_checkpoint_dir("llm_judge")
    target_ckpt = ckpt_dir / "target_responses.jsonl"
    judge_ckpt = ckpt_dir / "judge_decisions.jsonl"

    # ---- Load existing target responses ----
    target_map = _load_jsonl_as_map(target_ckpt, key_field="uuid")
    uuid_to_resp: Dict[str, ModelResponse] = {}
    for uuid_, rec in target_map.items():
        txt = rec.get("raw_text")
        if isinstance(txt, str):
            uuid_to_resp[uuid_] = ModelResponse(uuid=uuid_, raw_text=txt)

    logging.info(
        "LLM-as-judge RESUME: loaded %d target responses from %s",
        len(uuid_to_resp),
        target_ckpt,
    )

    # ---- Phase 1: target model calls (only missing UUIDs) ----
    for idx, ex in enumerate(examples, start=1):
        if ex.uuid in uuid_to_resp:
            continue
        logging.info("[LLM-as-judge] Target %d/%d uuid=%s", idx, len(examples), ex.uuid)
        resp = call_target_model_freeform(ex, model_name=target_model, temperature=model_temperature)
        uuid_to_resp[ex.uuid] = resp
        _atomic_append_jsonl(
            target_ckpt,
            {
                "uuid": resp.uuid,
                "raw_text": resp.raw_text,
                "target_model": target_model,
                "temperature": model_temperature,
                "api_seed": API_SEED,
            },
        )

    # ---- Load existing judge decisions ----
    judge_map = _load_jsonl_as_map(judge_ckpt, key_field="uuid")
    uuid_to_label: Dict[str, str] = {}
    decisions: List[JudgeDecision] = []

    for uuid_, rec in judge_map.items():
        lab = rec.get("predicted_label")
        raw = rec.get("judge_raw")
        if isinstance(lab, str) and isinstance(raw, str):
            uuid_to_label[uuid_] = lab
            decisions.append(
                JudgeDecision(
                    uuid=uuid_,
                    predicted_label=lab,
                    judge_raw=raw,
                    judge_rationale=rec.get("judge_rationale") if isinstance(rec.get("judge_rationale"), str) else None,
                    judge_parse_failed_first=bool(rec.get("judge_parse_failed_first", False)),
                    judge_parse_failed_second=bool(rec.get("judge_parse_failed_second", False)),
                    judge_used_retry=bool(rec.get("judge_used_retry", False)),
                    judge_fallback_to_cannot_answer=bool(rec.get("judge_fallback_to_cannot_answer", False)),
                    judge_exception_first=rec.get("judge_exception_first") if isinstance(rec.get("judge_exception_first"), str) else None,
                    judge_exception_second=rec.get("judge_exception_second") if isinstance(rec.get("judge_exception_second"), str) else None,
                )
            )

    logging.info(
        "LLM-as-judge RESUME: loaded %d judge decisions from %s",
        len(uuid_to_label),
        judge_ckpt,
    )

    # ---- Phase 2: judge calls (only missing UUIDs) ----
    for idx, ex in enumerate(examples, start=1):
        if ex.uuid in uuid_to_label:
            continue
        logging.info("[LLM-as-judge] Judge %d/%d uuid=%s", idx, len(examples), ex.uuid)

        resp = uuid_to_resp.get(ex.uuid)
        if resp is None:
            # This should not happen, but keep it safe.
            resp = call_target_model_freeform(ex, model_name=target_model, temperature=model_temperature)
            uuid_to_resp[ex.uuid] = resp
            _atomic_append_jsonl(
                target_ckpt,
                {
                    "uuid": resp.uuid,
                    "raw_text": resp.raw_text,
                    "target_model": target_model,
                    "temperature": model_temperature,
                    "api_seed": API_SEED,
                },
            )

        dec = call_judge_model(
            ex,
            resp,
            judge_model=judge_model,
            temperature=judge_temperature,
        )
        decisions.append(dec)
        uuid_to_label[ex.uuid] = dec.predicted_label

        _atomic_append_jsonl(
            judge_ckpt,
            {
                "uuid": dec.uuid,
                "predicted_label": dec.predicted_label,
                "judge_raw": dec.judge_raw,
                "judge_rationale": dec.judge_rationale,
                "judge_model": judge_model,
                "temperature": judge_temperature,

                # NEW flags persisted in checkpoint
                "judge_parse_failed_first": bool(dec.judge_parse_failed_first),
                "judge_parse_failed_second": bool(dec.judge_parse_failed_second),
                "judge_used_retry": bool(dec.judge_used_retry),
                "judge_fallback_to_cannot_answer": bool(dec.judge_fallback_to_cannot_answer),
                "judge_exception_first": dec.judge_exception_first,
                "judge_exception_second": dec.judge_exception_second,
            },
        )

    # ---- Build responses list in input order (stable) ----
    responses: List[ModelResponse] = [uuid_to_resp[ex.uuid] for ex in examples if ex.uuid in uuid_to_resp]

    metrics = compute_metrics(
        examples,
        uuid_to_label,
        audit_exp_name="llm_judge",
        audit_context={"variant": "judge"},
        predicted_labels_raw=uuid_to_label,
    )
    return responses, decisions, metrics


# =============================================================================
# MCQ (STRING) + MCQ (LOGPROB INDEX)
# =============================================================================

def call_model_mcq_classifier(
    example: When2CallExample,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 32,
) -> Tuple[int, List[str], int, str]:
    """
    MCQ string-based CANONICO (uguale per tutti i modelli).
    Non usa Harmony / template-family del target. Solo Qwen ha eccezione di trasporto.
    """
    # CANONICAL: sempre NVIDIA
    system_prompt, user_prompt, choices, target_index, answer_names = templates.build_prompts_nvidia(example)

    user_msg: List[str] = [
        "PROMPT THAT WILL BE GIVEN TO THE ASSISTANT:",
        "------------------------------------------",
        "SYSTEM MESSAGE:",
        system_prompt,
        "",
        "USER MESSAGE:",
        user_prompt,
        "",
        "CANDIDATE BEHAVIORS (index, behavior_type, candidate_response):",
        "",
    ]

    for i, (name, text) in enumerate(zip(answer_names, choices)):
        user_msg.append(f"[{i}] behavior_type={name}")
        user_msg.append("candidate_response:")
        user_msg.append(text)
        user_msg.append("")

    user_msg.append(
        "Based on the behavior type definitions, the AVAILABLE tools in the prompt above, "
        "and the 4 candidate responses, which behavior index (0, 1, 2, or 3) is MOST appropriate?"
    )
    user_msg.append("IMPORTANT: Output MUST start with the digit 0/1/2/3 as the first character. No <think>, no preamble.")
    user_msg.append('Answer with ONLY one character: "0", "1", "2", or "3".')

    meta_base = {
        "pipeline": "mcq_classifier",
        "uuid": example.uuid,
        "source": example.source,
        "source_id": example.source_id,
    }

    def _parse_digit(raw: str) -> Optional[int]:
        raw_str = (raw or "").strip()
        m = re.fullmatch(r"\s*([0-3])\s*", raw_str)
        if m:
            return int(m.group(1))
        m2 = re.search(r"\b([0-3])\b", raw_str)
        if m2:
            return int(m2.group(1))
        return None

    is_qwen = "qwen" in (model_name or "").lower()

    # Attempt 1: standard (system + user)
    raw = raw_chat_completion(
        model=model_name,
        messages=[
            {"role": "system", "content": prompts.MCQ_CLASSIFIER_PROMPT},
            {"role": "user", "content": "\n".join(user_msg)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        meta=meta_base,
    )
    parsed = _parse_digit(raw)
    if parsed is not None:
        return parsed, answer_names, target_index, (raw or "").strip()

    # Attempt 2: more tokens
    retry_max_tokens = max(256, int(max_tokens))
    raw_retry = raw_chat_completion(
        model=model_name,
        messages=[
            {"role": "system", "content": prompts.MCQ_CLASSIFIER_PROMPT},
            {"role": "user", "content": "\n".join(user_msg)},
        ],
        temperature=temperature,
        max_tokens=retry_max_tokens,
        meta={**meta_base, "retry": 1, "retry_max_tokens": retry_max_tokens},
    )
    parsed = _parse_digit(raw_retry)
    if parsed is not None:
        return parsed, answer_names, target_index, (raw_retry or "").strip()

    # Attempt 3: Qwen transport only (contenuto uguale, solo packing diverso)
    if is_qwen:
        packed = prompts.MCQ_CLASSIFIER_PROMPT + "\n\n" + "\n".join(user_msg)
        raw_qwen = raw_chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": packed}],
            temperature=temperature,
            max_tokens=retry_max_tokens,
            meta={**meta_base, "retry": 2, "qwen_single_user_message": True, "retry_max_tokens": retry_max_tokens},
        )
        parsed = _parse_digit(raw_qwen)
        if parsed is not None:
            return parsed, answer_names, target_index, (raw_qwen or "").strip()
        return -1, answer_names, target_index, (raw_qwen or "").strip()

    return -1, answer_names, target_index, (raw_retry or "").strip()


def run_mcq_evaluation_logprob_index(
    examples: List[When2CallExample],
    target_model: str,
    temperature: float = 0.0,
    forced_delimiter: Optional[str] = None,
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    EvaluationMetrics,
    Dict[str, int],
]:
    """
    Score MCQ choices via /completions echo logprobs and select the best index per UUID.
    Uses delimiter rules + checkpoints; falls back to string-based MCQ if all scores are -inf.

    NEW:
      - On string_fallback: persist raw output, pred_idx, trigger fields in mcq_logprob_predictions.jsonl
        + emit audit event.
      - On LCP split usage: emit audit event (token_prefix_mismatch_lcp_split) with choice_idx, lcp, prompt_tok_count.
      - Metrics emits audit JSONL (missing uuid + coercion) into checkpoints/mcq_logprob/audit_fallbacks.jsonl
        (with variant tags for raw/norm_bytes/norm_tokens/norm_chars).
    """
    ckpt_dir = get_child_checkpoint_dir("mcq_logprob")
    pred_ckpt = ckpt_dir / "mcq_logprob_predictions.jsonl"
    debug_ckpt = ckpt_dir / "debug_per_choice.jsonl"

    existing = _load_jsonl_as_map(pred_ckpt, key_field="uuid")

    uuid_to_pred_norm_bytes: Dict[str, str] = {}
    uuid_to_pred_raw: Dict[str, str] = {}
    uuid_to_pred_norm_tokens: Dict[str, str] = {}
    uuid_to_pred_norm_chars: Dict[str, str] = {}

    # Restore from checkpoint
    for uuid_, rec in existing.items():
        if isinstance(rec.get("predicted_label_norm_bytes"), str):
            uuid_to_pred_norm_bytes[uuid_] = rec["predicted_label_norm_bytes"]
        if isinstance(rec.get("predicted_label_raw"), str):
            uuid_to_pred_raw[uuid_] = rec["predicted_label_raw"]
        if isinstance(rec.get("predicted_label_norm_tokens"), str):
            uuid_to_pred_norm_tokens[uuid_] = rec["predicted_label_norm_tokens"]
        if isinstance(rec.get("predicted_label_norm_chars"), str):
            uuid_to_pred_norm_chars[uuid_] = rec["predicted_label_norm_chars"]

    logging.info("MCQ-LOGPROB RESUME: loaded %d predictions from %s", len(existing), pred_ckpt)

    delim = templates.resolve_target_delimiter(target_model, forced=forced_delimiter)

    mode_counts = {"llama_scoring": 0, "string_fallback": 0}

    for idx, ex in enumerate(examples, start=1):
        if ex.uuid in existing:
            continue

        logging.info("[MCQ-LOGPROB] Example %d/%d uuid=%s", idx, len(examples), ex.uuid)

        prompt, choices, target_idx, answer_names = templates.build_when2call_mcq_item_for_model(ex, target_model)

        scores_raw: List[float] = []
        scores_norm_bytes: List[float] = []
        scores_norm_tokens: List[float] = []
        scores_norm_chars: List[float] = []

        for i, ch in enumerate(choices):
            used_lcp = False
            prompt_tok_count = 0
            lcp = 0

            try:
                lp, num_tokens, used_lcp, prompt_tok_count, lcp = score_choice_with_completions_lmeval_token_split(
                    prompt, ch, model_name=target_model, target_delimiter=delim
                )

                # NEW: Persist LCP split usage for audit (drift quantification)
                if used_lcp:
                    append_audit_event(
                        "mcq_logprob",
                        {
                            "uuid": ex.uuid,
                            "pipeline": "mcq_logprob",
                            "stage": "scoring",
                            "fallback_type": "token_prefix_mismatch_lcp_split",
                            "severity": "info",
                            "details": {
                                "choice_idx": int(i),
                                "prompt_tok_count": int(prompt_tok_count),
                                "lcp": int(lcp),
                                "forced_delimiter": repr(forced_delimiter),
                                "effective_delimiter": repr(delim),
                            },
                        },
                    )

                if DEBUG_MCQ_TOKEN_MISMATCH and idx <= DEBUG_MCQ_TOKEN_MISMATCH_MAX:
                    debug_compare_old_vs_new_choice_tokens(
                        model_name=target_model,
                        prompt=prompt,
                        choice=ch,
                        target_delimiter=delim,
                        uuid=ex.uuid,
                        choice_idx=i,
                        max_token_snippet=5,
                    )
            except Exception as e:
                logging.error("[MCQ-LOGPROB] Error scoring choice %d for uuid=%s: %s", i, ex.uuid, e)
                lp, num_tokens = float("-inf"), 0
                used_lcp = False
                prompt_tok_count = 0
                lcp = 0

            scores_raw.append(lp)

            scored_suffix = delim + ch
            length_bytes = len(scored_suffix.encode("utf-8")) or 1
            length_chars = len(scored_suffix) or 1

            scores_norm_bytes.append(float("-inf") if lp == float("-inf") else lp / float(length_bytes))

            if lp == float("-inf") or num_tokens <= 0:
                token_norm_score = float("-inf")
                scores_norm_tokens.append(float("-inf"))
            else:
                token_norm_score = lp / float(num_tokens)
                scores_norm_tokens.append(token_norm_score)

            scores_norm_chars.append(float("-inf") if lp == float("-inf") else lp / float(length_chars))

            # Persist per-choice debug (optional but useful for resume analysis)
            preview = ch if len(ch) <= 240 else ch[:240] + "…"
            _atomic_append_jsonl(
                debug_ckpt,
                {
                    "uuid": ex.uuid,
                    "gold_label": ex.correct_answer,
                    "answer_name": answer_names[i] if 0 <= i < len(answer_names) else None,
                    "choice_text_preview": preview,
                    "len_chars": int(length_chars),
                    "len_bytes": int(length_bytes),
                    "num_tokens": int(num_tokens),
                    "raw_score": float(lp),
                    "token_norm_score": float(token_norm_score),
                    "is_tool_call_empty": bool((ch.strip() in {"[]", ""})),

                    # NEW: make drift observable even without reading audit_fallbacks
                    "used_lcp_split": bool(used_lcp),
                    "prompt_tok_count": int(prompt_tok_count),
                    "lcp": int(lcp),
                },
            )

        if all(s == float("-inf") for s in scores_raw):
            logging.warning("[MCQ-LOGPROB] All logprobs -inf for uuid=%s; falling back to string-based MCQ.", ex.uuid)

            pred_idx_fb, answer_names_fb, _target_idx_fb, raw_fb = call_model_mcq_classifier(
                ex, model_name=target_model, temperature=temperature
            )
            mode_counts["string_fallback"] += 1

            label_fb = answer_names_fb[pred_idx_fb] if 0 <= pred_idx_fb < len(answer_names_fb) else "invalid"

            # NEW: audit event for string fallback
            append_audit_event(
                "mcq_logprob",
                {
                    "uuid": ex.uuid,
                    "pipeline": "mcq_logprob",
                    "stage": "scoring",
                    "fallback_type": "all_logprobs_-inf_string_fallback",
                    "severity": "warning",
                    "details": {
                        "fallback_trigger": "all_logprobs_-inf",
                        "pred_idx_fb": int(pred_idx_fb),
                        "label_fb": label_fb,
                        "raw_fb_preview": (raw_fb or "")[:200],
                        "forced_delimiter": repr(forced_delimiter),
                        "effective_delimiter": repr(delim),
                    },
                },
            )

            uuid_to_pred_raw[ex.uuid] = label_fb
            uuid_to_pred_norm_bytes[ex.uuid] = label_fb
            uuid_to_pred_norm_tokens[ex.uuid] = label_fb
            uuid_to_pred_norm_chars[ex.uuid] = label_fb

            out_rec = {
                "uuid": ex.uuid,
                "gold_label": ex.correct_answer,
                "predicted_label_norm_bytes": label_fb,
                "predicted_label_raw": label_fb,
                "predicted_label_norm_tokens": label_fb,
                "predicted_label_norm_chars": label_fb,
                "mode": "string_fallback",
                "target_model": target_model,
                "forced_delimiter": repr(forced_delimiter),
                "effective_delimiter": repr(delim),

                # NEW: string fallback details persisted
                "fallback_trigger": "all_logprobs_-inf",
                "string_fallback_pred_idx": int(pred_idx_fb),
                "string_fallback_raw_output": (raw_fb or ""),
                "string_fallback_answer_names": answer_names_fb,
            }
            _atomic_append_jsonl(pred_ckpt, out_rec)
            existing[ex.uuid] = out_rec
            continue

        mode_counts["llama_scoring"] += 1

        best_raw_idx = max(range(len(scores_raw)), key=lambda i: scores_raw[i])
        best_norm_bytes_idx = max(range(len(scores_norm_bytes)), key=lambda i: scores_norm_bytes[i])
        best_norm_tokens_idx = max(range(len(scores_norm_tokens)), key=lambda i: scores_norm_tokens[i])
        best_norm_chars_idx = max(range(len(scores_norm_chars)), key=lambda i: scores_norm_chars[i])

        lab_raw = answer_names[best_raw_idx] if 0 <= best_raw_idx < len(answer_names) else "invalid"
        lab_b = answer_names[best_norm_bytes_idx] if 0 <= best_norm_bytes_idx < len(answer_names) else "invalid"
        lab_t = answer_names[best_norm_tokens_idx] if 0 <= best_norm_tokens_idx < len(answer_names) else "invalid"
        lab_c = answer_names[best_norm_chars_idx] if 0 <= best_norm_chars_idx < len(answer_names) else "invalid"

        uuid_to_pred_raw[ex.uuid] = lab_raw
        uuid_to_pred_norm_bytes[ex.uuid] = lab_b
        uuid_to_pred_norm_tokens[ex.uuid] = lab_t
        uuid_to_pred_norm_chars[ex.uuid] = lab_c

        out_rec = {
            "uuid": ex.uuid,
            "gold_label": ex.correct_answer,
            "predicted_label_norm_bytes": lab_b,
            "predicted_label_raw": lab_raw,
            "predicted_label_norm_tokens": lab_t,
            "predicted_label_norm_chars": lab_c,
            "mode": "llama_scoring",
            "target_model": target_model,
            "forced_delimiter": repr(forced_delimiter),
            "effective_delimiter": repr(delim),
            "scores_raw": scores_raw,
            "scores_norm_bytes": scores_norm_bytes,
            "scores_norm_tokens": scores_norm_tokens,
            "scores_norm_chars": scores_norm_chars,
            "gold_index": target_idx,
        }
        _atomic_append_jsonl(pred_ckpt, out_rec)
        existing[ex.uuid] = out_rec

    # Metrics (+ audit per variant)
    metrics_norm_bytes = compute_metrics(
        examples,
        uuid_to_pred_norm_bytes,
        audit_exp_name="mcq_logprob",
        audit_context={"variant": "norm_bytes"},
        predicted_labels_raw=uuid_to_pred_norm_bytes,
    )
    metrics_raw = compute_metrics(
        examples,
        uuid_to_pred_raw,
        audit_exp_name="mcq_logprob",
        audit_context={"variant": "raw"},
        predicted_labels_raw=uuid_to_pred_raw,
    )
    metrics_norm_tokens = compute_metrics(
        examples,
        uuid_to_pred_norm_tokens,
        audit_exp_name="mcq_logprob",
        audit_context={"variant": "norm_tokens"},
        predicted_labels_raw=uuid_to_pred_norm_tokens,
    )
    metrics_norm_chars = compute_metrics(
        examples,
        uuid_to_pred_norm_chars,
        audit_exp_name="mcq_logprob",
        audit_context={"variant": "norm_chars"},
        predicted_labels_raw=uuid_to_pred_norm_chars,
    )

    return (
        uuid_to_pred_norm_bytes,
        uuid_to_pred_raw,
        uuid_to_pred_norm_tokens,
        uuid_to_pred_norm_chars,
        metrics_norm_bytes,
        metrics_raw,
        metrics_norm_tokens,
        metrics_norm_chars,
        mode_counts,
    )


def run_mcq_evaluation(
    examples: List[When2CallExample],
    target_model: str,
    temperature: float = 0.0,
) -> Tuple[Dict[str, str], EvaluationMetrics, List[Dict[str, Any]]]:
    """
    MCQ string-based con checkpoint persistente (JSONL) per resume.

    NEW:
      - Metrics emits audit JSONL (missing uuid + coercion) into checkpoints/mcq/audit_fallbacks.jsonl
    """
    ckpt_dir = get_child_checkpoint_dir("mcq")
    pred_ckpt = ckpt_dir / "mcq_predictions.jsonl"

    existing = _load_jsonl_as_map(pred_ckpt, key_field="uuid")
    uuid_to_pred_label: Dict[str, str] = {}
    example_logs: List[Dict[str, Any]] = []

    # Restore
    for uuid_, rec in existing.items():
        lab = rec.get("predicted_label")
        if isinstance(lab, str):
            uuid_to_pred_label[uuid_] = lab
            example_logs.append(rec)

    logging.info("MCQ-classifier RESUME: loaded %d predictions from %s", len(existing), pred_ckpt)
    logging.info("MCQ-classifier: target=%s, examples=%d", target_model, len(examples))

    for idx, ex in enumerate(examples, start=1):
        if ex.uuid in existing:
            continue

        logging.info("[MCQ] Example %d/%d uuid=%s", idx, len(examples), ex.uuid)
        pred_idx, answer_names, target_idx, raw_str = call_model_mcq_classifier(
            ex, model_name=target_model, temperature=temperature
        )

        label = answer_names[pred_idx] if 0 <= pred_idx < len(answer_names) else "invalid"

        logging.info(
            "[MCQ-STRING-PRED] uuid=%s gold=%s(gold_idx=%d) pred=%s(pred_idx=%d) raw=%r",
            ex.uuid,
            ex.correct_answer,
            target_idx,
            label,
            pred_idx,
            (raw_str or "")[:120],
        )

        rec = {
            "uuid": ex.uuid,
            "gold_label": ex.correct_answer,
            "gold_index": target_idx,
            "answer_names": answer_names,
            "predicted_index": pred_idx,
            "predicted_label": label,
            "raw_mcq_output": raw_str,
            "target_model": target_model,
            "temperature": temperature,
            "api_seed": API_SEED,
        }

        _atomic_append_jsonl(pred_ckpt, rec)
        existing[ex.uuid] = rec
        uuid_to_pred_label[ex.uuid] = label
        example_logs.append(rec)

    metrics = compute_metrics(
        examples,
        uuid_to_pred_label,
        audit_exp_name="mcq",
        audit_context={"variant": "string"},
        predicted_labels_raw=uuid_to_pred_label,
    )
    return uuid_to_pred_label, metrics, example_logs


def call_model_mcq_logprob_index(
    example: When2CallExample,
    model_name: str,
    temperature: float = 0.0,
    forced_delimiter: Optional[str] = None,
) -> Tuple[int, List[str], int, str]:
    """
    Compute logprob scores for the MCQ choices of a single example and pick the best index.
    Returns a mode flag ("llama_scoring" vs "string_fallback") for audit/debug.
    """
    prompt, choices, target_index, answer_names = templates.build_when2call_mcq_item_for_model(example, model_name)
    delim = templates.resolve_target_delimiter(model_name, forced=forced_delimiter)

    scores: List[float] = []
    for i, ch in enumerate(choices):
        try:
            lp, _ntok, _used_lcp, _ptc, _lcp = score_choice_with_completions_lmeval_token_split(
                prompt, ch, model_name=model_name, target_delimiter=delim
            )
        except Exception as e:
            logging.error("[MCQ-LOGPROB] Error scoring choice %d for uuid=%s: %s", i, example.uuid, e)
            lp = float("-inf")
        scores.append(lp)

    if all(s == float("-inf") for s in scores):
        logging.warning("All logprobs -inf for uuid=%s; falling back to string-based MCQ.", example.uuid)
        pred_idx, answer_names2, target_index2, _ = call_model_mcq_classifier(
            example, model_name=model_name, temperature=temperature
        )
        return pred_idx, answer_names2, target_index2, "string_fallback"

    best_index = max(range(len(scores)), key=lambda i: scores[i])
    logging.info(
        "[MCQ-LOGPROB] uuid=%s scores=%s best_index=%d (gold=%d)",
        example.uuid,
        [round(s, 3) for s in scores],
        best_index,
        target_index,
    )
    return best_index, answer_names, target_index, "llama_scoring"


# =============================================================================
# METRICS + DEBUG PRINTS
# =============================================================================

def compute_metrics(
    examples: List[When2CallExample],
    predicted_labels: Dict[str, str],
    *,
    audit_exp_name: Optional[str] = None,
    audit_context: Optional[Dict[str, Any]] = None,
    predicted_labels_raw: Optional[Dict[str, str]] = None,
) -> EvaluationMetrics:
    """
    Compute accuracy, macro-F1, per-class F1/support, and a confusion matrix over 4 labels.

    NEW (audit):
      - If audit_exp_name is provided, writes JSONL audit events for:
        (a) missing uuid in predicted_labels
        (b) invalid label coercion -> 'cannot_answer'
      - audit_context is merged into event 'details' (e.g., {"variant":"raw"}).
      - predicted_labels_raw, if provided, is used to log the original (pre-coercion) label.
    """
    conf = {t: {p: 0 for p in templates.BEHAVIOR_LABELS} for t in templates.BEHAVIOR_LABELS}

    n_total = 0
    n_correct = 0

    n_tool_hall_denom = 0
    n_tool_hall_num = 0

    n_ans_hall_num = 0
    n_param_hall_num = 0
    n_param_hall_denom = 0

    support_by_true: Dict[str, int] = {lab: 0 for lab in templates.BEHAVIOR_LABELS}

    n_missing_uuid = 0
    n_invalid_label = 0
    n_converted_to_cannot_answer = 0

    ctx_details = dict(audit_context or {})

    for ex in examples:
        true_label = ex.correct_answer

        is_missing = ex.uuid not in predicted_labels
        if is_missing:
            n_missing_uuid += 1
            if audit_exp_name:
                append_audit_event(
                    audit_exp_name,
                    {
                        "uuid": ex.uuid,
                        "pipeline": audit_exp_name,
                        "stage": "metrics",
                        "fallback_type": "missing_prediction_uuid",
                        "severity": "error",
                        "details": {
                            **ctx_details,
                            "gold_label": true_label,
                        },
                    },
                )

        pred_label = predicted_labels.get(ex.uuid, "invalid")

        if pred_label not in templates.BEHAVIOR_LABELS:
            n_invalid_label += 1
            n_converted_to_cannot_answer += 1

            raw_label = None
            if predicted_labels_raw is not None:
                raw_label = predicted_labels_raw.get(ex.uuid)

            if audit_exp_name:
                append_audit_event(
                    audit_exp_name,
                    {
                        "uuid": ex.uuid,
                        "pipeline": audit_exp_name,
                        "stage": "metrics",
                        "fallback_type": "invalid_label_coercion_to_cannot_answer",
                        "severity": "warning",
                        "details": {
                            **ctx_details,
                            "gold_label": true_label,
                            "pred_label_raw": raw_label if raw_label is not None else pred_label,
                            "pred_label_coerced": "cannot_answer",
                            "reason": "missing_uuid" if is_missing else "invalid_label",
                        },
                    },
                )

            pred_label = "cannot_answer"

        conf[true_label][pred_label] += 1
        support_by_true[true_label] += 1
        n_total += 1
        if true_label == pred_label:
            n_correct += 1

        if true_label == "cannot_answer" and not ex.tools_raw:
            n_tool_hall_denom += 1
            if pred_label == "tool_call":
                n_tool_hall_num += 1

        if pred_label == "direct" and true_label != "direct":
            n_ans_hall_num += 1

        if true_label == "request_for_info":
            n_param_hall_denom += 1
            if pred_label == "tool_call":
                n_param_hall_num += 1

    accuracy = n_correct / n_total if n_total else 0.0

    per_class_f1: Dict[str, float] = {}
    for label in templates.BEHAVIOR_LABELS:
        tp = conf[label][label]
        fp = sum(conf[t][label] for t in templates.BEHAVIOR_LABELS if t != label)
        fn = sum(conf[label][p] for p in templates.BEHAVIOR_LABELS if p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_f1[label] = f1

    macro_f1_all = sum(per_class_f1.values()) / len(templates.BEHAVIOR_LABELS)
    labels_no_direct = [lab for lab in templates.BEHAVIOR_LABELS if lab != "direct"]
    macro_f1_no_direct = (
        sum(per_class_f1[lab] for lab in labels_no_direct) / len(labels_no_direct) if labels_no_direct else 0.0
    )

    tool_hall = (n_tool_hall_num / n_tool_hall_denom) if n_tool_hall_denom else None
    ans_hall = (n_ans_hall_num / n_total) if n_total else None
    param_hall = (n_param_hall_num / n_param_hall_denom) if n_param_hall_denom else None

    return EvaluationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1_all,
        macro_f1_no_direct=macro_f1_no_direct,
        per_class_f1=per_class_f1,
        per_class_support=support_by_true,
        confusion_matrix=conf,
        tool_hallucination_rate=tool_hall,
        answer_hallucination_rate=ans_hall,
        parameter_hallucination_rate=param_hall,
        n_total=n_total,
        n_missing_uuid=n_missing_uuid,
        n_invalid_label=n_invalid_label,
        n_converted_to_cannot_answer=n_converted_to_cannot_answer,
    )


def debug_tool_hallucination(examples: List[When2CallExample], uuid_to_pred: Dict[str, str]) -> None:
    """
    Print a compact distribution of (gold, pred) for examples where no tools are available.
    Useful to diagnose tool-call hallucinations when the correct behavior is not tool_call.
    """
    no_tools = [ex for ex in examples if not ex.tools_raw]
    print(f"\n[DEBUG] # examples without tools: {len(no_tools)}")
    counts = Counter()
    for ex in no_tools:
        pred = uuid_to_pred.get(ex.uuid, "MISSING")
        counts[(ex.correct_answer, pred)] += 1
    print("[DEBUG] Distribution (gold, pred) for cases without tools:")
    for (gold, pred), c in counts.items():
        print(f"  gold={gold:15s} pred={pred:15s} count={c}")


def print_invalid_summary(title: str, metrics: EvaluationMetrics) -> None:
    """
    Pretty-print counts of missing UUIDs and invalid predicted labels for a given run.
    Helps verify checkpoint integrity and label normalization behavior.
    """
    line = "=" * 72
    print("\n" + line)
    print(f"INVALID PREDICTIONS SUMMARY — {title}")
    print(line)
    print(f"Total items evaluated              : {metrics.n_total}")
    print(f"Missing uuid in predictions dict   : {metrics.n_missing_uuid}")
    print(f"Invalid predicted labels           : {metrics.n_invalid_label}")
    print(f"Converted to 'cannot_answer'       : {metrics.n_converted_to_cannot_answer}")
    print(line + "\n")


# =============================================================================
# STABILITY (UNIFIED)
# =============================================================================

import math


def _shannon_entropy_from_counts(counts: Counter, *, n_total: int) -> float:
    """
    Shannon entropy (base-2) of a categorical distribution defined by counts.
    Returns 0.0 for degenerate or empty distributions.
    """
    if n_total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / float(n_total)
        # p>0 guaranteed here
        h -= p * math.log2(p)
    return float(h)


def _normalized_entropy(h: float, *, n_labels: int) -> float:
    """
    Normalizes entropy by the maximum possible entropy log2(n_labels).
    If n_labels <= 1, returns 0.0.
    """
    if n_labels <= 1:
        return 0.0
    hmax = math.log2(float(n_labels))
    return float(h / hmax) if hmax > 0 else 0.0


def _flip_rate(run_labels: List[str]) -> float:
    """
    Flip rate: fraction of adjacent transitions where the label changes.
    Defined as (# of i where run_labels[i] != run_labels[i-1]) / (k-1).
    Returns 0.0 if k <= 1.
    """
    k = len(run_labels)
    if k <= 1:
        return 0.0
    flips = 0
    for i in range(1, k):
        if run_labels[i] != run_labels[i - 1]:
            flips += 1
    return float(flips) / float(k - 1)


def run_stability_evaluation(
    eval_type: str,
    examples: List[When2CallExample],
    target_model: str,
    n_runs: int = 5,
    temperature: float = 0.5,
    judge_model: Optional[str] = None,
    forced_delimiter: Optional[str] = None,
    ckpt_name: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Run k repeated predictions per item to estimate stability (consistency/entropy/flip-rate).
    Uses JSONL checkpoints keyed by UUID for resume; supports mcq, mcq_logprob, and llm_judge.
    """
    if not examples:
        return {}, []

    eval_type = eval_type.lower()
    if eval_type not in {"mcq", "mcq_logprob", "llm_judge"}:
        raise ValueError(f"Unknown eval_type '{eval_type}'. Expected 'mcq', 'mcq_logprob' or 'llm_judge'.")
    if eval_type == "llm_judge" and not judge_model:
        raise ValueError("judge_model must be provided when eval_type='llm_judge'.")

    # IMPORTANT: checkpoint namespace must be UNIQUE per sweep (k + temperature),
    # otherwise results from different temperatures get mixed.
    if not ckpt_name:
        ckpt_name = f"stability_{eval_type}_k={n_runs}_T={temperature}"

    ckpt_dir = get_child_checkpoint_dir(ckpt_name)
    ckpt_file = ckpt_dir / f"{ckpt_name}.jsonl"
    existing = _load_jsonl_as_map(ckpt_file, key_field="uuid")

    stability_example_logs: List[Dict[str, Any]] = list(existing.values())

    total_items = len(examples)

    # ---- Aggregates (computed from checkpoint + newly processed items) ----
    processed_items = 0

    stable_items = 0
    stable_and_correct_items = 0
    stable_but_wrong_items = 0

    mode_correct_items = 0

    sum_consistency = 0.0
    sum_entropy = 0.0
    sum_norm_entropy = 0.0
    sum_flip_rate = 0.0
    sum_mean_accuracy_across_runs = 0.0

    per_gold_stats: Dict[str, Dict[str, float]] = {
        lab: {
            "count": 0.0,
            "stable": 0.0,
            "sum_consistency": 0.0,
            "sum_entropy": 0.0,
            "sum_norm_entropy": 0.0,
            "sum_flip_rate": 0.0,
            "sum_mean_accuracy_across_runs": 0.0,
            "mode_correct": 0.0,
        }
        for lab in templates.BEHAVIOR_LABELS
    }

    def _ensure_derived_fields(log_entry: Dict[str, Any]) -> Dict[str, Any]:
        gold_label = log_entry.get("gold_label")
        run_labels = log_entry.get("run_labels")
        k = int(log_entry.get("n_runs") or 0)

        if not isinstance(gold_label, str) or gold_label not in templates.BEHAVIOR_LABELS:
            return log_entry
        if not isinstance(run_labels, list) or not run_labels:
            return log_entry
        if k <= 0:
            k = len(run_labels)

        if "mode_label" not in log_entry or "mode_count" not in log_entry:
            cnt = Counter(run_labels)
            mode_label, mode_count = cnt.most_common(1)[0]
            log_entry["mode_label"] = mode_label
            log_entry["mode_count"] = int(mode_count)
        else:
            cnt = Counter(run_labels)

        mode_label = log_entry.get("mode_label")
        mode_count = int(log_entry.get("mode_count") or 0)

        if "consistency" not in log_entry:
            log_entry["consistency"] = float(mode_count) / float(k) if k else 0.0
        if "is_stable" not in log_entry:
            log_entry["is_stable"] = bool(mode_count == k)
        if "is_mode_correct" not in log_entry:
            log_entry["is_mode_correct"] = bool(mode_label == gold_label)
        if "is_stable_and_correct" not in log_entry:
            log_entry["is_stable_and_correct"] = bool(log_entry["is_stable"] and log_entry["is_mode_correct"])
        if "is_stable_but_wrong" not in log_entry:
            log_entry["is_stable_but_wrong"] = bool(log_entry["is_stable"] and (not log_entry["is_mode_correct"]))

        if "mean_accuracy_across_runs" not in log_entry:
            corrects = sum(1 for lab in run_labels if lab == gold_label)
            log_entry["mean_accuracy_across_runs"] = float(corrects) / float(len(run_labels)) if run_labels else 0.0

        if "entropy" not in log_entry:
            h = _shannon_entropy_from_counts(cnt, n_total=len(run_labels))
            log_entry["entropy"] = float(h)
        if "normalized_entropy" not in log_entry:
            log_entry["normalized_entropy"] = float(
                _normalized_entropy(float(log_entry["entropy"]), n_labels=len(templates.BEHAVIOR_LABELS))
            )

        if "flip_rate" not in log_entry:
            log_entry["flip_rate"] = float(_flip_rate(run_labels))

        return log_entry

    def _accumulate(log_entry_in: Dict[str, Any]) -> None:
        nonlocal processed_items, stable_items, stable_and_correct_items, stable_but_wrong_items
        nonlocal mode_correct_items, sum_consistency, sum_entropy, sum_norm_entropy
        nonlocal sum_flip_rate, sum_mean_accuracy_across_runs

        log_entry = _ensure_derived_fields(log_entry_in)

        gold_label = log_entry["gold_label"]
        consistency = float(log_entry.get("consistency", 0.0))
        entropy = float(log_entry.get("entropy", 0.0))
        norm_entropy = float(log_entry.get("normalized_entropy", 0.0))
        flip_rate = float(log_entry.get("flip_rate", 0.0))
        mean_acc_runs = float(log_entry.get("mean_accuracy_across_runs", 0.0))

        is_stable = bool(log_entry.get("is_stable", False))
        is_stable_and_correct = bool(log_entry.get("is_stable_and_correct", False))
        is_stable_but_wrong = bool(log_entry.get("is_stable_but_wrong", False))
        is_mode_correct = bool(log_entry.get("is_mode_correct", False))

        processed_items += 1

        per_gold_stats[gold_label]["count"] += 1.0
        per_gold_stats[gold_label]["stable"] += float(is_stable)
        per_gold_stats[gold_label]["sum_consistency"] += consistency
        per_gold_stats[gold_label]["sum_entropy"] += entropy
        per_gold_stats[gold_label]["sum_norm_entropy"] += norm_entropy
        per_gold_stats[gold_label]["sum_flip_rate"] += flip_rate
        per_gold_stats[gold_label]["sum_mean_accuracy_across_runs"] += mean_acc_runs
        per_gold_stats[gold_label]["mode_correct"] += float(is_mode_correct)

        stable_items += int(is_stable)
        stable_and_correct_items += int(is_stable_and_correct)
        stable_but_wrong_items += int(is_stable_but_wrong)
        mode_correct_items += int(is_mode_correct)

        sum_consistency += consistency
        sum_entropy += entropy
        sum_norm_entropy += norm_entropy
        sum_flip_rate += flip_rate
        sum_mean_accuracy_across_runs += mean_acc_runs

    # Recompute aggregates from existing logs
    for rec in existing.values():
        try:
            _accumulate(rec)
        except Exception:
            logging.exception("[STABILITY %s] Failed to accumulate existing record; skipping one record.", eval_type)

    # ---- Process missing UUIDs only ----
    for idx, ex in enumerate(examples, start=1):
        if ex.uuid in existing:
            continue

        logging.info("[STABILITY %s] Example %d/%d uuid=%s", eval_type, idx, len(examples), ex.uuid)
        gold_label = ex.correct_answer

        run_labels: List[str] = []
        run_indices: List[int] = []
        run_aux1: List[Any] = []
        run_aux2: List[Any] = []

        for _ in range(n_runs):
            if eval_type == "mcq":
                pred_idx, answer_names, _target_idx, raw_str = call_model_mcq_classifier(
                    ex, model_name=target_model, temperature=temperature
                )
                label = answer_names[pred_idx] if 0 <= pred_idx < len(answer_names) else "invalid"
                run_labels.append(label)
                run_indices.append(pred_idx)
                run_aux1.append(raw_str)

            elif eval_type == "mcq_logprob":
                pred_idx, answer_names, _target_idx, mode = call_model_mcq_logprob_index(
                    ex,
                    model_name=target_model,
                    temperature=temperature,
                    forced_delimiter=forced_delimiter,
                )
                label = answer_names[pred_idx] if 0 <= pred_idx < len(answer_names) else "invalid"
                run_labels.append(label)
                run_indices.append(pred_idx)
                run_aux1.append(mode)

            else:
                target_resp = call_target_model_freeform(ex, model_name=target_model, temperature=temperature)
                dec = call_judge_model(ex, target_resp, judge_model=judge_model, temperature=0.0)  # type: ignore[arg-type]
                run_labels.append(dec.predicted_label)
                run_aux1.append(target_resp.raw_text)
                run_aux2.append(dec.judge_raw)

        label_counts = Counter(run_labels)
        mode_label, mode_count = label_counts.most_common(1)[0]

        consistency = float(mode_count) / float(n_runs) if n_runs else 0.0
        is_stable = (mode_count == n_runs)
        is_mode_correct = (mode_label == gold_label)

        mean_accuracy_across_runs = (
            float(sum(1 for lab in run_labels if lab == gold_label)) / float(len(run_labels))
            if run_labels else 0.0
        )

        entropy = _shannon_entropy_from_counts(label_counts, n_total=len(run_labels))
        normalized_entropy = _normalized_entropy(entropy, n_labels=len(templates.BEHAVIOR_LABELS))
        flip_rate = _flip_rate(run_labels)

        log_entry: Dict[str, Any] = {
            "uuid": ex.uuid,
            "gold_label": gold_label,
            "n_runs": n_runs,
            "run_labels": run_labels,

            "mode_label": mode_label,
            "mode_count": int(mode_count),
            "consistency": float(consistency),

            "is_stable": bool(is_stable),
            "is_mode_correct": bool(is_mode_correct),
            "is_stable_and_correct": bool(is_stable and is_mode_correct),
            "is_stable_but_wrong": bool(is_stable and (not is_mode_correct)),

            "mean_accuracy_across_runs": float(mean_accuracy_across_runs),
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "flip_rate": float(flip_rate),

            "eval_type": eval_type,
            "target_model": target_model,
            "temperature": temperature,
            "judge_model": judge_model,
            "forced_delimiter": repr(forced_delimiter),

            # NEW: make the checkpoint scope explicit for auditability
            "ckpt_name": ckpt_name,
        }

        if eval_type in {"mcq", "mcq_logprob"}:
            log_entry["run_indices"] = run_indices
            log_entry["run_aux"] = run_aux1
        else:
            log_entry["run_target_texts"] = run_aux1
            log_entry["run_judge_raw_outputs"] = run_aux2

        _atomic_append_jsonl(ckpt_file, log_entry)
        existing[ex.uuid] = log_entry
        stability_example_logs.append(log_entry)

        _accumulate(log_entry)

    denom = float(total_items) if total_items else 1.0
    coverage = float(processed_items) / float(total_items) if total_items else 0.0

    stability_at_k = float(stable_items) / denom
    mean_consistency_at_k = float(sum_consistency) / denom
    stable_and_correct_rate = float(stable_and_correct_items) / denom
    stable_but_wrong_rate = float(stable_but_wrong_items) / denom

    mode_correct_rate = float(mode_correct_items) / denom

    mean_entropy = float(sum_entropy) / denom
    mean_normalized_entropy = float(sum_norm_entropy) / denom
    mean_flip_rate = float(sum_flip_rate) / denom
    mean_accuracy_across_runs = float(sum_mean_accuracy_across_runs) / denom

    stability_metrics: Dict[str, float] = {
        "n_items_total": float(total_items),
        "n_items_processed": float(processed_items),
        "coverage": float(coverage),

        "stability_at_k": float(stability_at_k),
        "mean_consistency_at_k": float(mean_consistency_at_k),
        "stable_and_correct_rate": float(stable_and_correct_rate),
        "stable_but_wrong_rate": float(stable_but_wrong_rate),

        "mode_correct_rate": float(mode_correct_rate),
        "mean_accuracy_across_runs": float(mean_accuracy_across_runs),
        "mean_entropy": float(mean_entropy),
        "mean_normalized_entropy": float(mean_normalized_entropy),
        "mean_flip_rate": float(mean_flip_rate),
    }

    for lab in templates.BEHAVIOR_LABELS:
        stats = per_gold_stats[lab]
        count_lab = stats["count"]

        lab_stability = (stats["stable"] / count_lab) if count_lab else 0.0
        lab_mean_consistency = (stats["sum_consistency"] / count_lab) if count_lab else 0.0

        lab_mean_entropy = (stats["sum_entropy"] / count_lab) if count_lab else 0.0
        lab_mean_norm_entropy = (stats["sum_norm_entropy"] / count_lab) if count_lab else 0.0
        lab_mean_flip_rate = (stats["sum_flip_rate"] / count_lab) if count_lab else 0.0
        lab_mean_acc_runs = (stats["sum_mean_accuracy_across_runs"] / count_lab) if count_lab else 0.0
        lab_mode_correct = (stats["mode_correct"] / count_lab) if count_lab else 0.0

        stability_metrics[f"stability_at_k_gold_{lab}"] = float(lab_stability)
        stability_metrics[f"mean_consistency_at_k_gold_{lab}"] = float(lab_mean_consistency)

        stability_metrics[f"mean_entropy_gold_{lab}"] = float(lab_mean_entropy)
        stability_metrics[f"mean_normalized_entropy_gold_{lab}"] = float(lab_mean_norm_entropy)
        stability_metrics[f"mean_flip_rate_gold_{lab}"] = float(lab_mean_flip_rate)
        stability_metrics[f"mean_accuracy_across_runs_gold_{lab}"] = float(lab_mean_acc_runs)
        stability_metrics[f"mode_correct_rate_gold_{lab}"] = float(lab_mode_correct)

    return stability_metrics, stability_example_logs


# _________________________________________-

WORKDIR_BASE = Path(CFG["run"]["workdir_base"]).resolve()


def _sanitize_run_key(s: str) -> str:
    """
    Sanitize a RUN_KEY to be filesystem-safe (letters/digits/_/- only).
    Use before creating run directories to avoid OS/path issues.
    """
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    return s


def resolve_run_key_notebook(workdir_base: Path, configured_run_key: Optional[str] = None) -> str:
    """
    Resolve RUN_KEY from config/env first, otherwise prompt the user (notebook-friendly).
    Returns a UUID-based key when input is empty to guarantee uniqueness.
    """
    # 1) priorità a config (che già incorpora eventuale env RUN_KEY)
    cfg_key = (configured_run_key or "").strip()
    if cfg_key:
        return _sanitize_run_key(cfg_key)

    # 2) altrimenti chiedi in input
    try:
        name = input("RUN_KEY (name folder) [Enter = generate uuid]: ").strip()
    except Exception:
        name = ""

    if not name:
        return uuid.uuid4().hex
    return _sanitize_run_key(name)


run_key = resolve_run_key_notebook(WORKDIR_BASE, CFG["run"].get("run_key"))
RUN_DIR = (WORKDIR_BASE / "runs" / run_key).resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)

# messaggio esplicito
if any(RUN_DIR.iterdir()):
    print(f"[RESUME] RUN_DIR esistente: {RUN_DIR}")
else:
    print(f"[NEW] RUN_DIR creato: {RUN_DIR}")


def enforce_no_mixed_stability_runs(
    do_mcq_stability: bool,
    do_llm_judge_stability: bool,
    *,
    do_mcq_lp_stability: bool = False,
) -> None:
    """
    Prevent running multiple stability pipelines in the same execution.

    Rationale: combining stability pipelines in one run can silently reuse checkpoints,
    split artifacts across different exp_name conventions, and make llm_calls look empty
    due to resume behavior. Run each stability sweep in a separate run/session.
    """
    if do_mcq_stability and do_llm_judge_stability:
        msg = (
            "ATTENTION: You enabled BOTH MCQ stability and LLM-as-judge stability in the same run.\n"
            "Please run these stability sweeps SEPARATELY (one per execution) to avoid checkpoint reuse "
            "and confusing artifacts/llm_calls.\n"
            "Set either DO_MCQ_STABILITY=False OR DO_LLM_JUDGE_STABILITY=False."
        )
        raise RuntimeError(msg)

    # (Optional) if you also want to forbid mixing MCQ stability and MCQ logprob stability:
    if do_mcq_stability and do_mcq_lp_stability:
        msg = (
            "ATTENTION: You enabled BOTH MCQ stability and MCQ logprob stability in the same run.\n"
            "Please run these stability sweeps SEPARATELY (one per execution)."
        )
        raise RuntimeError(msg)


# =============================================================================
# MAIN (CONFIG VALUES FROM TOML / ENV)
# =============================================================================

RUN_KEY = run_key

# -----------------------------
# DATASET / MODELS / PIPELINES
# -----------------------------
EVAL_DATA_PATH = str(CFG["data"]["eval_data_path"])

JUDGE_MODEL = str(CFG["models"]["judge_model"])
TARGET_MODEL = str(CFG["models"]["target_model"])

N_PER_LABEL = int(CFG["data"]["n_per_label"])  # if USE_FULL_DATASET is True no sampling occurs

# OVERRIDE DELIMITER
FORCE_TARGET_DELIMITER: Optional[str] = CFG["models"].get("force_target_delimiter")

DO_LLM_JUDGE = bool(CFG["pipelines"]["do_llm_judge"])
DO_MCQ = bool(CFG["pipelines"]["do_mcq"])
DO_MCQ_LOGPROB = bool(CFG["pipelines"]["do_mcq_logprob"])

# Temperatures (kept as globals for downstream notebook runner compatibility)
TARGET_TEMPERATURE = float(CFG["pipelines"].get("target_temperature", 0.0))
JUDGE_TEMPERATURE = float(CFG["pipelines"].get("judge_temperature", 0.0))
MCQ_TEMPERATURE = float(CFG["pipelines"].get("mcq_temperature", 0.0))

# -----------------------------
# DATASET SELECTION
# -----------------------------
USE_FULL_DATASET = bool(CFG["data"]["use_full_dataset"])  # True = usa tutti gli esempi; False = stratified_subsample_by_label
SUBSAMPLE_SEED = int(CFG["data"]["subsample_seed"])

# ------------------------------
# API SEED
# ------------------------------

API_SEED = int(CFG["run"]["api_seed"])

# -----------------------------
# REPRODUCIBILITY / SEED CONTEXT (LOGGED EVERYWHERE)
# -----------------------------
API_SEED = int(API_SEED)  # ensure stable type

SEED_CONTEXT: Dict[str, Any] = {
    "api_seed": API_SEED,
    "subsample_seed": (None if USE_FULL_DATASET else SUBSAMPLE_SEED),
    "python_random_seed": (None if USE_FULL_DATASET else SUBSAMPLE_SEED),
}

# -----------------------------
# STABILITY SWEEPS (recommended)
# -----------------------------
MCQ_STABILITY_N_RUNS = int(CFG["stability"]["mcq_stability_n_runs"])
MCQ_STABILITY_TEMPERATURES = [float(x) for x in CFG["stability"]["mcq_stability_temperatures"]]

LLM_JUDGE_STABILITY_N_RUNS = int(CFG["stability"]["llm_judge_stability_n_runs"])
TARGET_STABILITY_TEMPERATURES = [float(x) for x in CFG["stability"]["target_stability_temperatures"]]  # this is T_target; judge stays at 0

DO_MCQ_STABILITY = bool(CFG["stability"]["do_mcq_stability"])
DO_LLM_JUDGE_STABILITY = bool(CFG["stability"]["do_llm_judge_stability"])

# DO_MCQ_STABILITY = False
# MCQ_STABILITY_N_RUNS = 5
# MCQ_STABILITY_TEMPERATURE = 0.0

# DO_LLM_JUDGE_STABILITY = False
# LLM_JUDGE_STABILITY_N_RUNS = 5
# TARGET_STABILITY_TEMPERATURE = 0.0

DO_MCQ_LP_STABILITY = bool(CFG["stability"]["do_mcq_lp_stability"])
MCQ_LP_STABILITY_N_RUNS = int(CFG["stability"]["mcq_lp_stability_n_runs"])
MCQ_LP_STABILITY_TEMPERATURE = float(CFG["stability"]["mcq_lp_stability_temperature"])

_DEFAULT_AVAILABLE_MODELS = [
    "qwen3-32b",
    "minimax-m2",
    "gpt-oss-120b",
    "react-agent-mistral-3.2",
    "qwen3-coder-30b-a3b-instruct",
    "gpt-4o",
    "qwen-coder-2.5-base",
    "llama-3.3-70b-instruct",
    "mistral-small-3.2-24b",
]
_cfg_models = CFG.get("available_models", {}).get("items", [])
AVAILABLE_MODELS = [str(m) for m in _cfg_models] if _cfg_models else _DEFAULT_AVAILABLE_MODELS

logging.info("Loaded config from: %s", W2C_CONFIG_PATH)
logging.info("Config fingerprint payload (sanitized): %s", config_pretty(W2C_CONFIG_FOR_FINGERPRINT))

from w2c_notebook_runner import run_when2call_notebook
run_when2call_notebook(globals())