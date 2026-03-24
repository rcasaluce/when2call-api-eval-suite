from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


def _read_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if tomllib is not None:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return data if isinstance(data, dict) else {}
    # Python 3.10 fallback
    try:
        import tomli  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "Python < 3.11 detected and 'tomli' is not installed. "
            "Install it with: pip install tomli"
        ) from e
    with path.open("rb") as f:
        data = tomli.load(f)
    return data if isinstance(data, dict) else {}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict merge. `updates` overrides `base`.
    """
    out = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)  # type: ignore[index]
        else:
            out[k] = v
    return out


def _parse_bool_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_force_delimiter(value: Any) -> Optional[str]:
    """
    Normalizes delimiter override values.
    Accepted:
      - None / "null" / "none" -> None
      - "newline" / "\\n"      -> "\n"
      - "" / " " / any string  -> as-is
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"models.force_target_delimiter must be string|null, got {type(value).__name__}")
    v = value.strip()
    if v.lower() in {"none", "null"}:
        return None
    if v.lower() == "newline":
        return "\n"
    if v == "\\n":
        return "\n"
    return value  # preserve exact string (including empty string / single space)


def _normalize_model_prefixes(value: Any) -> list[str]:
    if value is None:
        return ["gemini-"]
    if isinstance(value, str):
        # support accidental comma-separated string
        return [p.strip().lower() for p in value.split(",") if p.strip()]
    if isinstance(value, list):
        out = []
        for p in value:
            if not isinstance(p, str):
                raise ValueError("providers.gemini.model_prefixes must contain only strings")
            s = p.strip().lower()
            if s:
                out.append(s)
        return out or ["gemini-"]
    raise ValueError("providers.gemini.model_prefixes must be list[str] or comma-separated string")


def _require_type(cfg: Dict[str, Any], path: str, typ: type) -> None:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"Missing required config field: {path}")
        cur = cur[part]
    if not isinstance(cur, typ):
        raise ValueError(f"Invalid type for '{path}': expected {typ.__name__}, got {type(cur).__name__}")


def _default_config() -> Dict[str, Any]:
    return {
        "run": {
            "run_key": None,            # resolved later (env or interactive fallback)
            "workdir_base": ".",
            "api_seed": 25,
        },
        "data": {
            "eval_data_path": "When2Call/data/test/when2call_test_mcq.jsonl",
            "use_full_dataset": False,
            "n_per_label": 5,
            "subsample_seed": 11,
        },
        "models": {
            "target_model": "gpt-oss-120b",
            "judge_model": "gpt-oss-120b",
            "force_target_delimiter": None,
            "reasoning_effort": "low",
            "reasoning_models": ["gpt-oss-120b"],
        },
        "pipelines": {
            "do_llm_judge": True,
            "do_mcq": False,
            "do_mcq_logprob": False,
            "target_temperature": 0.0,
            "judge_temperature": 0.0,
            "mcq_temperature": 0.0,
        },
        "http": {
            "max_retries": 3,
            "retry_sleep_seconds": 10.0,
            "base_delay_seconds": 1.0,
            "timeout_seconds": 60,
        },
        "providers": {
            "jrc": {
                "base_url": "https://api-gpt.jrc.ec.europa.eu/v1",
                "token": None,  # from env only
            },
            "gemini": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model_prefixes": ["gemini-"],
                "token": None,  # from env only
            },
        },
        "stability": {
            "do_mcq_stability": False,
            "mcq_stability_n_runs": 10,
            "mcq_stability_temperatures": [0.0, 0.3, 0.7],
            "do_llm_judge_stability": False,
            "llm_judge_stability_n_runs": 10,
            "target_stability_temperatures": [0.0, 0.3, 0.7],
            "do_mcq_lp_stability": False,
            "mcq_lp_stability_n_runs": 5,
            "mcq_lp_stability_temperature": 0.0,
        },
        "debug": {
            "debug_mcq_token_mismatch": False,
            "debug_mcq_token_mismatch_max": 10**9,
            "print_template_family_once": True,
        },
        "available_models": {
            "items": [],
        },
    }


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Load app configuration with precedence:
      defaults < config.toml < environment variables (.env included)

    Secrets are read from environment only:
      - TOKEN_JRC or OPENAI_API_KEY
      - TOKEN_GEMINI or GEMINI_API_KEY

    Returns a normalized dict ready to use in the current script.
    """
    load_dotenv()  # keeps current behavior

    cfg = _default_config()

    # 1) TOML file (optional)
    path = Path(config_path)
    if path.exists():
        file_cfg = _read_toml(path)
        cfg = _deep_update(cfg, file_cfg)

    # 2) ENV overrides (non-secret)
    # Run / paths
    if os.getenv("RUN_KEY"):
        cfg["run"]["run_key"] = os.getenv("RUN_KEY")
    if os.getenv("W2C_WORKDIR_BASE"):
        cfg["run"]["workdir_base"] = os.getenv("W2C_WORKDIR_BASE")

    # HTTP / retry
    if os.getenv("MAX_RETRIES"):
        cfg["http"]["max_retries"] = int(os.getenv("MAX_RETRIES", "3"))
    if os.getenv("RETRY_SLEEP_SECONDS"):
        cfg["http"]["retry_sleep_seconds"] = float(os.getenv("RETRY_SLEEP_SECONDS", "10"))
    if os.getenv("BASE_DELAY_SECONDS"):
        cfg["http"]["base_delay_seconds"] = float(os.getenv("BASE_DELAY_SECONDS", "1"))

    # Providers
    if os.getenv("JRC_BASE_URL"):
        cfg["providers"]["jrc"]["base_url"] = os.getenv("JRC_BASE_URL")
    if os.getenv("GEMINI_BASE_URL"):
        cfg["providers"]["gemini"]["base_url"] = os.getenv("GEMINI_BASE_URL")
    if os.getenv("GEMINI_MODEL_PREFIXES"):
        prefixes = [p.strip().lower() for p in os.getenv("GEMINI_MODEL_PREFIXES", "").split(",") if p.strip()]
        if prefixes:
            cfg["providers"]["gemini"]["model_prefixes"] = prefixes

    # Models / reasoning
    if os.getenv("REASONING_EFFORT") is not None:
        cfg["models"]["reasoning_effort"] = os.getenv("REASONING_EFFORT", "")

    # Optional env overrides for booleans (only if you want quick CI toggles)
    for env_name, path_str in [
        ("DO_LLM_JUDGE", "pipelines.do_llm_judge"),
        ("DO_MCQ", "pipelines.do_mcq"),
        ("DO_MCQ_LOGPROB", "pipelines.do_mcq_logprob"),
        ("DO_MCQ_STABILITY", "stability.do_mcq_stability"),
        ("DO_LLM_JUDGE_STABILITY", "stability.do_llm_judge_stability"),
        ("DO_MCQ_LP_STABILITY", "stability.do_mcq_lp_stability"),
    ]:
        v = os.getenv(env_name)
        if v is not None:
            parts = path_str.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = _parse_bool_env(v)

    # 3) Secrets from env only (never from TOML)
    cfg["providers"]["jrc"]["token"] = os.getenv("TOKEN_JRC") or os.getenv("OPENAI_API_KEY")
    cfg["providers"]["gemini"]["token"] = os.getenv("TOKEN_GEMINI") or os.getenv("GEMINI_API_KEY")

    # 4) Normalization
    cfg["providers"]["jrc"]["base_url"] = str(cfg["providers"]["jrc"]["base_url"]).rstrip("/")
    cfg["providers"]["gemini"]["base_url"] = str(cfg["providers"]["gemini"]["base_url"]).rstrip("/")

    cfg["models"]["force_target_delimiter"] = _coerce_force_delimiter(
        cfg["models"].get("force_target_delimiter")
    )

    cfg["providers"]["gemini"]["model_prefixes"] = _normalize_model_prefixes(
        cfg["providers"]["gemini"].get("model_prefixes")
    )

    # Normalize reasoning_models to a set-friendly list of strings
    rm = cfg["models"].get("reasoning_models", [])
    if isinstance(rm, list):
        cfg["models"]["reasoning_models"] = [str(x) for x in rm]
    elif isinstance(rm, str):
        cfg["models"]["reasoning_models"] = [rm]
    else:
        raise ValueError("models.reasoning_models must be list[str] or string")

    # 5) Minimal validation
    _require_type(cfg, "data.eval_data_path", str)
    _require_type(cfg, "data.use_full_dataset", bool)
    _require_type(cfg, "data.n_per_label", int)
    _require_type(cfg, "data.subsample_seed", int)

    _require_type(cfg, "models.target_model", str)
    _require_type(cfg, "models.judge_model", str)

    _require_type(cfg, "pipelines.do_llm_judge", bool)
    _require_type(cfg, "pipelines.do_mcq", bool)
    _require_type(cfg, "pipelines.do_mcq_logprob", bool)

    _require_type(cfg, "http.max_retries", int)
    if int(cfg["http"]["max_retries"]) < 1:
        raise ValueError("http.max_retries must be >= 1")

    # At least one pipeline enabled (optional but helpful)
    if not any(
        [
            cfg["pipelines"]["do_llm_judge"],
            cfg["pipelines"]["do_mcq"],
            cfg["pipelines"]["do_mcq_logprob"],
            cfg["stability"]["do_mcq_stability"],
            cfg["stability"]["do_llm_judge_stability"],
            cfg["stability"]["do_mcq_lp_stability"],
        ]
    ):
        raise ValueError("No pipeline enabled (all DO_* flags are false).")

    # JRC token check only if needed by at least one path that uses JRC endpoints in your current code.
    # (MCQ logprob currently requires JRC /completions)
    needs_jrc = bool(cfg["pipelines"]["do_mcq_logprob"] or cfg["stability"]["do_mcq_lp_stability"])
    if needs_jrc and not cfg["providers"]["jrc"]["token"]:
        raise RuntimeError(
            "MCQ logprob is enabled but JRC token is missing. "
            "Set TOKEN_JRC or OPENAI_API_KEY in .env / environment."
        )

    return cfg


def config_for_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a sanitized config dict suitable for deterministic fingerprinting.
    Removes secrets and other non-essential volatile fields.
    """
    c = copy.deepcopy(cfg)
    try:
        c["providers"]["jrc"].pop("token", None)
        c["providers"]["gemini"].pop("token", None)
    except Exception:
        pass
    return c


def config_pretty(cfg: Dict[str, Any]) -> str:
    """
    Debug/helper pretty printer (without secrets if used with config_for_fingerprint()).
    """
    return json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True)