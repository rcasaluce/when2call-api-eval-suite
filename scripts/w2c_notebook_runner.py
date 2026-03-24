# w2c_notebook_runner.py
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def run_when2call_notebook(g: dict) -> None:
    """
    Notebook runner:
      - Consumes the notebook namespace (globals()) as dict `g`
      - Executes the equivalent of the old `if __name__ == "__main__": ...` block
      - Updates g["RUN_DIR"] (and other globals where needed) so notebook functions see the correct globals

    Compatibility goals:
      - Backward compatible with "all symbols defined in the notebook"
      - Forward compatible with "some symbols moved to w2c_templates.py"
      - Robust defaults for optional configuration knobs
    """

    # -------------------------------------------------------------------------
    # 0) Optional imports / module fallbacks
    # -------------------------------------------------------------------------
    # Some helpers may live in a dedicated module rather than notebook globals.
    # We resolve symbols from `g` first, then fall back to `w2c_templates` if importable.
    try:
        import w2c_templates as templates  # type: ignore
    except Exception:
        templates = None  # type: ignore

    def _resolve_symbol(name: str) -> Any:
        """
        Resolve a symbol with this precedence:
          1) Notebook globals dict `g`
          2) w2c_templates module (if available)
        """
        if name in g:
            return g[name]
        if templates is not None and hasattr(templates, name):
            return getattr(templates, name)
        return None

    def _require_symbol(name: str) -> Any:
        """
        Resolve a required symbol or raise a clear error.
        """
        v = _resolve_symbol(name)
        if v is None:
            raise RuntimeError(
                f"Missing required symbol: {name} (not found in notebook globals or w2c_templates)"
            )
        return v

    def _optional_symbol(name: str, default: Any) -> Any:
        """
        Resolve an optional symbol; if missing, return default and also set it in `g`
        so downstream notebook code sees a consistent global.
        """
        v = _resolve_symbol(name)
        if v is None:
            g[name] = default
            return default
        return v

    # -------------------------------------------------------------------------
    # 1) Required symbols check
    # -------------------------------------------------------------------------
    # Important:
    # - FORCE_TARGET_DELIMITER is intentionally OPTIONAL: default is None.
    # - BEHAVIOR_LABELS can be resolved from either notebook globals or w2c_templates.
    required = [
        # dirs / keys
        "WORKDIR_BASE",
        "RUN_KEY",

        # config
        "EVAL_DATA_PATH",
        "USE_FULL_DATASET",
        "N_PER_LABEL",
        "SUBSAMPLE_SEED",
        "TARGET_MODEL",
        "JUDGE_MODEL",
        "DO_LLM_JUDGE",
        "DO_MCQ",
        "DO_MCQ_LOGPROB",
        "DO_MCQ_STABILITY",
        "MCQ_STABILITY_N_RUNS",
        "MCQ_STABILITY_TEMPERATURES",
        "DO_LLM_JUDGE_STABILITY",
        "LLM_JUDGE_STABILITY_N_RUNS",
        "TARGET_STABILITY_TEMPERATURES",
        "DO_MCQ_LP_STABILITY",
        "MCQ_LP_STABILITY_N_RUNS",
        "MCQ_LP_STABILITY_TEMPERATURE",

        # NOTE: FORCE_TARGET_DELIMITER is OPTIONAL now
        # "FORCE_TARGET_DELIMITER",

        "BASE_URL",
        "MAX_RETRIES",
        "RETRY_SLEEP_SECONDS",
        "BASE_DELAY_SECONDS",
        "REASONING_EFFORT",
        "REASONING_MODELS",
        "API_SEED",
        "SEED_CONTEXT",

        # core functions
        "infer_when2call_template_family",
        "resolve_target_delimiter",
        "enforce_no_mixed_stability_runs",
        "init_or_resume_session",

        "bind_runtime_dependencies",
        "_load_mlflow_ids",
        "_save_mlflow_ids",
        "_utc_now_iso",

        "init_logging_context",
        "log_json_artifact",
        "log_jsonl_artifact",
        "flush_inputs_parent",
        "mlflow_child_run",

        "load_when2call_jsonl",
        "stratified_subsample_by_label",
        "method_status_summary",
        "should_run_method",
        "get_child_checkpoint_dir",
        "mark_method_done",

        # pipelines
        "run_llm_as_judge_evaluation",
        "run_mcq_evaluation",
        "run_mcq_evaluation_logprob_index",
        "recompute_mcq_logprob_mode_counts_from_ckpt",
        "summarize_mcq_logprob_string_fallback_details",
        "summarize_audit_fallbacks",

        # misc
        "BEHAVIOR_LABELS",
        "debug_tool_hallucination",
        "print_invalid_summary",

        # logging runtime binding
        "get_child_artifact_dir",
    ]

    missing = [k for k in required if _resolve_symbol(k) is None]
    if missing:
        raise RuntimeError(
            "Missing required symbols in notebook (and w2c_templates fallback where applicable): "
            f"{missing}"
        )

    # Optional: if not provided, we use a minimal header.
    build_metrics_header = g.get("build_metrics_header")

    # -------------------------------------------------------------------------
    # 2) Bind notebook globals -> locals
    # -------------------------------------------------------------------------
    WORKDIR_BASE: Path = _require_symbol("WORKDIR_BASE")
    RUN_KEY: str = _require_symbol("RUN_KEY")

    EVAL_DATA_PATH: str = _require_symbol("EVAL_DATA_PATH")
    USE_FULL_DATASET: bool = _require_symbol("USE_FULL_DATASET")
    N_PER_LABEL: int = _require_symbol("N_PER_LABEL")
    SUBSAMPLE_SEED: int = _require_symbol("SUBSAMPLE_SEED")

    TARGET_MODEL: str = _require_symbol("TARGET_MODEL")
    JUDGE_MODEL: str = _require_symbol("JUDGE_MODEL")

    DO_LLM_JUDGE: bool = _require_symbol("DO_LLM_JUDGE")
    DO_MCQ: bool = _require_symbol("DO_MCQ")
    DO_MCQ_LOGPROB: bool = _require_symbol("DO_MCQ_LOGPROB")

    DO_MCQ_STABILITY: bool = _require_symbol("DO_MCQ_STABILITY")
    MCQ_STABILITY_N_RUNS: int = _require_symbol("MCQ_STABILITY_N_RUNS")
    MCQ_STABILITY_TEMPERATURES = _require_symbol("MCQ_STABILITY_TEMPERATURES")

    DO_LLM_JUDGE_STABILITY: bool = _require_symbol("DO_LLM_JUDGE_STABILITY")
    LLM_JUDGE_STABILITY_N_RUNS: int = _require_symbol("LLM_JUDGE_STABILITY_N_RUNS")
    TARGET_STABILITY_TEMPERATURES = _require_symbol("TARGET_STABILITY_TEMPERATURES")

    DO_MCQ_LP_STABILITY: bool = _require_symbol("DO_MCQ_LP_STABILITY")
    MCQ_LP_STABILITY_N_RUNS: int = _require_symbol("MCQ_LP_STABILITY_N_RUNS")
    MCQ_LP_STABILITY_TEMPERATURE: float = _require_symbol("MCQ_LP_STABILITY_TEMPERATURE")

    # Optional knob: default is None.
    FORCE_TARGET_DELIMITER = _optional_symbol("FORCE_TARGET_DELIMITER", None)

    BASE_URL: str = _require_symbol("BASE_URL")
    MAX_RETRIES: int = _require_symbol("MAX_RETRIES")
    RETRY_SLEEP_SECONDS: float = _require_symbol("RETRY_SLEEP_SECONDS")
    BASE_DELAY_SECONDS: float = _require_symbol("BASE_DELAY_SECONDS")
    REASONING_EFFORT = _require_symbol("REASONING_EFFORT")
    REASONING_MODELS = _require_symbol("REASONING_MODELS")

    API_SEED: int = _require_symbol("API_SEED")
    SEED_CONTEXT: Dict[str, Any] = _require_symbol("SEED_CONTEXT")

    BEHAVIOR_LABELS = _require_symbol("BEHAVIOR_LABELS")

    infer_when2call_template_family = _require_symbol("infer_when2call_template_family")
    resolve_target_delimiter = _require_symbol("resolve_target_delimiter")
    enforce_no_mixed_stability_runs = _require_symbol("enforce_no_mixed_stability_runs")
    init_or_resume_session = _require_symbol("init_or_resume_session")

    bind_runtime_dependencies = _require_symbol("bind_runtime_dependencies")
    _load_mlflow_ids = _require_symbol("_load_mlflow_ids")
    _save_mlflow_ids = _require_symbol("_save_mlflow_ids")
    _utc_now_iso = _require_symbol("_utc_now_iso")

    init_logging_context = _require_symbol("init_logging_context")
    log_json_artifact = _require_symbol("log_json_artifact")
    log_jsonl_artifact = _require_symbol("log_jsonl_artifact")
    flush_inputs_parent = _require_symbol("flush_inputs_parent")
    mlflow_child_run = _require_symbol("mlflow_child_run")

    load_when2call_jsonl = _require_symbol("load_when2call_jsonl")
    stratified_subsample_by_label = _require_symbol("stratified_subsample_by_label")
    method_status_summary = _require_symbol("method_status_summary")
    should_run_method = _require_symbol("should_run_method")
    get_child_checkpoint_dir = _require_symbol("get_child_checkpoint_dir")
    mark_method_done = _require_symbol("mark_method_done")
    get_child_artifact_dir = _require_symbol("get_child_artifact_dir")

    run_llm_as_judge_evaluation = _require_symbol("run_llm_as_judge_evaluation")
    run_mcq_evaluation = _require_symbol("run_mcq_evaluation")
    run_mcq_evaluation_logprob_index = _require_symbol("run_mcq_evaluation_logprob_index")

    recompute_mcq_logprob_mode_counts_from_ckpt = _require_symbol("recompute_mcq_logprob_mode_counts_from_ckpt")
    summarize_mcq_logprob_string_fallback_details = _require_symbol("summarize_mcq_logprob_string_fallback_details")
    summarize_audit_fallbacks = _require_symbol("summarize_audit_fallbacks")

    debug_tool_hallucination = _require_symbol("debug_tool_hallucination")
    print_invalid_summary = _require_symbol("print_invalid_summary")

    # -------------------------------------------------------------------------
    # 3) MLflow setup (order matters)
    # -------------------------------------------------------------------------
    # Use a stable tracking dir under WORKDIR_BASE so notebooks run consistently.
    mlruns_root = (WORKDIR_BASE / "mlruns").resolve()
    mlflow.set_tracking_uri(f"file:{mlruns_root.as_posix()}")

    # Ensure the experiment exists (creates it if missing).
    mlflow.set_experiment("when2call-llm-eval")

    # In notebooks, an active run may already exist; close it to avoid nesting confusion.
    try:
        mlflow.end_run()
    except Exception:
        pass

    run_name = f"when2call_target={TARGET_MODEL}_judge={JUDGE_MODEL}_n_per_label={N_PER_LABEL}"

    # -------------------------------------------------------------------------
    # 4) Stable RUN_DIR (propagate to notebook globals)
    # -------------------------------------------------------------------------
    run_dir = (WORKDIR_BASE / "runs" / RUN_KEY).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    g["RUN_DIR"] = run_dir

    # -------------------------------------------------------------------------
    # 5) Multi-session + config fingerprint
    # -------------------------------------------------------------------------
    template_family = infer_when2call_template_family(TARGET_MODEL)

    # Some families prefer an empty delimiter by default.
    auto_forced_delim = FORCE_TARGET_DELIMITER
    if auto_forced_delim is None and template_family == "harmony_gpt_oss":
        auto_forced_delim = ""

    effective_target_delimiter = resolve_target_delimiter(TARGET_MODEL, forced=auto_forced_delim)

    eval_config = {
        "eval_data_path": EVAL_DATA_PATH,
        "use_full_dataset": USE_FULL_DATASET,
        "n_per_label": (None if USE_FULL_DATASET else N_PER_LABEL),
        "subsample_seed": (None if USE_FULL_DATASET else SUBSAMPLE_SEED),

        "target_model": TARGET_MODEL,
        "judge_model": JUDGE_MODEL,

        "do_llm_judge": DO_LLM_JUDGE,
        "do_mcq": DO_MCQ,
        "do_mcq_logprob": DO_MCQ_LOGPROB,

        "do_mcq_stability": DO_MCQ_STABILITY,
        "mcq_stability_n_runs": MCQ_STABILITY_N_RUNS,
        "mcq_stability_temperatures": list(MCQ_STABILITY_TEMPERATURES),

        "do_llm_judge_stability": DO_LLM_JUDGE_STABILITY,
        "llm_judge_stability_n_runs": LLM_JUDGE_STABILITY_N_RUNS,
        "target_stability_temperatures": list(TARGET_STABILITY_TEMPERATURES),
        "judge_stability_temperature": 0.0,

        "do_mcq_lp_stability": DO_MCQ_LP_STABILITY,
        "mcq_lp_stability_n_runs": MCQ_LP_STABILITY_N_RUNS,
        "mcq_lp_stability_temperature": MCQ_LP_STABILITY_TEMPERATURE,

        "base_url": BASE_URL,
        "when2call_template_family": template_family,
        "forced_target_delimiter": repr(auto_forced_delim),
        "effective_target_delimiter": repr(effective_target_delimiter),

        "max_retries": MAX_RETRIES,
        "retry_sleep_seconds": RETRY_SLEEP_SECONDS,
        "base_delay_seconds": BASE_DELAY_SECONDS,
        "reasoning_effort": REASONING_EFFORT,
        "reasoning_models": ",".join(sorted(REASONING_MODELS)),
        "api_seed": API_SEED,
    }

    enforce_no_mixed_stability_runs(
        DO_MCQ_STABILITY,
        DO_LLM_JUDGE_STABILITY,
        do_mcq_lp_stability=DO_MCQ_LP_STABILITY,
    )

    session = init_or_resume_session(run_dir, config=eval_config)

    bind_runtime_dependencies(
        get_child_artifact_dir=get_child_artifact_dir,
        get_child_checkpoint_dir=get_child_checkpoint_dir,
        load_mlflow_ids=_load_mlflow_ids,
        save_mlflow_ids=_save_mlflow_ids,
        utc_now_iso=_utc_now_iso,
    )

    # -------------------------------------------------------------------------
    # 6) Parent run resume
    # -------------------------------------------------------------------------
    ids = _load_mlflow_ids()
    parent_run_id = ids.get("parent_run_id")
    parent_initialized = bool(ids.get("parent_initialized", False))

    if parent_run_id:
        parent_ctx = mlflow.start_run(run_id=parent_run_id)
    else:
        parent_ctx = mlflow.start_run(run_name=run_name)

    with parent_ctx:
        if not parent_run_id:
            active = mlflow.active_run()
            ids["parent_run_id"] = active.info.run_id
            ids.setdefault("child_run_ids", {})
            ids["parent_initialized"] = False
            _save_mlflow_ids(ids)
            mlflow.set_tag("created_at_utc", _utc_now_iso())
        else:
            mlflow.set_tag("resumed_at_utc", _utc_now_iso())

        artifact_dir = init_logging_context("parent")

        logging.info("Stable RUN_DIR: %s", run_dir)
        logging.info("Session fingerprint: %s", session.fingerprint)
        logging.info("Session dir: %s", session.session_dir)
        logging.info("MLflow artifact dir (parent/local): %s", artifact_dir)

        mlflow.set_tags(
            {
                "run_key": RUN_KEY,
                "session_fingerprint": session.fingerprint,
                "session_dir": str(session.session_dir),
                "pipeline": "when2call_eval_mcq_and_llm_judge",
            }
        )

        if not parent_initialized:
            mlflow.set_tags(
                {
                    "has_llm_judge": str(DO_LLM_JUDGE).lower(),
                    "has_mcq": str(DO_MCQ).lower(),
                    "has_mcq_logprob": str(DO_MCQ_LOGPROB).lower(),
                    "has_mcq_stability": str(DO_MCQ_STABILITY).lower(),
                    "has_mcq_logprob_stability": str(DO_MCQ_LP_STABILITY).lower(),
                    "has_llm_judge_stability": str(DO_LLM_JUDGE_STABILITY).lower(),
                    "api_seed": str(API_SEED),
                }
            )

            mlflow.log_params(
                {
                    "eval_data_path": EVAL_DATA_PATH,
                    "use_full_dataset": USE_FULL_DATASET,
                    "n_per_label": (None if USE_FULL_DATASET else N_PER_LABEL),
                    "subsample_seed": (None if USE_FULL_DATASET else SUBSAMPLE_SEED),

                    "target_model": TARGET_MODEL,
                    "judge_model": JUDGE_MODEL,

                    "do_llm_judge": DO_LLM_JUDGE,
                    "do_mcq": DO_MCQ,
                    "do_mcq_logprob": DO_MCQ_LOGPROB,

                    "base_target_temperature": 0.0,
                    "base_judge_temperature": 0.0,

                    "do_mcq_stability": DO_MCQ_STABILITY,
                    "mcq_stability_n_runs": MCQ_STABILITY_N_RUNS,
                    "mcq_stability_temperatures": json.dumps(list(MCQ_STABILITY_TEMPERATURES)),

                    "do_llm_judge_stability": DO_LLM_JUDGE_STABILITY,
                    "llm_judge_stability_n_runs": LLM_JUDGE_STABILITY_N_RUNS,
                    "target_stability_temperatures": json.dumps(list(TARGET_STABILITY_TEMPERATURES)),
                    "judge_stability_temperature": 0.0,

                    "do_mcq_lp_stability": DO_MCQ_LP_STABILITY,
                    "mcq_lp_stability_n_runs": MCQ_LP_STABILITY_N_RUNS,
                    "mcq_lp_stability_temperature": MCQ_LP_STABILITY_TEMPERATURE,

                    "base_url": BASE_URL,
                    "max_retries": MAX_RETRIES,
                    "retry_sleep_seconds": RETRY_SLEEP_SECONDS,
                    "base_delay_seconds": BASE_DELAY_SECONDS,
                    "reasoning_effort": REASONING_EFFORT,
                    "reasoning_models": ",".join(sorted(REASONING_MODELS)),
                    "when2call_template_family": template_family,

                    # Log both the user override knob and the effective delimiter.
                    "forced_target_delimiter": repr(FORCE_TARGET_DELIMITER),
                    "effective_target_delimiter": repr(effective_target_delimiter),

                    "run_key": RUN_KEY,
                    "session_fingerprint": session.fingerprint,
                    "api_seed": API_SEED,
                }
            )

            ids = _load_mlflow_ids()
            ids["parent_initialized"] = True
            _save_mlflow_ids(ids)

        # ---------------------------------------------------------------------
        # 7) Load + subsample dataset
        # ---------------------------------------------------------------------
        examples_all = load_when2call_jsonl(EVAL_DATA_PATH)

        label_counts_full = Counter(ex.correct_answer for ex in examples_all)
        print("\nLabel counts in full dataset:")
        for lab in BEHAVIOR_LABELS:
            print(f"  {lab:15s}: {label_counts_full.get(lab, 0)}")

        log_json_artifact(
            {lab: label_counts_full.get(lab, 0) for lab in BEHAVIOR_LABELS},
            "inputs/label_counts_full.json",
        )

        if USE_FULL_DATASET:
            examples_sub = examples_all
            logging.info("Using FULL dataset: %d examples", len(examples_sub))
        else:
            examples_sub = stratified_subsample_by_label(examples_all, n_per_label=N_PER_LABEL, seed=SUBSAMPLE_SEED)
            logging.info(
                "Using STRATIFIED subsample: %d examples (n_per_label=%s)",
                len(examples_sub),
                N_PER_LABEL,
            )

        logging.info("Config: target_model=%s, judge_model=%s", TARGET_MODEL, JUDGE_MODEL)
        logging.info("Effective subsample: %d examples", len(examples_sub))
        logging.info("API_SEED=%d", API_SEED)
        logging.info("SEED_CONTEXT=%s", SEED_CONTEXT)
        logging.info("FORCE_TARGET_DELIMITER=%r effective_target_delimiter=%r", FORCE_TARGET_DELIMITER, effective_target_delimiter)

        flush_inputs_parent()

        # ---------------------------------------------------------------------
        # 8) Session status
        # ---------------------------------------------------------------------
        expected_n = len(examples_sub)
        llm_judge_ckpt_main = get_child_checkpoint_dir("llm_judge") / "judge_decisions.jsonl"
        mcq_ckpt_main = get_child_checkpoint_dir("mcq") / "mcq_predictions.jsonl"
        mcq_lp_ckpt_main = get_child_checkpoint_dir("mcq_logprob") / "mcq_logprob_predictions.jsonl"

        logging.info("[SESSION STATUS] %s", method_status_summary("llm_judge", llm_judge_ckpt_main, expected_n))
        logging.info("[SESSION STATUS] %s", method_status_summary("mcq", mcq_ckpt_main, expected_n))
        logging.info("[SESSION STATUS] %s", method_status_summary("mcq_logprob", mcq_lp_ckpt_main, expected_n))

        # ---------------------------------------------------------------------
        # 9) Pipelines (behavior preserved)
        # ---------------------------------------------------------------------
        if DO_LLM_JUDGE and should_run_method("llm_judge"):
            with mlflow_child_run(
                "llm_judge",
                tags={"experiment": "llm_judge", "target_model": TARGET_MODEL, "judge_model": JUDGE_MODEL},
            ):
                responses, decisions, metrics_llm = run_llm_as_judge_evaluation(
                    examples_sub,
                    target_model=TARGET_MODEL,
                    judge_model=JUDGE_MODEL,
                    model_temperature=0.0,
                    judge_temperature=0.0,
                )

                print("\n=== LLM-AS-JUDGE METRICS ===")
                print("Accuracy:", round(metrics_llm.accuracy, 3))
                print("Macro-F1 (4 labels):", round(metrics_llm.macro_f1, 3))
                print("Macro-F1 (no direct_answer):", round(metrics_llm.macro_f1_no_direct, 3))
                print("Support per class (gold):", metrics_llm.per_class_support)
                for lab in BEHAVIOR_LABELS:
                    print(f"F1 {lab:15s}:", round(metrics_llm.per_class_f1[lab], 3))
                print("Tool hallucination rate:", metrics_llm.tool_hallucination_rate)
                print("Answer hallucination rate:", metrics_llm.answer_hallucination_rate)
                print("Parameter hallucination rate:", metrics_llm.parameter_hallucination_rate)
                print("Confusion matrix:", metrics_llm.confusion_matrix)
                print_invalid_summary("LLM-AS-JUDGE", metrics_llm)

                mlflow.log_metrics(
                    {
                        "accuracy": metrics_llm.accuracy,
                        "macro_f1": metrics_llm.macro_f1,
                        "macro_f1_no_direct": metrics_llm.macro_f1_no_direct,
                        "tool_hallucination_rate": metrics_llm.tool_hallucination_rate or 0.0,
                        "answer_hallucination_rate": metrics_llm.answer_hallucination_rate or 0.0,
                        "parameter_hallucination_rate": metrics_llm.parameter_hallucination_rate or 0.0,
                    }
                )
                for lab in BEHAVIOR_LABELS:
                    mlflow.log_metric(f"f1_{lab}", metrics_llm.per_class_f1[lab])
                    mlflow.log_metric(f"support_{lab}", metrics_llm.per_class_support[lab])

                metrics_llm_dict = asdict(metrics_llm)
                n_judge_parse_failed_first = sum(1 for d in decisions if getattr(d, "judge_parse_failed_first", False))
                n_judge_parse_failed_second = sum(1 for d in decisions if getattr(d, "judge_parse_failed_second", False))
                n_judge_used_retry = sum(1 for d in decisions if getattr(d, "judge_used_retry", False))
                n_judge_fallback_to_cannot = sum(
                    1 for d in decisions if getattr(d, "judge_fallback_to_cannot_answer", False)
                )

                metrics_llm_dict.update(
                    {
                        "n_judge_parse_failed_first": int(n_judge_parse_failed_first),
                        "n_judge_parse_failed_second": int(n_judge_parse_failed_second),
                        "n_judge_used_retry": int(n_judge_used_retry),
                        "n_judge_fallback_to_cannot_answer": int(n_judge_fallback_to_cannot),
                        "judge_fallback_to_cannot_answer_rate": (
                            float(n_judge_fallback_to_cannot) / float(metrics_llm.n_total) if metrics_llm.n_total else 0.0
                        ),
                        "audit_fallbacks_summary": summarize_audit_fallbacks("llm_judge"),
                    }
                )

                header = (
                    build_metrics_header(
                        method="llm_judge",
                        target_model=TARGET_MODEL,
                        judge_model=JUDGE_MODEL,
                        eval_data_path=EVAL_DATA_PATH,
                        n_per_label=N_PER_LABEL,
                        forced_delimiter=FORCE_TARGET_DELIMITER,
                        effective_delimiter=effective_target_delimiter,
                    )
                    if callable(build_metrics_header)
                    else {"method": "llm_judge"}
                )

                log_json_artifact(
                    {"header": header, "metrics": metrics_llm_dict},
                    "outputs/llm_judge/metrics_llm_judge.json",
                )
                log_jsonl_artifact([asdict(r) for r in responses], "outputs/llm_judge/target_responses.json")
                log_jsonl_artifact([asdict(d) for d in decisions], "outputs/llm_judge/judge_decisions.json")

                mark_method_done(
                    "llm_judge",
                    payload={"accuracy": metrics_llm.accuracy, "macro_f1": metrics_llm.macro_f1, "n_total": metrics_llm.n_total},
                )
        else:
            logging.info("[SKIP] llm_judge already DONE in this session (or disabled).")

        if DO_MCQ and should_run_method("mcq"):
            with mlflow_child_run("mcq", tags={"experiment": "mcq", "target_model": TARGET_MODEL}):
                uuid_to_pred, metrics_mcq, mcq_example_logs = run_mcq_evaluation(
                    examples_sub,
                    target_model=TARGET_MODEL,
                    temperature=0.0,
                )
                debug_tool_hallucination(examples_sub, uuid_to_pred)

                print("\n=== MCQ (API CLASSIFIER, STRING PARSING) METRICS ===")
                print("Accuracy:", round(metrics_mcq.accuracy, 3))
                print("Macro-F1 (4 labels):", round(metrics_mcq.macro_f1, 3))
                print("Macro-F1 (no direct_answer):", round(metrics_mcq.macro_f1_no_direct, 3))
                print("Support per class (gold):", metrics_mcq.per_class_support)
                for lab in BEHAVIOR_LABELS:
                    print(f"F1 {lab:15s}:", round(metrics_mcq.per_class_f1[lab], 3))
                print("Tool hallucination rate:", metrics_mcq.tool_hallucination_rate)
                print("Answer hallucination rate:", metrics_mcq.answer_hallucination_rate)
                print("Parameter hallucination rate:", metrics_mcq.parameter_hallucination_rate)
                print("Confusion matrix:", metrics_mcq.confusion_matrix)
                print_invalid_summary("MCQ-STRING", metrics_mcq)

                mlflow.log_metrics(
                    {
                        "accuracy": metrics_mcq.accuracy,
                        "macro_f1": metrics_mcq.macro_f1,
                        "macro_f1_no_direct": metrics_mcq.macro_f1_no_direct,
                        "tool_hallucination_rate": metrics_mcq.tool_hallucination_rate or 0.0,
                        "answer_hallucination_rate": metrics_mcq.answer_hallucination_rate or 0.0,
                        "parameter_hallucination_rate": metrics_mcq.parameter_hallucination_rate or 0.0,
                    }
                )
                for lab in BEHAVIOR_LABELS:
                    mlflow.log_metric(f"f1_{lab}", metrics_mcq.per_class_f1[lab])
                    mlflow.log_metric(f"support_{lab}", metrics_mcq.per_class_support[lab])

                metrics_mcq_dict = asdict(metrics_mcq)
                n_predicted_index_minus1 = sum(1 for r in mcq_example_logs if int(r.get("predicted_index", -999)) == -1)
                n_predicted_label_invalid_raw = sum(1 for r in mcq_example_logs if r.get("predicted_label") not in BEHAVIOR_LABELS)

                metrics_mcq_dict.update(
                    {
                        "n_mcq_predicted_index_minus1": int(n_predicted_index_minus1),
                        "n_mcq_predicted_label_invalid_raw": int(n_predicted_label_invalid_raw),
                        "mcq_predicted_index_minus1_rate": (
                            float(n_predicted_index_minus1) / float(metrics_mcq.n_total) if metrics_mcq.n_total else 0.0
                        ),
                        "audit_fallbacks_summary": summarize_audit_fallbacks("mcq"),
                    }
                )

                header = (
                    build_metrics_header(
                        method="mcq",
                        target_model=TARGET_MODEL,
                        eval_data_path=EVAL_DATA_PATH,
                        n_per_label=N_PER_LABEL,
                        forced_delimiter=FORCE_TARGET_DELIMITER,
                        effective_delimiter=effective_target_delimiter,
                    )
                    if callable(build_metrics_header)
                    else {"method": "mcq"}
                )

                log_json_artifact({"header": header, "metrics": metrics_mcq_dict}, "outputs/mcq/metrics_mcq.json")
                log_jsonl_artifact(mcq_example_logs, "outputs/mcq/mcq_predictions.json")

                mark_method_done(
                    "mcq",
                    payload={"accuracy": metrics_mcq.accuracy, "macro_f1": metrics_mcq.macro_f1, "n_total": metrics_mcq.n_total},
                )
        else:
            logging.info("[SKIP] mcq already DONE in this session (or disabled).")

        if DO_MCQ_LOGPROB and should_run_method("mcq_logprob"):
            with mlflow_child_run("mcq_logprob", tags={"experiment": "mcq_logprob", "target_model": TARGET_MODEL}):
                (
                    uuid_to_pred_lp_norm_bytes,
                    uuid_to_pred_lp_raw,
                    uuid_to_pred_lp_norm_tokens,
                    uuid_to_pred_lp_norm_chars,
                    metrics_mcq_lp_norm_bytes,
                    metrics_mcq_lp_raw,
                    metrics_mcq_lp_norm_tokens,
                    metrics_mcq_lp_norm_chars,
                    mode_counts,
                ) = run_mcq_evaluation_logprob_index(
                    examples_sub,
                    target_model=TARGET_MODEL,
                    temperature=0.0,
                    forced_delimiter=FORCE_TARGET_DELIMITER,
                )

                mode_counts = recompute_mcq_logprob_mode_counts_from_ckpt()
                mcq_lp_string_fb_details = summarize_mcq_logprob_string_fallback_details()
                mcq_lp_audit_summary = summarize_audit_fallbacks("mcq_logprob")

                debug_tool_hallucination(examples_sub, uuid_to_pred_lp_norm_bytes)

                print("\n=== MCQ (LOGPROB INDEX - RAW SUM) METRICS ===")
                print("Accuracy (raw):", round(metrics_mcq_lp_raw.accuracy, 3))
                print("Macro-F1 (4 labels, raw):", round(metrics_mcq_lp_raw.macro_f1, 3))
                print("Macro-F1 (no direct_answer, raw):", round(metrics_mcq_lp_raw.macro_f1_no_direct, 3))
                print("Tool hallucination rate (raw):", metrics_mcq_lp_raw.tool_hallucination_rate)
                print("Answer hallucination rate (raw):", metrics_mcq_lp_raw.answer_hallucination_rate)
                print("Parameter hallucination rate (raw):", metrics_mcq_lp_raw.parameter_hallucination_rate)
                print("Confusion matrix (raw):", metrics_mcq_lp_raw.confusion_matrix)

                print("\n=== MCQ (LOGPROB INDEX - BYTE-LENGTH NORMALIZED) METRICS ===")
                print("Accuracy (byte-norm):", round(metrics_mcq_lp_norm_bytes.accuracy, 3))
                print("Macro-F1 (4 labels, byte-norm):", round(metrics_mcq_lp_norm_bytes.macro_f1, 3))
                print("Macro-F1 (no direct_answer, byte-norm):", round(metrics_mcq_lp_norm_bytes.macro_f1_no_direct, 3))
                print("Tool hallucination rate (byte-norm):", metrics_mcq_lp_norm_bytes.tool_hallucination_rate)
                print("Answer hallucination rate (byte-norm):", metrics_mcq_lp_norm_bytes.answer_hallucination_rate)
                print("Parameter hallucination rate (byte-norm):", metrics_mcq_lp_norm_bytes.parameter_hallucination_rate)
                print("Confusion matrix (byte-norm):", metrics_mcq_lp_norm_bytes.confusion_matrix)

                print("\n=== MCQ (LOGPROB INDEX - TOKEN-LENGTH NORMALIZED) METRICS ===")
                print("Accuracy (token-norm):", round(metrics_mcq_lp_norm_tokens.accuracy, 3))
                print("Macro-F1 (4 labels, token-norm):", round(metrics_mcq_lp_norm_tokens.macro_f1, 3))
                print("Macro-F1 (no direct_answer, token-norm):", round(metrics_mcq_lp_norm_tokens.macro_f1_no_direct, 3))
                print("Tool hallucination rate (token-norm):", metrics_mcq_lp_norm_tokens.tool_hallucination_rate)
                print("Answer hallucination rate (token-norm):", metrics_mcq_lp_norm_tokens.answer_hallucination_rate)
                print("Parameter hallucination rate (token-norm):", metrics_mcq_lp_norm_tokens.parameter_hallucination_rate)
                print("Confusion matrix (token-norm):", metrics_mcq_lp_norm_tokens.confusion_matrix)

                print("\n=== MCQ (LOGPROB INDEX - CHAR-LENGTH NORMALIZED) METRICS ===")
                print("Accuracy (char-norm):", round(metrics_mcq_lp_norm_chars.accuracy, 3))
                print("Macro-F1 (4 labels, char-norm):", round(metrics_mcq_lp_norm_chars.macro_f1, 3))
                print("Macro-F1 (no direct_answer, char-norm):", round(metrics_mcq_lp_norm_chars.macro_f1_no_direct, 3))
                print("Tool hallucination rate (char-norm):", metrics_mcq_lp_norm_chars.tool_hallucination_rate)
                print("Answer hallucination rate (char-norm):", metrics_mcq_lp_norm_chars.answer_hallucination_rate)
                print("Parameter hallucination rate (char-norm):", metrics_mcq_lp_norm_chars.parameter_hallucination_rate)
                print("Confusion matrix (char-norm):", metrics_mcq_lp_norm_chars.confusion_matrix)

                print_invalid_summary("MCQ-LOGPROB RAW", metrics_mcq_lp_raw)
                print_invalid_summary("MCQ-LOGPROB BYTE-NORM", metrics_mcq_lp_norm_bytes)
                print_invalid_summary("MCQ-LOGPROB TOKEN-NORM", metrics_mcq_lp_norm_tokens)
                print_invalid_summary("MCQ-LOGPROB CHAR-NORM", metrics_mcq_lp_norm_chars)

                mlflow.log_metrics(
                    {
                        "raw_accuracy": metrics_mcq_lp_raw.accuracy,
                        "raw_macro_f1": metrics_mcq_lp_raw.macro_f1,
                        "raw_macro_f1_no_direct": metrics_mcq_lp_raw.macro_f1_no_direct,

                        "byte_accuracy": metrics_mcq_lp_norm_bytes.accuracy,
                        "byte_macro_f1": metrics_mcq_lp_norm_bytes.macro_f1,
                        "byte_macro_f1_no_direct": metrics_mcq_lp_norm_bytes.macro_f1_no_direct,

                        "tok_accuracy": metrics_mcq_lp_norm_tokens.accuracy,
                        "tok_macro_f1": metrics_mcq_lp_norm_tokens.macro_f1,
                        "tok_macro_f1_no_direct": metrics_mcq_lp_norm_tokens.macro_f1_no_direct,

                        "char_accuracy": metrics_mcq_lp_norm_chars.accuracy,
                        "char_macro_f1": metrics_mcq_lp_norm_chars.macro_f1,
                        "char_macro_f1_no_direct": metrics_mcq_lp_norm_chars.macro_f1_no_direct,

                        "mode_llama_scoring": mode_counts.get("llama_scoring", 0),
                        "mode_string_fallback": mode_counts.get("string_fallback", 0),
                    }
                )

                for lab in BEHAVIOR_LABELS:
                    mlflow.log_metric(f"raw_f1_{lab}", metrics_mcq_lp_raw.per_class_f1[lab])
                    mlflow.log_metric(f"raw_support_{lab}", metrics_mcq_lp_raw.per_class_support[lab])

                header = (
                    build_metrics_header(
                        method="mcq_logprob",
                        target_model=TARGET_MODEL,
                        eval_data_path=EVAL_DATA_PATH,
                        n_per_label=N_PER_LABEL,
                        forced_delimiter=FORCE_TARGET_DELIMITER,
                        effective_delimiter=effective_target_delimiter,
                    )
                    if callable(build_metrics_header)
                    else {"method": "mcq_logprob"}
                )

                log_json_artifact(
                    {
                        "header": header,
                        "metrics": {
                            "norm_bytes": asdict(metrics_mcq_lp_norm_bytes),
                            "raw": asdict(metrics_mcq_lp_raw),
                            "norm_tokens": asdict(metrics_mcq_lp_norm_tokens),
                            "norm_chars": asdict(metrics_mcq_lp_norm_chars),
                        },
                        "mode_counts": mode_counts,
                        "fallbacks": {
                            **mcq_lp_string_fb_details,
                            "audit_fallbacks_summary": mcq_lp_audit_summary,
                        },
                    },
                    "outputs/mcq_logprob/metrics_mcq_logprob.json",
                )

                log_json_artifact(
                    [
                        {
                            "uuid": ex.uuid,
                            "gold_label": ex.correct_answer,
                            "predicted_label_norm_bytes": uuid_to_pred_lp_norm_bytes.get(ex.uuid, "invalid"),
                            "predicted_label_raw": uuid_to_pred_lp_raw.get(ex.uuid, "invalid"),
                            "predicted_label_norm_tokens": uuid_to_pred_lp_norm_tokens.get(ex.uuid, "invalid"),
                            "predicted_label_norm_chars": uuid_to_pred_lp_norm_chars.get(ex.uuid, "invalid"),
                        }
                        for ex in examples_sub
                    ],
                    "outputs/mcq_logprob/mcq_logprob_predictions.json",
                )

                mark_method_done(
                    "mcq_logprob",
                    payload={
                        "raw_accuracy": metrics_mcq_lp_raw.accuracy,
                        "byte_accuracy": metrics_mcq_lp_norm_bytes.accuracy,
                        "tok_accuracy": metrics_mcq_lp_norm_tokens.accuracy,
                        "char_accuracy": metrics_mcq_lp_norm_chars.accuracy,
                        "mode_counts": mode_counts,
                        "n_total": metrics_mcq_lp_raw.n_total,
                    },
                )
        else:
            logging.info("[SKIP] mcq_logprob already DONE in this session (or disabled).")

        # ---------------------------------------------------------------------
        # 10) Stability sweeps (behavior preserved)
        # ---------------------------------------------------------------------
        if DO_MCQ_STABILITY:
            run_stability_evaluation = _require_symbol("run_stability_evaluation")
            for T in MCQ_STABILITY_TEMPERATURES:
                exp_name = f"mcq_stability_k={MCQ_STABILITY_N_RUNS}_T={T}"
                if not should_run_method(exp_name):
                    logging.info("[SKIP] %s already DONE in this session.", exp_name)
                    continue
                with mlflow_child_run(
                    exp_name,
                    tags={"experiment": "mcq_stability", "target_model": TARGET_MODEL, "n_runs": str(MCQ_STABILITY_N_RUNS), "temperature": str(T)},
                ):
                    stability_metrics, stability_example_logs = run_stability_evaluation(
                        eval_type="mcq",
                        examples=examples_sub,
                        target_model=TARGET_MODEL,
                        n_runs=MCQ_STABILITY_N_RUNS,
                        temperature=T,
                        ckpt_name=exp_name,
                    )
                    mlflow.log_metrics(stability_metrics)
                    log_json_artifact(
                        {"exp_name": exp_name, "n_runs": MCQ_STABILITY_N_RUNS, "temperature": T, "metrics": stability_metrics},
                        f"outputs/{exp_name}/metrics_mcq_stability.json",
                    )
                    log_jsonl_artifact(stability_example_logs, f"outputs/{exp_name}/mcq_stability_predictions.jsonl")
                    mark_method_done(
                        exp_name,
                        payload={
                            "stability_at_k": stability_metrics.get("stability_at_k"),
                            "mean_consistency_at_k": stability_metrics.get("mean_consistency_at_k"),
                            "n_runs": MCQ_STABILITY_N_RUNS,
                            "temperature": T,
                        },
                    )
        else:
            logging.info("[SKIP] mcq_stability disabled.")

        if DO_MCQ_LP_STABILITY:
            run_stability_evaluation = _require_symbol("run_stability_evaluation")
            exp_name_lp = f"mcq_logprob_stability_k={MCQ_LP_STABILITY_N_RUNS}_T={MCQ_LP_STABILITY_TEMPERATURE}"
            if should_run_method(exp_name_lp):
                with mlflow_child_run(
                    exp_name_lp,
                    tags={"experiment": "mcq_logprob_stability", "target_model": TARGET_MODEL, "n_runs": str(MCQ_LP_STABILITY_N_RUNS), "temperature": str(MCQ_LP_STABILITY_TEMPERATURE)},
                ):
                    lp_stability_metrics, lp_stability_example_logs = run_stability_evaluation(
                        eval_type="mcq_logprob",
                        examples=examples_sub,
                        target_model=TARGET_MODEL,
                        n_runs=MCQ_LP_STABILITY_N_RUNS,
                        temperature=MCQ_LP_STABILITY_TEMPERATURE,
                        forced_delimiter=FORCE_TARGET_DELIMITER,
                        ckpt_name=exp_name_lp,
                    )
                    mlflow.log_metrics(lp_stability_metrics)
                    log_json_artifact(
                        {
                            "exp_name": exp_name_lp,
                            "n_runs": MCQ_LP_STABILITY_N_RUNS,
                            "temperature": MCQ_LP_STABILITY_TEMPERATURE,
                            "forced_delimiter": repr(FORCE_TARGET_DELIMITER),
                            "metrics": lp_stability_metrics,
                        },
                        f"outputs/{exp_name_lp}/metrics_mcq_logprob_stability.json",
                    )
                    log_jsonl_artifact(lp_stability_example_logs, f"outputs/{exp_name_lp}/mcq_logprob_stability_predictions.jsonl")
                    mark_method_done(
                        exp_name_lp,
                        payload={
                            "stability_at_k": lp_stability_metrics.get("stability_at_k"),
                            "mean_consistency_at_k": lp_stability_metrics.get("mean_consistency_at_k"),
                            "n_runs": MCQ_LP_STABILITY_N_RUNS,
                            "temperature": MCQ_LP_STABILITY_TEMPERATURE,
                            "forced_delimiter": repr(FORCE_TARGET_DELIMITER),
                        },
                    )
            else:
                logging.info("[SKIP] %s already DONE in this session.", exp_name_lp)
        else:
            logging.info("[SKIP] mcq_logprob_stability disabled.")

        if DO_LLM_JUDGE_STABILITY:
            run_stability_evaluation = _require_symbol("run_stability_evaluation")
            for T in TARGET_STABILITY_TEMPERATURES:
                exp_name = f"llm_judge_stability_k={LLM_JUDGE_STABILITY_N_RUNS}_Ttarget={T}"
                if not should_run_method(exp_name):
                    logging.info("[SKIP] %s already DONE in this session.", exp_name)
                    continue
                with mlflow_child_run(
                    exp_name,
                    tags={"experiment": "llm_judge_stability", "target_model": TARGET_MODEL, "judge_model": JUDGE_MODEL, "n_runs": str(LLM_JUDGE_STABILITY_N_RUNS), "target_temperature": str(T), "judge_temperature": "0.0"},
                ):
                    judge_stability_metrics, judge_stability_logs = run_stability_evaluation(
                        eval_type="llm_judge",
                        examples=examples_sub,
                        target_model=TARGET_MODEL,
                        judge_model=JUDGE_MODEL,
                        n_runs=LLM_JUDGE_STABILITY_N_RUNS,
                        temperature=T,
                        ckpt_name=exp_name,
                    )
                    mlflow.log_metrics(judge_stability_metrics)
                    log_json_artifact(
                        {"exp_name": exp_name, "n_runs": LLM_JUDGE_STABILITY_N_RUNS, "target_temperature": T, "judge_temperature": 0.0, "metrics": judge_stability_metrics},
                        f"outputs/{exp_name}/metrics_llm_judge_stability.json",
                    )
                    log_jsonl_artifact(judge_stability_logs, f"outputs/{exp_name}/llm_judge_stability_predictions.jsonl")
                    mark_method_done(
                        exp_name,
                        payload={
                            "stability_at_k": judge_stability_metrics.get("stability_at_k"),
                            "mean_consistency_at_k": judge_stability_metrics.get("mean_consistency_at_k"),
                            "n_runs": LLM_JUDGE_STABILITY_N_RUNS,
                            "target_temperature": T,
                            "judge_model": JUDGE_MODEL,
                            "judge_temperature": 0.0,
                        },
                    )
        else:
            logging.info("[SKIP] llm_judge_stability disabled.")

        print("\n[MLflow] Parent run completed/resumed. Inputs uploaded; each experiment saved in its own child run.")
