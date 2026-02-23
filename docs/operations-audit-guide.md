# Operations & Audit Guide

## When2Call API Evaluation Suite

### Runs, Checkpoints, Audit Trails, and MLflow Traceability

This document specifies the operational model, traceability mechanisms, and audit artifacts of the When2Call API Evaluation Suite, with reference to the behavior implemented in the main evaluation script.

It is intended for technical reviewers, maintainers, and research teams requiring reproducible and inspectable API-based evaluation pipelines.

---

## 1. Scope

This guide describes:

* run/session organization
* checkpointing and resume behavior
* structured audit logging
* method-specific artifacts
* MLflow linkage and access patterns
* traceability and reproducibility controls

It complements the general README by focusing on **operational evidence and auditability**, not only execution instructions.

---

## 2. Operational Rationale

API-based LLM evaluation introduces operational variability not present in offline inference workflows. Common sources include:

* transient network failures and timeouts
* rate limiting (429)
* provider-side errors (5xx)
* provider-specific response schema differences
* malformed judge outputs
* logprob failures or boundary misalignment
* stochastic behavior across repeated runs

Without structured persistence and audit trails, such events reduce confidence in reported metrics.

The implementation addresses this by treating evaluation as a **stateful, traceable process** with explicit recovery and audit mechanisms.

---

## 3. Core Design Principles

### 3.1 Deterministic sessioning by configuration fingerprint

Each configuration is serialized canonically and hashed to produce a deterministic session fingerprint. Session outputs are stored under:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/
```

This design provides:

* separation of configurations under a shared `RUN_KEY`
* deterministic resume behavior
* protection against result contamination when parameters change

### 3.2 Append-only UUID-level checkpoints

Per-example outputs are written incrementally to JSONL checkpoint files. State reconstruction on resume is based on UUID-keyed loading with **last-write-wins** semantics.

This enables:

* interruption recovery
* partial-progress preservation
* idempotent resume
* post-hoc reconstruction of processing history

### 3.3 Explicit completion markers

Each pipeline writes a `_DONE.json` marker in its checkpoint directory. The marker is used to determine whether a method should be skipped or resumed.

This prevents accidental reruns of completed methods and preserves method-level execution status.

### 3.4 Structured fallback auditing

Fallbacks, coercions, and recovery actions are recorded as structured JSONL events (`audit_fallbacks.jsonl`) rather than only console logs.

This enables:

* quantitative reporting of fallback frequency
* UUID-level forensic inspection
* stage-specific failure analysis
* reproducibility of forced decisions

### 3.5 Dual persistence: filesystem + MLflow linkage

The runtime persists local artifacts and stores MLflow run identifiers in session scope (`mlflow_ids.json`).

This provides:

* a durable local evidence trail (checkpoints, audit logs, artifacts)
* an experiment-tracking layer (MLflow)
* restart-safe continuity between local session state and MLflow runs

---

## 4. Run and Session Organization

### 4.1 `RUN_KEY` (top-level run bucket)

`RUN_KEY` identifies the top-level output directory:

```text
runs/<RUN_KEY>/
```

It may be supplied via environment variable or interactive notebook input. The value is sanitized for filesystem safety.

### 4.2 Session fingerprint (configuration instance)

Each distinct configuration under a given `RUN_KEY` is mapped to a deterministic session fingerprint:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/
```

The fingerprint is derived from canonical JSON (sorted keys, compact separators) and hashed (SHA-256, truncated).

This mechanism ensures configuration changes produce a new session rather than overwriting prior artifacts.

---

## 5. Session Directory Layout

Typical session structure:

```text
runs/
└── <RUN_KEY>/
    └── sessions/
        └── <FINGERPRINT>/
            ├── manifest.json
            ├── mlflow_ids.json
            ├── checkpoints/
            │   ├── llm_judge/
            │   ├── mcq/
            │   ├── mcq_logprob/
            │   ├── stability_mcq_k=10_T=0.0/
            │   ├── stability_llm_judge_k=10_T=0.3/
            │   └── ...
            └── artifacts_local/
                ├── llm_judge/
                ├── mcq/
                ├── mcq_logprob/
                ├── stability_mcq_k=10_T=0.0/
                └── ...
```

This directory is the authoritative local record of a session.

---

## 6. Session Metadata Files

### 6.1 `manifest.json`

`manifest.json` records session identity and configuration context. It includes, typically:

* `schema_version`
* `fingerprint`
* `created_at`
* `updated_at`
* `config`

Operational use:

* verification of configuration provenance
* indexing and discovery of sessions
* traceability for reporting and publication

### 6.2 `mlflow_ids.json`

`mlflow_ids.json` stores MLflow linkage information for the session, typically:

* `parent_run_id`
* `child_run_ids` keyed by method

Operational use:

* resume-safe MLflow continuity
* deterministic mapping from filesystem session to MLflow UI entities
* debugging of interrupted or partially logged runs

---

## 7. Method Execution Lifecycle

The script implements a method lifecycle via helpers such as:

* `should_run_method(exp_name)`
* `is_method_done(exp_name)`
* `mark_method_done(exp_name, payload)`

The standard lifecycle is:

1. method start or resume
2. incremental checkpoint writes
3. metrics/artifact generation
4. completion marker write

This pattern supports long-running API evaluations without requiring all-or-nothing execution.

---

## 8. Checkpointing Model

### 8.1 Format choice: JSONL

JSONL is used because it supports:

* append-only writes
* incremental processing
* recovery after interruption
* simple downstream ingestion (pandas, scripting)
* straightforward line-oriented inspection

### 8.2 UUID-keyed reconstruction

Checkpoint files are loaded into memory as maps keyed by `uuid`. If multiple rows exist for the same UUID, the last entry is retained.

This behavior is intentional and compatible with append-only resume workflows.

### 8.3 Append semantics

Checkpoint writes are performed via an append helper that ensures directory creation and writes one JSON object per line.

This design minimizes operational risk during long runs.

---

## 9. Audit Event Model (`audit_fallbacks.jsonl`)

Each method may emit structured audit events to:

```text
checkpoints/<exp_name>/audit_fallbacks.jsonl
```

### 9.1 Common event structure

Events generally contain:

* `ts_utc`
* `run_key`
* `session_fingerprint`
* `exp_name`
* `api_seed`

Plus event-specific fields such as:

* `uuid`
* `pipeline`
* `stage`
* `fallback_type`
* `severity`
* `details` (structured metadata)

### 9.2 Audit value

This event model provides machine-readable operational evidence for:

* fallback paths
* forced coercions
* parsing failures
* scoring boundary issues
* metrics-stage normalization actions

---

## 10. Audit Taxonomy Implemented in the Main Script

### 10.1 LLM-as-judge parse/retry/fallback events

When judge output parsing fails:

1. first parse failure is logged
2. repair/rewrite retry is attempted
3. second parse failure triggers fallback to `cannot_answer`
4. fallback is logged

Typical event types include:

* `judge_json_parse_failed_first`
* `judge_json_parse_failed_second_fallback_to_cannot_answer`

### 10.2 MCQ logprob all-`-inf` fallback

If all MCQ logprob scores are non-finite, the pipeline falls back to MCQ string-based classification.

Typical event type:

* `all_logprobs_-inf_string_fallback`

### 10.3 MCQ logprob token-prefix mismatch / LCP split

If prompt-token prefix alignment fails during score attribution, the pipeline can use LCP-based token splitting and record the event.

Typical event type:

* `token_prefix_mismatch_lcp_split`

### 10.4 Metrics-stage coercions and missing predictions

`compute_metrics(...)` can emit audit events for:

* missing prediction UUIDs
* invalid predicted labels coerced to `cannot_answer`

Typical event types:

* `missing_prediction_uuid`
* `invalid_label_coercion_to_cannot_answer`

---

## 11. Method-Specific Checkpoints and Audit Artifacts

### 11.1 LLM-as-Judge (`exp_name = llm_judge`)

Directory:

```text
checkpoints/llm_judge/
```

#### `target_responses.jsonl`

Stores target-model free-form outputs per UUID.

Typical fields:

* `uuid`
* `raw_text`
* `target_model`
* `temperature`
* `api_seed`

#### `judge_decisions.jsonl`

Stores parsed judge decisions and parsing/fallback metadata.

Typical fields:

* `uuid`
* `predicted_label`
* `judge_raw`
* `judge_rationale`
* `judge_parse_failed_first`
* `judge_parse_failed_second`
* `judge_used_retry`
* `judge_fallback_to_cannot_answer`
* `judge_exception_first`
* `judge_exception_second`

#### `audit_fallbacks.jsonl`

Stores structured judge parsing and metrics-stage audit events.

#### `_DONE.json`

Method completion marker.

---

### 11.2 MCQ String-Based (`exp_name = mcq`)

Directory:

```text
checkpoints/mcq/
```

#### `mcq_predictions.jsonl`

Stores per-example MCQ classifier outputs.

Typical fields:

* `uuid`
* `gold_label`
* `gold_index`
* `answer_names`
* `predicted_index`
* `predicted_label`
* `raw_mcq_output`
* `target_model`
* `temperature`
* `api_seed`

#### `audit_fallbacks.jsonl`

Primarily metrics-stage coercion/missing-prediction events.

#### `_DONE.json`

Method completion marker.

---

### 11.3 MCQ Logprob (`exp_name = mcq_logprob`)

Directory:

```text
checkpoints/mcq_logprob/
```

This method produces the most detailed scoring diagnostics.

#### `mcq_logprob_predictions.jsonl`

Stores final labels for multiple scoring variants and fallback metadata.

Typical fields:

* `uuid`
* `gold_label`
* `predicted_label_raw`
* `predicted_label_norm_bytes`
* `predicted_label_norm_tokens`
* `predicted_label_norm_chars`
* `mode` (`llama_scoring` or `string_fallback`)
* `target_model`
* delimiter metadata

Regular scoring rows may include:

* `scores_raw`
* `scores_norm_bytes`
* `scores_norm_tokens`
* `scores_norm_chars`
* `gold_index`

Fallback rows may include:

* `fallback_trigger`
* `string_fallback_pred_idx`
* `string_fallback_raw_output`
* `string_fallback_answer_names`

#### `debug_per_choice.jsonl`

Per-choice scoring diagnostics for boundary and normalization analysis.

Typical fields:

* `uuid`
* `gold_label`
* `answer_name`
* `choice_text_preview`
* `len_chars`
* `len_bytes`
* `num_tokens`
* `raw_score`
* `token_norm_score`
* `used_lcp_split`
* `prompt_tok_count`
* `lcp`

#### `audit_fallbacks.jsonl`

Stores:

* all-`-inf` fallback events
* LCP split events
* metrics-stage coercions

#### `_DONE.json`

Method completion marker.

---

### 11.4 Stability Sweeps (`stability_*`)

Stability runs are isolated by method and temperature, e.g.:

* `stability_mcq_k=10_T=0.0`
* `stability_llm_judge_k=10_T=0.7`
* `stability_mcq_logprob_k=5_T=0.0`

This prevents checkpoint mixing across stability configurations.

#### `<ckpt_name>.jsonl`

Stores repeated-run traces per example.

Typical fields:

* `uuid`
* `gold_label`
* `n_runs`
* `run_labels`
* `mode_label`
* `mode_count`
* `consistency`
* `is_stable`
* `is_mode_correct`
* `is_stable_and_correct`
* `is_stable_but_wrong`
* `mean_accuracy_across_runs`
* `entropy`
* `normalized_entropy`
* `flip_rate`
* `eval_type`
* `target_model`
* `temperature`
* `judge_model`
* `forced_delimiter`
* `ckpt_name`

Method-specific additions:

* MCQ / MCQ-logprob: `run_indices`, `run_aux`
* LLM-as-judge: `run_target_texts`, `run_judge_raw_outputs`

---

## 12. Audit Summaries and Fallback Statistics

The script includes helpers (e.g., `summarize_audit_fallbacks(...)`) that aggregate `audit_fallbacks.jsonl` into summary statistics such as:

* total event count
* number of UUIDs affected
* event counts by `fallback_type`
* event counts by `stage`
* event counts by `severity`
* forced fallback event count
* forced fallback UUID count

These summaries support quality assurance, robustness reporting, and operational review.

---

## 13. Metrics Traceability and Coercion Transparency

`compute_metrics(...)` can emit audit events during metric computation, including:

* missing UUIDs in prediction dictionaries
* invalid labels coerced to `cannot_answer`

As a result, final metrics can be traced to explicit normalization behavior rather than opaque post-processing.

This supports:

* reproducibility review
* discrepancy analysis
* publication-grade reporting of edge-case handling

---

## 14. API Call Traceability

The `raw_chat_completion(...)` wrapper records incremental API call metadata via `log_llm_call_incremental(...)`, including:

* provider (`jrc` / `gemini`)
* base URL
* model name
* generation settings
* pipeline metadata (`pipeline`, `uuid`, `source`)
* request payload structure (including payload keys)
* response payload (when parseable)
* parsed text content
* HTTP status code
* latency

This establishes a trace chain from request execution to final metrics:

**API call record → checkpoint row → audit events → metrics/artifacts**

---

## 15. Multi-Provider Operation and Audit Implications

`/chat/completions` routing is selected by model-name prefix:

* `gemini-*` → Gemini OpenAI-compatible endpoint
* otherwise → JRC endpoint

Audit interpretation should always consider:

* provider
* base URL
* model
* endpoint family (`/chat/completions` vs `/completions`)

### Current limitation

The MCQ logprob pipeline relies on `/completions` echo/logprobs and is implemented against the JRC-compatible path in the current script.

This limitation should be treated as part of the operational configuration and documented in run metadata.

---

## 16. MLflow Integration Model

The script integrates with an MLflow helper layer (`w2c_mlflow_logging`) and persists MLflow identifiers per session.

The observable behavior supports:

* parent/child run organization
* JSON/JSONL artifact logging
* incremental call logging
* resume-safe run reuse

### 16.1 Recommended MLflow hierarchy

A suitable hierarchy is:

* **Parent run**: session-level context (config, dataset, shared metadata)
* **Child runs**: method-level executions

  * `llm_judge`
  * `mcq`
  * `mcq_logprob`
  * stability sweeps

This aligns MLflow structure with filesystem structure.

### 16.2 Filesystem and MLflow complementarity

Filesystem artifacts are optimized for:

* checkpoint-based recovery
* local forensic inspection
* line-level auditability

MLflow is optimized for:

* experiment comparison
* team-facing run browsing
* metric dashboards
* centralized artifact access

The dual-record model supports both operational recovery and analytical review.

---

## 17. Accessing the MLflow Platform

The MLflow tracking backend is configured outside the main script (environment + helper module). The main script provides run linkage via `mlflow_ids.json`.

### 17.1 Hosted MLflow deployments

To locate a run in a hosted MLflow UI:

1. Open the tracking UI configured for the environment.
2. Identify the relevant run using:

   * `RUN_KEY`
   * session fingerprint
   * target/judge model names
   * method names
3. Open the parent run and inspect child runs.

Local-to-MLflow mapping is stored in:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/mlflow_ids.json
```

### 17.2 Local MLflow UI

If using local MLflow storage, a local UI can be started with:

```bash
mlflow ui
```

(Default local URL is commonly `http://127.0.0.1:5000`, unless configured otherwise.)

### 17.3 If runs are not visible in MLflow

Recommended validation sequence:

1. Confirm session via `manifest.json`
2. Confirm MLflow run IDs via `mlflow_ids.json`
3. Confirm method artifacts exist under `artifacts_local/<exp_name>/`
4. Confirm MLflow tracking URI configuration in the environment/helper layer

---

## 18. Typical Artifacts in `artifacts_local/` and MLflow

The exact contents depend on the notebook runner and helper implementation, but the architecture supports the following pattern.

### 18.1 Local staging (`artifacts_local/<exp_name>/`)

Typical files include:

* `metrics.json`
* confusion-matrix summaries
* prediction exports
* audit summaries
* stability summaries
* diagnostic JSONL copies

### 18.2 MLflow artifacts (child runs)

Typical child-run artifacts include:

* metrics JSON files
* checkpoint summaries
* audit logs or audit summaries
* method-specific diagnostics
* configuration metadata

### 18.3 Recommended audit cross-checks

For method-level validation, compare:

* local `artifacts_local/<exp_name>/metrics.json`
* corresponding MLflow child-run artifacts
* `checkpoints/<exp_name>/audit_fallbacks.jsonl`

This confirms alignment between reported metrics and recorded operational events.

---

## 19. Standard Operational Workflows

### 19.1 Standard execution

1. Set or resolve `RUN_KEY`
2. Launch the notebook runner
3. Confirm session creation/resume
4. Allow methods to execute incrementally
5. Confirm `_DONE.json` markers
6. Review audit logs
7. Review method artifacts
8. Review MLflow runs

### 19.2 Resume after interruption

Re-running with the same `RUN_KEY` and identical configuration results in:

* same session fingerprint
* checkpoint reload
* processing only of missing UUIDs
* reuse of MLflow linkage via `mlflow_ids.json`

### 19.3 Rerun with modified configuration

Any change in configuration (model, seeds, temperatures, delimiter, dataset settings, etc.) produces a new fingerprint and a new session directory under the same `RUN_KEY`.

This prevents artifact mixing across configurations.

---

## 20. QA and Audit Checklist

### Session integrity

* [ ] `manifest.json` present and consistent with intended configuration
* [ ] session fingerprint matches expected config
* [ ] `mlflow_ids.json` present (if MLflow enabled)

### Method completion

* [ ] `_DONE.json` present for each reported method
* [ ] checkpoint file counts are plausible

### Audit review

* [ ] `audit_fallbacks.jsonl` reviewed for each method
* [ ] forced fallback counts quantified
* [ ] non-trivial parse/fallback events documented

### Metrics validation

* [ ] `metrics.json` present locally and/or in MLflow
* [ ] missing UUID and invalid-label counts reviewed
* [ ] confusion matrix reviewed for expected structure

### Stability validation (if reported)

* [ ] checkpoint namespace matches intended `k` and temperature
* [ ] no mixed stability sweeps in the same execution context
* [ ] per-item traces sampled for spot checks

---

## 21. Traceability and Reproducibility Characteristics

The implementation provides a strong operational foundation for API-based benchmarking:

* deterministic session identity by config fingerprint
* append-only UUID-level checkpoints
* explicit completion markers
* structured fallback audit logs
* per-method and per-example diagnostics
* resume-safe MLflow linkage
* repeated-run stability traces

These features make reported results inspectable, recoverable, and attributable to concrete execution records.

---

## 22. Operational Caveats

The following variability sources remain inherent to API-based evaluation:

* provider-side model updates
* backend routing changes
* partial non-determinism despite seed support
* endpoint-level parameter incompatibilities
* differences in support for `logprobs`, `reasoning`, or token settings

The runtime mitigates these issues by persisting provider metadata, retries, fallback paths, and stability outputs.

---

## 23. Documentation Blurb (Traceability / Auditability)

> **Traceability, auditability, and reproducibility are first-class concerns in this evaluation suite.** Each pipeline writes append-only UUID-level checkpoints, structured fallback audit logs, deterministic session manifests keyed by configuration fingerprints, and MLflow-linked artifacts with resume-safe run identifiers. Reported metrics can therefore be traced to raw outputs, parsing recovery paths, fallback decisions, and per-example scoring records.

---

## 24. Quick Reference (Initial Inspection Order)

### Session-level inspection

1. `runs/<RUN_KEY>/sessions/<FINGERPRINT>/manifest.json`
2. `runs/<RUN_KEY>/sessions/<FINGERPRINT>/mlflow_ids.json`

### Method-level inspection

1. `.../checkpoints/<exp_name>/audit_fallbacks.jsonl`
2. `.../checkpoints/<exp_name>/*.jsonl` (predictions / decisions)
3. `.../artifacts_local/<exp_name>/metrics.json`

### MCQ logprob anomaly triage

1. `mcq_logprob_predictions.jsonl`
2. `debug_per_choice.jsonl`
3. `audit_fallbacks.jsonl`

### LLM-as-judge anomaly triage

1. `target_responses.jsonl`
2. `judge_decisions.jsonl`
3. `audit_fallbacks.jsonl`

---
