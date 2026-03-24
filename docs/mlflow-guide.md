# MLflow Operations & Access Guide

## When2Call API Evaluation Suite

### MLflow Run Mapping, Artifacts, and Local Web UI Access

This document describes the MLflow integration model used by the When2Call API Evaluation Suite, including:

* MLflow run organization (parent/child runs)
* session-to-MLflow mapping
* local artifact staging vs MLflow artifacts
* resume behavior
* how to access the **local MLflow web interface**
* operational troubleshooting for MLflow visibility

This guide is intentionally limited to **MLflow-related behavior**.

---

## 1. Scope and Role of MLflow in This Project

MLflow is used as the experiment tracking and artifact browsing layer for API-based evaluation runs.

Within this project, MLflow complements the local filesystem by providing:

* run-level organization (session and method scope)
* centralized metrics and artifact browsing
* reproducible experiment comparison across runs
* resumable linkage between local session state and tracked runs

The local filesystem remains the authoritative checkpoint/audit store. MLflow serves as the experiment management and reporting surface.

---

## 2. MLflow Integration Pattern in the Main Script

The main script imports MLflow helper utilities from:

```python
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
```

This indicates a structured MLflow integration with support for:

* JSON/JSONL artifact logging
* incremental logging of LLM call traces
* parent/child run organization
* runtime metadata binding
* resumable child-run creation/reuse

The exact tracking URI and backend configuration are expected to be defined in the MLflow helper layer and/or environment configuration.

---

## 3. Session-to-MLflow Mapping (Critical for Traceability)

### 3.1 Session-scoped MLflow ID persistence

Each session stores MLflow linkage in:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/mlflow_ids.json
```

This file is used to persist MLflow run identifiers across process restarts.

Typical contents include:

* `parent_run_id`
* `child_run_ids` (dictionary keyed by method/pipeline)

### 3.2 Why this matters

This design provides:

* deterministic reconnection to the same MLflow runs on resume
* stable mapping between filesystem artifacts and MLflow UI runs
* protection against duplicated runs after interruption/restart

This is a strong traceability feature and should be preserved as part of the reproducibility record.

---

## 4. Recommended MLflow Run Hierarchy

A method-aligned hierarchy is the natural fit for this codebase:

* **Parent run**: session-level context

  * configuration fingerprint
  * dataset selection
  * global runtime context
* **Child runs**: pipeline-level execution units

  * `llm_judge`
  * `mcq`
  * `mcq_logprob`
  * `stability_*` sweeps

This structure mirrors the local directory model and simplifies auditing.

---

## 5. Local Artifacts vs MLflow Artifacts

### 5.1 Local staging directories

Method outputs are staged locally under:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/artifacts_local/<exp_name>/
```

These directories usually contain method-level outputs intended for MLflow upload/logging (depending on notebook runner/helper behavior), such as:

* `metrics.json`
* summaries
* predictions exports
* audit summaries
* diagnostics

### 5.2 MLflow artifact copies

The MLflow helper layer typically mirrors or logs equivalent artifacts into the corresponding MLflow child run.

### 5.3 Operational implication

When validating results, the following should be cross-checked:

* local staged artifact (`artifacts_local/<exp_name>/...`)
* corresponding MLflow child-run artifact
* MLflow run IDs in `mlflow_ids.json`

---

## 6. Resume Behavior and MLflow Continuity

The script persists MLflow identifiers in session scope and reloads them when resuming a session.

Operationally, this enables:

* reuse of the same parent run after interruption
* reuse of method child runs (rather than creating duplicates)
* consistent artifact lineage across multiple executions of the same session

This behavior is central to long-running API evaluations and should be considered part of the project’s reproducibility guarantees.

---

## 7. How to Access the MLflow Web Interface (Local)

### 7.1 Prerequisite

MLflow must be installed in the active environment:

```bash
pip install mlflow
```

### 7.2 Start the local MLflow UI (default local store)

If the local tracking store is the default `mlruns/` directory, start the UI with:

```bash
mlflow ui
```

By default, this usually starts a web server on:

```text
http://127.0.0.1:5000
```

### 7.3 Start the local MLflow UI with explicit backend store path

If the tracking store is located in a specific directory, run:

```bash
mlflow ui --backend-store-uri /path/to/mlruns
```

Example (current directory):

```bash
mlflow ui --backend-store-uri ./mlruns
```

### 7.4 Bind host/port explicitly (recommended)

For explicit local binding and a custom port:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

### 7.5 Expose on local network (optional)

To make the UI reachable from another machine on the same network:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Then access from a browser using the machine IP address:

```text
http://<machine-ip>:5000
```

> Security note: binding to `0.0.0.0` exposes the UI on the network and should only be used in trusted environments.

---

## 8. How to Locate the Correct Run in the MLflow UI

### 8.1 Primary lookup method (session mapping)

Use the session file:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/mlflow_ids.json
```

This provides the authoritative mapping to:

* the parent run
* each child run by method name

### 8.2 Secondary lookup metadata (when manual search is required)

If manual lookup in the MLflow UI is required, use combinations of:

* `RUN_KEY`
* session fingerprint
* target model name
* judge model name (if applicable)
* method name (`llm_judge`, `mcq`, `mcq_logprob`, `stability_*`)
* execution timestamp

---

## 9. What to Inspect in MLflow (Method-Level Validation)

For each method child run, inspection should include:

* logged artifacts (e.g., `metrics.json`, summaries)
* method parameters (models, temperatures, seeds, delimiters where applicable)
* any logged diagnostic JSON/JSONL
* run status and completion consistency with local `_DONE` markers (filesystem check)

Recommended cross-checks:

1. Local `artifacts_local/<exp_name>/metrics.json`
2. MLflow child-run artifacts
3. Session `mlflow_ids.json` mapping

---

## 10. Hosted MLflow / Remote Tracking (General Pattern)

If a remote MLflow server is used, the tracking URI is typically configured via environment or helper code (outside the main script).

Common pattern (environment-based configuration):

```bash
export MLFLOW_TRACKING_URI="http://<mlflow-server>:5000"
```

Then run the evaluation normally. Logged runs will appear in the configured MLflow tracking server.

Local run mapping still remains available via:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/mlflow_ids.json
```

---

## 11. Troubleshooting: Local MLflow Web Page Not Opening

### 11.1 MLflow command not found

Cause:

* MLflow not installed in the active environment

Resolution:

```bash
pip install mlflow
```

### 11.2 Port 5000 already in use

Cause:

* another process is already listening on port `5000`

Resolution:

* start MLflow on another port, e.g. `5001`

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001
```

Then open:

```text
http://127.0.0.1:5001
```

### 11.3 Empty UI / no runs visible

Possible causes:

* wrong backend store path
* tracking URI mismatch between evaluation run and UI process
* runs logged to a remote MLflow server, while local UI is pointed to an empty local store

Validation sequence:

1. Confirm `mlflow_ids.json` exists in the session directory
2. Confirm local `artifacts_local/` exists (indicates method outputs were produced)
3. Check MLflow tracking URI configuration used during the evaluation
4. Start the UI against the correct backend store (if local)

### 11.4 UI reachable but artifacts missing

Possible causes:

* artifact logging failed
* artifact logging path differs from expected local staging
* MLflow helper configuration issue

Validation sequence:

1. Inspect `artifacts_local/<exp_name>/`
2. Inspect MLflow child run logs/artifacts
3. Inspect helper-layer logging configuration in `w2c_mlflow_logging.py`

---

## 12. Operational Best Practices for MLflow in This Project

### 12.1 Preserve `mlflow_ids.json` with the session

This file is required for deterministic run mapping and resume-safe MLflow continuity.

### 12.2 Keep filesystem session artifacts and MLflow records together

MLflow should be treated as complementary to local session artifacts, not a replacement for checkpoints/audit logs.

### 12.3 Use parent/child run organization consistently

This improves navigability and aligns with the project’s method-level execution model.

### 12.4 Cross-check local and MLflow artifacts before reporting results

Reported metrics should be validated against both:

* local staged artifacts
* MLflow child-run artifacts

---

## 13. MLflow Documentation Blurb (Project-Ready)

> **MLflow is used as the experiment tracking layer for this evaluation suite, with session-resumable parent/child run mapping persisted in `mlflow_ids.json`.** This enables stable linkage between local session artifacts (`runs/<RUN_KEY>/sessions/<FINGERPRINT>/...`) and MLflow UI runs, supporting reproducible experiment tracking, artifact inspection, and interruption-safe continuation of long API-based evaluations.

---

## 14. Quick Reference (MLflow-Only)

### Start local MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

### Open local web page

```text
http://127.0.0.1:5000
```

### Session-to-MLflow mapping file

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/mlflow_ids.json
```

### Local staged artifacts

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/artifacts_local/<exp_name>/
```
