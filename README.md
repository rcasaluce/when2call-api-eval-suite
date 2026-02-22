# When2Call API Evaluation Suite (LLM-as-Judge, MCQ, MCQ Log-Probability)

This repository reproduces and extends the evaluation protocols from **“When2Call: When (not) to Call Tools”** (arXiv:2504.18851) in an **API-deployed** setting (OpenAI-compatible endpoints). It is designed to mirror the behavior-oriented evaluation flow used in the official When2Call codebase and in **lm-evaluation-harness**, while adding robust checkpointing, richer metrics (including hallucination rates), and stability analyses.

* When2Call paper: [https://arxiv.org/pdf/2504.18851](https://arxiv.org/pdf/2504.18851)
* Official dataset & reference implementation: [https://github.com/NVIDIA/When2Call/tree/main](https://github.com/NVIDIA/When2Call/tree/main)

## Documentation

For operational details, auditability, and experiment tracking:

* [Operations & Audit Guide](docs/operations-audit-guide.md) — run/session structure, checkpoints, audit logs, traceability, and reproducibility workflow
* [MLflow Guide](docs/mlflow-guide.md) — MLflow run organization, artifact mapping, and how to access the local MLflow web UI

---

## What this code does

The suite evaluates tool-use behavior across the four When2Call labels:

* `direct`
* `tool_call`
* `request_for_info`
* `cannot_answer`

It implements three complementary evaluation paradigms:

1. **LLM-as-Judge** (ex-post classification of a generated response)
2. **MCQ (string-based)** (ex-ante decision via index-only classification, no logprobs)
3. **MCQ (log-probability scoring)** (ex-ante decision via likelihood scoring, aligned with `lm-eval` MCQ scoring)

Additionally, it computes:

* Accuracy, macro-F1, macro-F1 excluding `direct`, per-class F1/support
* Confusion matrices
* Hallucination rates: tool hallucination, answer hallucination, parameter hallucination
* Stability metrics under repeated runs (Stability@k, MeanConsistency@k, entropy, flip-rate, stable-correct/wrong, etc.)

---

## Repository structure

```text

.
├── README.md                          # Main project README (overview, setup, usage + links to docs)
├── config.example.toml                # Versioned configuration template (commit this)
├── config.toml                        # Local runtime config (copy from config.example.toml; usually gitignored)
├── .env.example                       # Versioned environment template for secrets/optional overrides
├── .env                               # Local secrets (copy from .env.example; usually gitignored)
├── docs/
│   ├── operations-audit-guide.md      # Operations, traceability, checkpoints, audit logs, reproducibility
│   └── mlflow-guide.md                # MLflow usage, run mapping, artifacts, local web UI access
├── When2Call/                         # Cloned from NVIDIA When2Call (dataset lives here)
│   └── data/
│       └── test/
│           ├── when2call_test_mcq.jsonl
│           └── when2call_test_llm_judge.jsonl
├── scripts/
│   ├── main_script.py                 # Main evaluation script (pipelines, checkpoints, audit, metrics)
│   ├── w2c_config.py                  # Config loader (TOML + .env merge/overrides)
│   ├── w2c_notebook_runner.py         # Notebook-friendly runner (config, orchestration, MLflow runs)
│   ├── w2c_templates.py               # Prompt templates + model-family routing (Harmony/Qwen/Llama, etc.)
│   ├── w2c_prompts.py                 # Prompt strings (judge prompt, MCQ classifier prompt, etc.)
│   └── w2c_mlflow_logging.py          # MLflow + JSON/JSONL artifact logging utilities
└── workdir/                           # Default base folder (configurable)
    └── runs/
        └── <RUN_KEY>/
            └── sessions/
                └── <FINGERPRINT>/
                    ├── manifest.json      # Session identity + deterministic config fingerprint
                    ├── mlflow_ids.json    # MLflow parent/child run IDs (resume-safe mapping)
                    ├── checkpoints/       # Append-only JSONL checkpoints per pipeline
                    └── artifacts_local/   # Local artifacts staged/logged to MLflow
```

### Data requirement

Download (or clone) the official NVIDIA When2Call repository so that the test JSONL files are available locally, e.g.:

* `When2Call/data/test/when2call_test_mcq.jsonl`
* `When2Call/data/test/when2call_test_llm_judge.jsonl`

The evaluation code loads these JSONL files directly and maps them into typed dataclasses.

---

## Configuration model (new workflow)

This project now uses a **TOML-first configuration workflow**:

* **`config.example.toml`** → committed template (safe defaults / documentation)
* **`config.toml`** → your local runtime config (models, dataset path, pipelines, retry settings, etc.)
* **`.env.example`** → committed env template (tokens + optional overrides)
* **`.env`** → your local secrets (tokens)

### Important behavior in `main_script.py`

`main_script.py` already loads config for you:

* it computes the repo root
* it looks for **`config.toml`**
* if missing, it falls back to **`config.example.toml`**
* it calls `load_config(...)` from `w2c_config.py`

So you **do not need to call `load_config()` manually**.

### Resolution order (practical)

The exact merge rules depend on your `w2c_config.py`, but the intended workflow is:

1. `config.example.toml` (template / defaults)
2. `config.toml` (your local values)
3. `.env` (secrets and optional overrides, if supported by the loader)

Typical `.env` usage in this repo:

* `TOKEN_JRC`
* `TOKEN_GEMINI`
* optional `RUN_KEY` override (if your loader maps it into `CFG["run"]["run_key"]`)

### Why both `config.example.toml` and `config.toml`?

Because they serve different purposes:

* **`config.example.toml`** = shared, versioned template/documentation
* **`config.toml`** = local, editable runtime config for your machine/experiment

This keeps the repo reproducible **without** committing personal tokens, local paths, or one-off experiment settings.

---

## Key concepts and evaluation paradigms

### 1) LLM-as-Judge (ex-post classification)

**Goal:** Evaluate observed behavior in a free-form answer.

**Flow:**

1. The **target model** receives tools + question and generates a free-form response.
2. A separate **judge model** reads:

   * the tool list,
   * the user question,
   * the target model response,
     and outputs **strict JSON** containing a behavior classification in $\mathcal{Y}$.

**Output format:**

```json
{"classification":"tool_call"}
```

**Robustness:** If the judge output cannot be parsed as valid JSON, the pipeline:

* retries with a rewrite prompt (repair pass),
* and if it still fails, falls back conservatively to `cannot_answer` (with audit logging).

This mirrors the structure of the official `run_openai_judge.py` workflow, adapted to an OpenAI-compatible gateway and hardened for real-world API responses.

---

### 2) MCQ (string-based classifier; logprob-free)

**Goal:** Evaluate “what the assistant should do” (ex-ante decision) using a lightweight MCQ interface.

**Flow:**

1. Build an MCQ meta-prompt that embeds:

   * the original system + tools + question prompt content,
   * the four candidate behaviors as indexed options `0..3`,
   * instructions to output only one digit (`0`, `1`, `2`, or `3`).
2. Call `/chat/completions`.
3. Parse the first digit `0..3` from the model’s output and map it back to the behavior label.

This is useful when:

* logprobs are not available from the API,
* or logprobs are unreliable for a specific model / gateway.

This protocol also acts as a fallback for logprob MCQ when all choice scores are `-inf`.

---

### 3) MCQ (log-probability scoring; LM-Eval-aligned)

**Goal:** Recreate standard multiple-choice likelihood scoring as done in **lm-evaluation-harness**, including Acc-Norm-style normalization.

For each example:

* Let the base prompt be $x$.
* Let the candidate choices be ${y_1,\dots,y_K}$ (here $K=4$).
* Concatenate with a fixed delimiter $d$:

$$
z_j = x ,\Vert, d ,\Vert, y_j.
$$

Call `/completions` with `echo=True` on $z_j$, extract token logprobs for the suffix region corresponding to $d \Vert y_j$, and compute the following scores.

#### Raw score (sum of log-probabilities)


```math
S_j^{\mathrm{raw}}
=\sum_{i \in I_j} \log p_\theta \bigl(t_i^{(j)} \mid t_{\lt i}^{(j)}\bigr),
\qquad
\hat{j}_{\mathrm{raw}}=\arg\max_j S_j^{\mathrm{raw}}.
```

#### Byte-normalized score (LM-Eval `acc_norm` analogue)

Let $B_j$ be the UTF-8 byte length of the suffix $(d \Vert y_j)$:

$$
B_j=\bigl\lvert \mathrm{UTF8}(d \Vert y_j)\bigr\rvert_{\text{bytes}}.
$$

$$
S_j^{\text{byte}}=\frac{S_j^{\text{raw}}}{B_j},
\qquad
\hat{j}_{\text{byte}}=\arg\max_j S_j^{\text{byte}}.
$$

This is the closest analogue to lm-eval Acc-Norm scoring.

#### Token-normalized score (additional normalization)

Let $N_j$ be the number of suffix tokens in the scored region:

$$
S_j^{\text{tok}}=\frac{S_j^{\text{raw}}}{N_j},
\qquad
\hat{j}_{\text{tok}}=\arg\max_j S_j^{\text{tok}}.
$$

Token normalization is tokenizer-dependent and is reported separately.

#### Failure handling

If all choices return `-inf` (or otherwise non-finite) scores for an example, the pipeline falls back to the MCQ string-based classifier and reuses that label for all logprob variants. All such events are recorded in an audit JSONL.

---

## Prompt templates and model families

Likelihood-based MCQ scoring is sensitive to prompt-template mismatches. This project supports template families (e.g., Llama-style, Qwen-style, Harmony for `gpt-oss`) and selects them automatically from the model identifier (with optional override).

In practice:

* **LLM-as-judge target generation** is intentionally kept consistent using a single “base” prompt family for comparability.
* **MCQ logprob** uses model-family-aware prompt construction to reduce spurious likelihood artifacts.

Implementation: `w2c_templates.py`

---

## Multi-provider support (JRC + Gemini, OpenAI-compatible)

The suite supports **multiple OpenAI-compatible providers** for `/chat/completions`, currently:

* **JRC gateway** (default)
* **Google Gemini OpenAI-compatible endpoint**

Provider selection is handled **automatically** at runtime based on the **model name**.

### How provider routing works (automatic by model name)

In `main_script.py`, the provider is selected by checking whether the model identifier starts with one of the prefixes in `CFG["providers"]["gemini"]["model_prefixes"]` (default: `["gemini-"]`).

Conceptually:

* if `model_name.startswith("gemini-")` → use **Gemini**
* otherwise → use **JRC**

This applies to all calls that go through the `raw_chat_completion(...)` wrapper (e.g.):

* **LLM-as-judge target generation**
* **LLM-as-judge judge calls**
* **MCQ string-based classifier**

### Examples

* `target_model = "gpt-oss-120b"` → routed to **JRC**
* `target_model = "llama-3.3-70b-instruct"` → routed to **JRC**
* `target_model = "gemini-2.5-pro"` → routed to **Gemini**
* `judge_model  = "gemini-2.0-flash"` → routed to **Gemini**

### Provider-specific credentials and base URLs (configured in `config.toml`)

#### JRC (default)

Configured under:

```toml
[providers.jrc]
base_url = "https://..."
token = ""
```

Typical token source (via `.env` + `w2c_config` loader): `TOKEN_JRC`

#### Gemini (OpenAI-compatible)

Configured under:

```toml
[providers.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
token = ""
model_prefixes = ["gemini-"]
```

Typical token source (via `.env` + `w2c_config` loader): `TOKEN_GEMINI`

If a Gemini model is selected but no Gemini token is configured, the script raises a runtime error.

### Important limitation: MCQ logprob scoring is currently JRC-only

The **MCQ log-probability pipeline** (`do_mcq_logprob = true`) uses `/completions` with echo + token logprobs, and in the current implementation that path is tied to the **JRC `BASE_URL`** (`CFG["providers"]["jrc"]["base_url"]`).

Therefore:

* **LLM-as-judge** and **MCQ string-based** can use **JRC or Gemini** (automatic routing by model name)
* **MCQ logprob** currently assumes **JRC-compatible `/completions` logprobs support**

If `target_model = "gemini-..."` and `do_mcq_logprob = true`, that pipeline may fail unless the `/completions` logprob path is extended for Gemini as well.

---

## Metrics

All paradigms produce a prediction $\hat y_i \in \mathcal{Y}$ for each example $i$, with gold label $y_i$.

### Classification metrics

* Accuracy
* Macro-F1 (averages per-class F1 equally)
* Macro-F1 (no direct): macro-F1 excluding `direct` (useful if `direct` is rare/absent)
* Confusion matrix over the 4 labels
* Per-class support and per-class F1

### Hallucination rates (as in When2Call Appendix E.1, with extensions)

Let $\mathrm{tools}_i$ be the set of available tools for example $i$, and let $\mathbf{1}[\cdot]$ be an indicator.

#### Tool hallucination rate (ToolHall)

```math
\mathrm{ToolHall}=
\frac{
\sum_{i=1}^{N}
\mathbf{1}\!\left[
y_i=\texttt{cannot\_answer}
\land
|\mathrm{tools}_i|=0
\land
\hat{y}_i=\texttt{tool\_call}
\right]
}{
\sum_{i=1}^{N}
\mathbf{1}\!\left[
y_i=\texttt{cannot\_answer}
\land
|\mathrm{tools}_i|=0
\right]
}.
```

* **numerator**: number of examples where:

  * gold label is `cannot_answer`
  * no tools are available (`len(tools_i) == 0`)
  * predicted label is `tool_call`
* **denominator**: number of examples where:

  * gold label is `cannot_answer`
  * no tools are available (`len(tools_i) == 0`)

So, ToolHall is the fraction of no-tool `cannot_answer` cases that are incorrectly predicted as `tool_call`.

Interpretation: probability of predicting a tool call when the gold label is `cannot_answer` and no tools are available.

#### Answer hallucination rate (AnswerHall)

$$
\mathrm{AnswerHall}=
\frac{
\sum_{i=1}^{N}
\mathbf{1}!\left[
\hat y_i=\texttt{direct}
\land
y_i\ne\texttt{direct}
\right]
}{
N
}.
$$



Interpretation: rate of predicting `direct` when gold is not `direct`.

This metric is an extension implemented in this evaluation suite and is **not** part of the original NVIDIA When2Call project scripts / reported metrics in the paper.

#### Parameter hallucination rate (ParamHall)



```math
\mathrm{ParamHall}=
\frac{
\sum_{i=1}^{N}
\mathbf{1}\!\left[
y_i=\texttt{request\_for\_info}
\land
\hat{y}_i=\texttt{tool\_call}
\right]
}{
\sum_{i=1}^{N}
\mathbf{1}\!\left[
y_i=\texttt{request\_for\_info}
\right]
}.
```
* **numerator**: number of examples where:

  * gold label is `request_for_info`
  * predicted label is `tool_call`
* **denominator**: number of examples where:

  * gold label is `request_for_info`

So, ParamHall is the fraction of `request_for_info` cases in which the model calls a tool instead of asking for the missing required information.

Interpretation: calling a tool instead of asking for missing required parameters.

This metric is an extension implemented in this evaluation suite and is **not** part of the original NVIDIA When2Call project scripts / reported metrics in the paper.

---

## Stability suite (optional but recommended)

Because API-deployed LLMs are stochastic, the suite includes a stability module that runs each evaluation method $k$ times per example and quantifies reproducibility.

Given predictions $(y_i^{(1)},\dots,y_i^{(k)})$, define:

* modal label $\tilde y_i$,
* modal multiplicity $m_i$.

### Stability@k

$$
\mathrm{Stability@}k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[m_i=k].
$$

### MeanConsistency@k

$$
\mathrm{MeanConsistency@}k=\frac{1}{N}\sum_{i=1}^{N}\frac{m_i}{k}.
$$

### Stable & correct / stable but wrong / modal correctness

$$
\mathrm{StableCorrectRate}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[m_i=k \wedge \tilde y_i=y_i^{\text{gold}}],
$$

$$
\mathrm{StableWrongRate}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[m_i=k \wedge \tilde y_i\neq y_i^{\text{gold}}],
$$

$$
\mathrm{ModeCorrectRate}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\tilde y_i=y_i^{\text{gold}}].
$$

### Entropy and flip rate

Let $p_i(y)$ be the empirical distribution across runs.

$$
H_i=-\sum_{y\in\mathcal{Y}} p_i(y)\log_2 p_i(y),
\qquad
H_i^{\mathrm{norm}}=\frac{H_i}{\log_2|\mathcal{Y}|}.
$$

Flip rate:

$$
\mathrm{FlipRate}*i=\frac{1}{k-1}\sum*{r=2}^{k}\mathbf{1}[y_i^{(r)}\neq y_i^{(r-1)}].
$$

### Mean accuracy across runs

$$
\mathrm{MeanAccAcrossRuns}=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{1}{k}\sum_{r=1}^{k}\mathbf{1}[y_i^{(r)}=y_i^{\text{gold}}]\right).
$$

---

## Checkpointing, resume, and audit logs

The suite is designed for long-running API jobs. It uses:

* append-only JSONL checkpoints per pipeline (UUID-keyed, “last write wins”),
* session-scoped run directories keyed by a deterministic config fingerprint,
* “DONE markers” to skip completed pipelines safely,
* audit JSONL logs for fallbacks and forced decisions (e.g., judge parsing failures, `-inf` logprob fallback).

This enables:

* safe resume after crashes or rate limits,
* reproducibility and traceability of forced behaviors.

---

## How to run (TOML + `.env` workflow)

### 1) Install dependencies

Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install requests python-dotenv mlflow
```

If your `w2c_config.py` uses TOML parsing on Python < 3.11, you may also need:

```bash
pip install tomli
```

---

### 2) Create local config files from the examples

From the repository root:

```bash
cp config.example.toml config.toml
cp .env.example .env
```

Then edit:

* **`config.toml`** → models, dataset path, pipeline toggles, retry settings, stability options
* **`.env`** → provider tokens (`TOKEN_JRC`, `TOKEN_GEMINI`) and optional overrides

> `main_script.py` already loads `config.toml` (fallback: `config.example.toml`) via `w2c_config.load_config(...)`.

---

### 3) Fill in `config.toml` (main runtime settings)

At minimum, check these sections:

* `[providers.jrc]` / `[providers.gemini]`
* `[models]`
* `[data]`
* `[pipelines]`
* `[stability]`
* `[run]`

Example (minimal JRC-only run):

```toml
[run]
workdir_base = "./workdir"
run_key = ""
api_seed = 42

[providers.jrc]
base_url = "https://api-gpt.jrc.ec.europa.eu/v1"
token = ""

[providers.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
token = ""
model_prefixes = ["gemini-"]

[http]
max_retries = 3
retry_sleep_seconds = 10.0
base_delay_seconds = 1.0
timeout_seconds = 60

[models]
target_model = "llama-3.3-70b-instruct"
judge_model = "gpt-oss-120b"
force_target_delimiter = ""
reasoning_models = ["gpt-oss-120b", "gpt-oss-20b"]
reasoning_effort = "low"

[data]
eval_data_path = "./When2Call/data/test/when2call_test_llm_judge.jsonl"
use_full_dataset = false
n_per_label = 50
subsample_seed = 42

[pipelines]
do_llm_judge = true
do_mcq = false
do_mcq_logprob = false
target_temperature = 0.0
judge_temperature = 0.0
mcq_temperature = 0.0
```

---

### 4) Fill in `.env` (tokens / secrets)

Example `.env`:

```dotenv
# Provider tokens
TOKEN_JRC=your_jrc_token_here
TOKEN_GEMINI=your_gemini_token_here

# Optional run key override (only if your w2c_config maps it into [run].run_key)
RUN_KEY=
```

Notes:

* You can leave `TOKEN_GEMINI` empty if you are not using Gemini models.
* If your `w2c_config.py` supports aliases (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`), you may use those instead — but `TOKEN_JRC` / `TOKEN_GEMINI` are the explicit names used in the current config examples.

---

### 5) Choose the dataset file in `config.toml`

Set `[data].eval_data_path` to the correct file for the run:

* MCQ evaluations:

  * `./When2Call/data/test/when2call_test_mcq.jsonl`
* LLM-as-judge evaluations:

  * `./When2Call/data/test/when2call_test_llm_judge.jsonl`

Because `main_script.py` reads a **single** `eval_data_path` per execution, the common pattern is to run separate experiments (separate `RUN_KEY`s or sessions) for MCQ vs LLM-as-judge datasets.

---

### 6) Select models and pipelines in `config.toml`

Typical examples:

#### LLM-as-judge only

```toml
[pipelines]
do_llm_judge = true
do_mcq = false
do_mcq_logprob = false
```

#### MCQ string-only

```toml
[pipelines]
do_llm_judge = false
do_mcq = true
do_mcq_logprob = false
```

#### MCQ logprob (JRC `/completions` support required)

```toml
[pipelines]
do_llm_judge = false
do_mcq = false
do_mcq_logprob = true
```

Notes:

* MCQ logprob requires `/completions` with echo + token logprobs support (current code path is JRC-oriented).
* MCQ string-based works on `/chat/completions`.
* `/chat/completions` calls are routed automatically to **JRC** or **Gemini** based on model name prefix (default Gemini prefix: `gemini-`).

---

### 7) Run the script

From the repository root:

```bash
python scripts/main_script.py
```

That script:

* loads config (`config.toml` → fallback `config.example.toml`)
* initializes providers, retry policy, models, pipeline flags
* resolves/creates `RUN_KEY`
* imports and calls the notebook runner internally:

```python
from w2c_notebook_runner import run_when2call_notebook
run_when2call_notebook(globals())
```

So **you do not need to launch the notebook runner manually** unless you explicitly want to.

---

### 8) `RUN_KEY` behavior (important)

`main_script.py` resolves the run key like this:

* if `CFG["run"]["run_key"]` is set (or overridden via `.env`, if your loader supports it) → use it
* otherwise it prompts you in the terminal:

  * press **Enter** to auto-generate a UUID-based run key
  * or type a custom run name

This controls the output folder:

```text
<workdir_base>/runs/<RUN_KEY>/
```

---

### 9) Resume behavior (session fingerprinting)

Inside each `RUN_KEY`, the code creates a **session** under:

```text
runs/<RUN_KEY>/sessions/<FINGERPRINT>/
```

The fingerprint is computed deterministically from a sanitized config payload. This means:

* same run key + same config fingerprint → **resume**
* same run key + changed config → **new session**
* pipeline checkpoints and artifacts remain separated and auditable

---

## Outputs

Per pipeline you will typically see:

* `checkpoints/<exp_name>/*.jsonl` with per-UUID predictions
* `checkpoints/<exp_name>/audit_fallbacks.jsonl` with structured fallback events
* summary `metrics.json` including accuracy/F1/confusion matrix/hallucinations
* stability checkpoints when enabled

When MLflow is enabled, artifacts are also logged to the MLflow run.

---

## Reproduction fidelity vs. official When2Call

This suite intentionally reproduces the core behavior setting from the NVIDIA When2Call repository while adapting it to an API setting:

* Tools are passed **in-text** (e.g., `<tool>...</tool>`) rather than as OpenAI-native structured tool objects.
* MCQ logprob scoring recreates the **LM-Eval** multiple-choice likelihood approach using `/completions echo=True`.
* The LLM-as-judge pipeline mirrors the “judge” approach used in the official scripts, but includes strict JSON parsing + repair retries + conservative fallback to ensure robustness in real API deployments.

It also extends the original analysis with:

* macro-F1, confusion matrices, hallucination rates
* stable resume/checkpointing and fallback auditing
* stability metrics under repeated sampling
* additional hallucination metrics (`AnswerHall`, `ParamHall`) beyond the original NVIDIA scripts/paper reporting

---

## References

* “When2Call: When (not) to Call Tools” (arXiv:2504.18851)
* Official repository and dataset: NVIDIA/When2Call
