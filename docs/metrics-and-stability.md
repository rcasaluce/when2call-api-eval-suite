# Metrics & Stability Guide

This document collects the formal metric definitions and stability analysis formulas used by this repository.

It is intended as a companion to the main README, which describes setup, configuration, providers, and execution workflow.

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
````

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


