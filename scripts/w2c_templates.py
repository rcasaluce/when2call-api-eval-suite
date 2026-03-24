# w2c_templates.py
# with harmony prompt added for gpt-oss: (13/01/26)
# WITH API SEED and no reasoning gpt for mcq string-based 11 Jannuary
# 8 January added audit fallbacks in the checkpoints folders
# with checkpoints and stability with new metrics
## SISTEMATO LIMITE MCQ
# version 4 30 dicembre

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import w2c_prompts as prompts

# =============================================================================
# TEMPLATE CONSTANTS / GLOBALS
# =============================================================================

BEHAVIOR_LABELS = ["direct", "tool_call", "request_for_info", "cannot_answer"]

TARGET_DELIMITER = " "  # lm-eval / NVIDIA delimiter

PRINT_TEMPLATE_FAMILY_ONCE = True
_PRINTED_TEMPLATE_FAMILY = False


# =============================================================================
# DELIMITER RESOLUTION (MODEL-AWARE + OPTIONAL FORCE VIA CODE)
# =============================================================================

# Ordered rules: first match wins. Keys are lowercase substrings checked in the model name.
# Values are the delimiter to prepend before the candidate choice when scoring logprobs.
MODEL_DELIMITER_RULES: List[Tuple[str, str]] = [
    ("qwen3-coder", ""),
    ("qwen3", ""),
    ("qwen-coder", ""),
    ("llama-3.3", ""),
    ("llama", ""),
    ("minimax", " "),
    ("gpt-oss", " "),
    ("mistral", " "),
]


def _normalize_forced_delimiter(forced: Optional[str]) -> Optional[str]:
    """
    Normalize a user-provided delimiter override.

    Accepted inputs:
      - None: no override
      - ""  : empty delimiter
      - " " : single space
      - "\\n" or "newline": newline delimiter
      - Any other string is returned as-is (allowed, but should be used intentionally)

    This function does not read environment variables. It only normalizes the value passed in code.
    """
    if forced is None:
        return None

    # Keep exact intent if caller explicitly provides empty string.
    if forced == "":
        return ""

    v = forced.strip()
    if v.lower() in {"\\n", "newline"}:
        return "\n"
    return forced


def resolve_target_delimiter(model_name: str, forced: Optional[str] = None) -> str:
    """
    Resolve the scoring delimiter for a given model.

    Precedence:
      1) 'forced' override provided by the caller (code-level override).
      2) MODEL_DELIMITER_RULES pattern match against model_name (case-insensitive).
      3) Fallback to TARGET_DELIMITER.

    Returns:
      A string delimiter to prepend before the MCQ candidate choice when building the scored suffix.
    """
    forced_norm = _normalize_forced_delimiter(forced)
    if forced_norm is not None:
        logging.info("[DELIM] Forced delimiter override -> using %r", forced_norm)
        return forced_norm

    m = (model_name or "").lower()
    for pattern, delim in MODEL_DELIMITER_RULES:
        if pattern in m:
            logging.info("[DELIM] Auto delimiter: model=%s matched %r -> %r", model_name, pattern, delim)
            return delim

    logging.warning(
        "[DELIM] No delimiter rule matched for model=%s. Falling back to TARGET_DELIMITER=%r",
        model_name,
        TARGET_DELIMITER,
    )
    return TARGET_DELIMITER


# =============================================================================
# HARMONY PROMPT BUILDERS (gpt-oss)
# =============================================================================

def build_prompts_harmony_chat(
    item: "When2CallExample",
    *,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, str, str, List[str], int, List[str]]:
    """
    Build Harmony chat prompts: (system_message, developer_message, user_message, choices, gold_idx, answer_names).

    Note:
      - We accept reasoning_effort as parameter to avoid coupling to the caller's globals.
      - Caller can pass reasoning_effort=(REASONING_EFFORT or "medium").
    """
    choices, target_index, answer_names = get_choices_and_index(item)
    system_msg = prompts.build_harmony_system_message(reasoning_effort=(reasoning_effort or "medium"))
    developer_msg = prompts.build_harmony_developer_message(item.tools_raw)
    user_msg = item.question
    return system_msg, developer_msg, user_msg, choices, target_index, answer_names


def build_mcq_prompt_harmony(
    item: "When2CallExample",
    *,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, List[str], int, List[str]]:
    """
    Build a single-string Harmony prompt suitable for /completions echo scoring.

    We end with an OPEN assistant final message header so the completion begins immediately.
    """
    choices, target_index, answer_names = get_choices_and_index(item)

    sys_text = prompts.build_harmony_system_message(reasoning_effort=(reasoning_effort or "medium"))
    dev_text = prompts.build_harmony_developer_message(item.tools_raw)
    user_text = item.question

    prompt = (
        "<|start|>system<|message|>" + sys_text + "<|end|>"
        "<|start|>developer<|message|>" + dev_text + "<|end|>"
        "<|start|>user<|message|>" + user_text + "<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    return prompt, choices, target_index, answer_names


def _merge_developer_into_system_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Some gateways do not accept role='developer'. If that happens, we merge developer content into the system message.
    """
    system_parts: List[str] = []
    user_messages: List[Dict[str, str]] = []
    other_messages: List[Dict[str, str]] = []

    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "developer":
            # Merge developer instruction into system
            system_parts.append("\n\n" + content)
        elif role == "user":
            user_messages.append(m)
        else:
            other_messages.append(m)

    merged_system = "\n".join([p for p in system_parts if p])
    out: List[Dict[str, str]] = []
    if merged_system.strip():
        out.append({"role": "system", "content": merged_system})
    out.extend(other_messages)
    out.extend(user_messages)
    return out


# =============================================================================
# TOOL / PROMPT UTILS
# =============================================================================

def default_format_tools(tools: List[Any]) -> str:
    """
    Format tool specs into the <tool>...</tool> blocks expected by eval templates.
    Input items may be dicts or raw JSON strings; output is a single text block.
    """
    return "\n\n".join(f"<tool>{_tool_to_json_str(t)}</tool>" for t in tools)


def _tool_to_json_str(t: Any) -> str:
    """
    Convert a tool spec (dict/str/other) into a JSON string for prompt inclusion.
    Falls back to str(t) if JSON serialization is not possible.
    """
    if isinstance(t, str):
        return t
    try:
        return json.dumps(t, ensure_ascii=False)
    except Exception:
        return str(t)


def _strip_trailing_spaces_like_lmeval(prompt: str) -> str:
    """
    Match lm-eval behavior by stripping only trailing spaces from the prompt.
    This reduces prompt/choice boundary discrepancies during logprob scoring.
    """
    return prompt.rstrip(" ")


def get_choices_and_index(item: "When2CallExample") -> Tuple[List[str], int, List[str]]:
    """
    Extract MCQ choices and the gold index from a When2CallExample.
    Returns (choice_texts, gold_index, answer_key_order) for consistent evaluation.
    """
    answer_names = list(item.answers.keys())
    choices = [item.answers[name] for name in answer_names]
    correct_answer_index = answer_names.index(item.correct_answer)
    return choices, correct_answer_index, answer_names


# =============================================================================
# NVIDIA PROMPT BUILDERS
# =============================================================================

def build_prompts_nvidia(item: "When2CallExample") -> Tuple[str, str, List[str], int, List[str]]:
    """
    Build (system_prompt, user_prompt) using the NVIDIA-style tool+question template.
    Also returns MCQ choices, gold index, and the ordered answer keys for logging.
    """
    choices, target_index, answer_names = get_choices_and_index(item)
    tool_string = default_format_tools(item.tools_raw)

    system_prompt = prompts.NVIDIA_SYSTEM_PROMPT_TEMPLATE.format(
        default_system_prompt=prompts.DEFAULT_SYSTEM_PROMPT,
        tool_use_instructions=prompts.TOOL_USE_INSTRUCTIONS,
    )
    user_prompt = prompts.NVIDIA_USER_PROMPT_TEMPLATE.format(
        tool_string=tool_string,
        question=item.question,
    )

    return system_prompt, user_prompt, choices, target_index, answer_names


def build_mcq_prompt_nvidia(item: "When2CallExample") -> Tuple[str, List[str], int, List[str]]:
    """
    Build a single combined MCQ prompt (system + tool list + question) for scoring.
    Returns (prompt, choices, gold_index, answer_keys) for logprob-based selection.
    """
    choices, target_index, answer_names = get_choices_and_index(item)
    tool_string = default_format_tools(item.tools_raw)

    prompt = prompts.NVIDIA_MCQ_PROMPT_TEMPLATE.format(
        default_system_prompt=prompts.DEFAULT_SYSTEM_PROMPT,
        tool_use_instructions=prompts.TOOL_USE_INSTRUCTIONS,
        tool_string=tool_string,
        question=item.question,
    )
    return prompt, choices, target_index, answer_names



def build_judge_prompt(example: When2CallExample, model_response: ModelResponse) -> str:
    """
    Construct the judge prompt containing tools, user question, and the model response.
    Returns a single text prompt to be sent as the judge model user message.
    """
    try:
        tools_parsed = [json.loads(t) for t in example.tools_raw]
    except Exception:
        tools_parsed = example.tools_raw
    tools_str = json.dumps(tools_parsed, ensure_ascii=False)
    return prompts.JUDGE_PROMPT.format(tools_str, example.question, model_response.raw_text)


def parse_judge_json(judge_raw: str) -> Tuple[str, Optional[str]]:
    """
    Parse judge output JSON and normalize the label into BEHAVIOR_LABELS.
    Raises if JSON is invalid or the classification field is missing/unknown.
    """
    text = judge_raw.strip()
    obj = json.loads(text)
    raw_label = obj.get("classification", obj.get("label"))
    if raw_label is None:
        raise ValueError("Missing 'classification' (or 'label') field in judge JSON.")

    label = str(raw_label).strip().lower()
    if label not in templates.BEHAVIOR_LABELS:
        raise ValueError(f"Unexpected classification '{label}'.")

    return label, None


# =============================================================================
# LLaMA / Qwen MCQ BUILDERS
# =============================================================================

def _format_tool_call_llama(tool_call_answer: Any) -> str:
    """
    Normalize a tool-call candidate into LLaMA function-call syntax: [name(arg=...)].
    Accepts dict or JSON string; returns "[]" when shape/JSON is invalid.
    """
    if isinstance(tool_call_answer, str):
        try:
            tool_call = json.loads(tool_call_answer)
        except Exception:
            logging.warning("[BAD TOOL_CALL JSON] tool_call_answer_repr=%r", tool_call_answer)
            return "[]"
    elif isinstance(tool_call_answer, dict):
        tool_call = tool_call_answer
    else:
        logging.warning("[BAD TOOL_CALL TYPE] type=%s repr=%r", type(tool_call_answer), tool_call_answer)
        return "[]"

    name = tool_call.get("name")
    args = tool_call.get("arguments", {})

    if not name or not isinstance(args, dict):
        logging.warning("[BAD TOOL_CALL SHAPE] name=%r args_type=%s tool_call_repr=%r", name, type(args), tool_call)
        return "[]"

    parts = []
    for arg, v in args.items():
        if isinstance(v, str):
            parts.append(f'{arg}="{v}"')
        else:
            parts.append(f"{arg}={v}")
    return f"[{name}({', '.join(parts)})]"


def build_llama3_2_mcq_item(item: "When2CallExample") -> Tuple[str, List[str], int, List[str]]:
    answer_names = list(item.answers.keys())
    choices: List[str] = []

    for name in answer_names:
        val = item.answers[name]
        if "tool" in name:
            choice = _format_tool_call_llama(val)
        else:
            choice = val if isinstance(val, str) else str(val)

        if not choice or not choice.strip():
            logging.warning(
                "[EMPTY CHOICE] uuid=%s answer_key=%s correct_answer=%s choice_repr=%r val_repr=%r",
                getattr(item, "uuid", None),
                name,
                getattr(item, "correct_answer", None),
                choice,
                val,
            )
            choice = "[]" if "tool" in name else " "

        choices.append(choice)

    target_index = answer_names.index(item.correct_answer)

    tools_json: List[Any] = []
    for t in item.tools_raw:
        if isinstance(t, str):
            # mantengo il comportamento "strict" originale per Llama template
            tools_json.append(json.loads(t))
        else:
            tools_json.append(t)

    tools_str = json.dumps(tools_json)

    prompt = prompts.LLAMA3_2_MCQ_PROMPT_TEMPLATE.format(
        llama_system_prompt=prompts.LLAMA32_SYSTEM_PROMPT,
        tools_str=tools_str,
        question=item.question,
    )

    return prompt, choices, target_index, answer_names


def build_qwen2_5_mcq_item(item: "When2CallExample") -> Tuple[str, List[str], int, List[str]]:
    answer_names = list(item.answers.keys())
    choices = [item.answers[name] for name in answer_names]
    target_index = answer_names.index(item.correct_answer)

    tools_str = "\n".join(_tool_to_json_str(t) for t in item.tools_raw).strip()

    prompt = prompts.QWEN2_5_MCQ_PROMPT_TEMPLATE.format(
        tools_str=tools_str,
        question=item.question,
    )
    return prompt, choices, target_index, answer_names


# =============================================================================
# TEMPLATE FAMILY ROUTING
# =============================================================================

def infer_when2call_template_family(model_name: str) -> str:
    """
    Decide which prompt template family to use (llama3_2, qwen2_5, harmony_gpt_oss, default).

    Priority:
      1) ENV override WHEN2CALL_TEMPLATE_FAMILY
      2) Model-name heuristic
    """
    forced = (os.getenv("WHEN2CALL_TEMPLATE_FAMILY") or "").strip().lower()
    if forced in {"llama3_2", "qwen2_5", "default", "harmony_gpt_oss"}:
        return forced

    m = (model_name or "").lower()
    if "gpt-oss" in m:
        return "harmony_gpt_oss"
    if "qwen" in m:
        return "qwen2_5"
    if "llama" in m:
        return "llama3_2"
    return "default"


def reset_template_family_print_once_flag() -> None:
    """
    Reset one-time template-family logging guard.
    Useful in notebooks if you want to re-print the inferred family after re-import/reload.
    """
    global _PRINTED_TEMPLATE_FAMILY
    _PRINTED_TEMPLATE_FAMILY = False


def build_when2call_mcq_item_for_model(
    item: "When2CallExample",
    target_model: str,
    *,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, List[str], int, List[str]]:
    """
    Build the model-specific MCQ prompt and candidate choices for When2Call.
    """
    fam = infer_when2call_template_family(target_model)

    global _PRINTED_TEMPLATE_FAMILY
    if PRINT_TEMPLATE_FAMILY_ONCE and not _PRINTED_TEMPLATE_FAMILY:
        forced = (os.getenv("WHEN2CALL_TEMPLATE_FAMILY") or "").strip().lower()
        logging.info(
            "[TEMPLATE] target_model=%s -> family=%s (forced=%s)",
            target_model,
            fam,
            forced if forced else "none",
        )
        _PRINTED_TEMPLATE_FAMILY = True

    if fam == "harmony_gpt_oss":
        return build_mcq_prompt_harmony(item, reasoning_effort=reasoning_effort)
    if fam == "qwen2_5":
        return build_qwen2_5_mcq_item(item)
    if fam == "llama3_2":
        return build_llama3_2_mcq_item(item)
    return build_mcq_prompt_nvidia(item)


def build_when2call_chat_messages_for_model(
    example: "When2CallExample",
    model_name: str,
    *,
    reasoning_effort: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[str], int, List[str]]:
    """
    Build chat messages for the target model, respecting the selected template family.

    Returns:
      (messages, choices, gold_idx, answer_names)
    """
    fam = infer_when2call_template_family(model_name)

    if fam == "harmony_gpt_oss":
        sys_msg, dev_msg, user_msg, choices, gold_idx, answer_names = build_prompts_harmony_chat(
            example, reasoning_effort=reasoning_effort
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "developer", "content": dev_msg},
            {"role": "user", "content": user_msg},
        ]
        return messages, choices, gold_idx, answer_names

    # Default (existing NVIDIA-style system+user with <tool> blocks)
    system_prompt, user_prompt, choices, gold_idx, answer_names = build_prompts_nvidia(example)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, choices, gold_idx, answer_names
