# w2c_prompts.py
"""
Prompt library for When2Call evaluation.

Changelog / notes:
- 2026-01-13: harmony prompt added for gpt-oss
- 2026-01-11: API seed + no reasoning for mcq string-based
- 2026-01-08: audit fallbacks in checkpoints folders
- 2025-12-30: version 4
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional


# =============================================================================
# PROMPTS (NVIDIA-style)
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. 
You have access to the following tools described in <tool></tool> which you can use to answer the user's questions.
Only use a tool if it directly answers the user's question.
"""

TOOL_USE_INSTRUCTIONS = """To use a tool, return JSON in the following format:
{"name": "tool_name", "arguments": {"argument1": "value1", "argument2": "value2", ...}}
"""

NVIDIA_SYSTEM_PROMPT_TEMPLATE = "{default_system_prompt}\n{tool_use_instructions}"
NVIDIA_USER_PROMPT_TEMPLATE = "{tool_string}\n\n{question}"
NVIDIA_MCQ_PROMPT_TEMPLATE = "{default_system_prompt}\n{tool_use_instructions}\n\n{tool_string}\n\n{question}"


JUDGE_PROMPT = """You are an expert at classifying responses from AI models.

Your task is to classify AI model's response into one of the following four categories:
(1) direct: The AI model responded to the User's questions based on it's existing knowledge, without requesting any additional information or using external tools.
(2) tool_call: The AI model decided to use a tool from the provided one's to help answer the question.
(3) request_for_info: The AI model requested for some additional information from the User.
(4) cannot_answer: The AI model refused to answer the User's questions by acknowledging the lack of required capabilities.

*You should not judge whether the AI model's response is accurate or not. Only provide the classification of the response into these four categories: [direct, tool_call, request_for_info, cannot_answer]*

- The tools available to the AI model are given in <AVAILABLE_TOOLS> </AVAILABLE_TOOLS>
- The User's question is provided in <USER_QUESTION> </USER_QUESTION>
- The AI model's response is provided in <AI_MODEL_RESPONSE> </AI_MODEL_RESPONSE> which may or may not invlove a tool call

<AVAILABLE_TOOLS>
{}
</AVAILABLE_TOOLS>

<USER_QUESTION>
{}
</USER_QUESTION>

<AI_MODEL_RESPONSE>
{}
</AI_MODEL_RESPONSE>

Please provide the classification in the following json format by filling in the placeholders in < >:
{{"classification": "<one of `direct`, `tool_call`, `request_for_info`, `cannot_answer`>"}}

Respond only in the prescribed json format with the placeholders filled in."""


MCQ_CLASSIFIER_PROMPT = """
You are evaluating which behavior an assistant SHOULD use in a tool-using chat.

You are NOT answering the user. You only pick ONE of four candidate behaviors (index 0–3).

Each candidate (given in the user message) has:
- a behavior_type in {direct, tool_call, request_for_info, cannot_answer}
- a candidate_response text

Behavior types:
- direct: answer with general knowledge only. No tools, no follow-up questions, no pretending to execute real-world actions (e.g. modify orders, charge cards, prepare drinks).
- tool_call: call one of the AVAILABLE tools. Only correct if the tool exists in the tool list and the user already provided enough parameters for that tool.
- request_for_info: ask the user for missing or unclear information BEFORE answering or calling tools.
- cannot_answer: explicitly say that the assistant cannot do what is asked (no suitable tool or impossible/out-of-scope capability).

Decision rule:
1) From the system+user prompt and tools, infer what the IDEAL behavior_type should be.
2) Then look at the 4 candidates and pick the index whose behavior_type AND candidate_response best match that ideal behavior.
   Penalize candidates that:
   - use tools that are not in the AVAILABLE tools list, or
   - claim to have already executed external actions the assistant cannot really perform.

Answer with only one character: 0, 1, 2, or 3.
Answer:
""".strip()


# =============================================================================
# PROMPTS (LLaMA / Qwen MCQ templates)
# =============================================================================

LLAMA32_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections. 
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] 
Here is a list of functions in JSON format that you can invoke. 
You SHOULD NOT include any other text in the response. """.strip()

LLAMA3_2_MCQ_PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{llama_system_prompt}\n"
    "{tools_str}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

QWEN2_5_MCQ_PROMPT_TEMPLATE = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


# =============================================================================
# LEGACY / COMMENTED-OUT PROMPTS (keep history without comments)
# =============================================================================

LEGACY_PROMPTS: Dict[str, str] = {
    "LLAMA32_SYSTEM_PROMPT_OLD": """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
Here is a list of functions in JSON format that you can invoke.
If no tool is needed, answer normally; if missing info, ask; if impossible, refuse.
""".strip(),
}


# =============================================================================
# HARMONY (gpt-oss) prompt family helpers
# =============================================================================

HARMONY_DEFAULT_KNOWLEDGE_CUTOFF = os.getenv("HARMONY_KNOWLEDGE_CUTOFF", "2024-06")
HARMONY_DEFAULT_CURRENT_DATE = os.getenv("HARMONY_CURRENT_DATE", "")

HARMONY_SYSTEM_MESSAGE_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: {knowledge_cutoff}\n"
    "Current date: {current_date}\n\n"
    "Reasoning: {reasoning_effort}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
    "Calls to these tools must go to the commentary channel: 'functions'."
)

HARMONY_DEVELOPER_INSTRUCTIONS = (
    "# Instructions\n\n"
    "You are a helpful AI assistant.\n"
    "You have access to function tools described in the Tools section below.\n"
    "Only call a tool if it directly helps answer the user's question.\n\n"
    "When calling a tool, output JSON in exactly this format:\n"
    '{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}\n\n'
    "Do not claim you executed external actions unless you are calling a tool.\n"
)


def _harmony_current_date_str() -> str:
    """
    Return a YYYY-MM-DD string used in the Harmony system message.
    """
    if HARMONY_DEFAULT_CURRENT_DATE.strip():
        return HARMONY_DEFAULT_CURRENT_DATE.strip()
    return date.today().isoformat()


def build_harmony_system_message(*, reasoning_effort: str) -> str:
    """
    Build a Harmony-compliant system message string.
    """
    return HARMONY_SYSTEM_MESSAGE_TEMPLATE.format(
        knowledge_cutoff=HARMONY_DEFAULT_KNOWLEDGE_CUTOFF,
        current_date=_harmony_current_date_str(),
        reasoning_effort=(reasoning_effort or "medium"),
    )


def _jsonschema_type_to_ts(t: Any) -> str:
    """
    Best-effort JSON-schema type to TS-like type mapping for Harmony tool defs.
    We keep it intentionally conservative to reduce tool-call errors.
    """
    if t == "string":
        return "string"
    if t in {"integer", "number"}:
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "array":
        return "any[]"
    if t == "object":
        return "any"
    return "any"


def _safe_ident(name: str) -> str:
    """
    Make a tool/function name safe for TS-like type identifiers.
    """
    name = (name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name:
        return "tool"
    if name[0].isdigit():
        name = f"tool_{name}"
    return name


def _tool_json_to_harmony_ts(tool_obj: Dict[str, Any]) -> str:
    """
    Convert a single tool spec (OpenAI-style JSON schema) to a Harmony TS-like tool type.
    Harmony guidance: wrap tools in namespace functions and define as type {name} = (_: {...}) => any;
    """
    name = _safe_ident(str(tool_obj.get("name", "tool")))
    desc = str(tool_obj.get("description", "")).strip()

    params = tool_obj.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    if not isinstance(required, list):
        required = []

    lines: List[str] = []
    if desc:
        lines.append(f"// {desc}")
    lines.append(f"type {name} = (_: {{")

    for prop_name, prop_spec in props.items():
        if not isinstance(prop_spec, dict):
            prop_spec = {}

        pdesc = str(prop_spec.get("description", "")).strip()
        ptype = _jsonschema_type_to_ts(prop_spec.get("type"))
        is_required = prop_name in required
        optional = "" if is_required else "?"

        enum_vals = prop_spec.get("enum")
        default_val = prop_spec.get("default", None)

        if pdesc:
            lines.append(f"  // {pdesc}")
        if isinstance(enum_vals, list) and enum_vals:
            lines.append(f"  // Allowed values: {enum_vals}")
        if default_val is not None:
            lines.append(f"  // Default: {default_val!r}")

        safe_prop = re.sub(r"[^A-Za-z0-9_]", "_", str(prop_name))
        if safe_prop != prop_name:
            lines.append(f"  // Original field name: {prop_name!r}")

        lines.append(f"  {safe_prop}{optional}: {ptype},")

    lines.append("}) => any;")
    return "\n".join(lines)


def build_harmony_tools_namespace(tools_raw: List[Any]) -> str:

    """
    Convert tools_raw (list of dicts or JSON strings) into a Harmony 'namespace functions' block.
    """
    tool_dicts: List[Dict[str, Any]] = []
    for t in tools_raw or []:
        if isinstance(t, dict):
            tool_dicts.append(t)
        elif isinstance(t, str):
            try:
                tool_dicts.append(json.loads(t))
            except Exception:
                tool_dicts.append({"name": "unknown_tool", "description": f"Unparseable tool spec: {t[:120]}"})
        else:
            tool_dicts.append({"name": "unknown_tool", "description": f"Unsupported tool spec type: {type(t)}"})

    defs: List[str] = []
    for td in tool_dicts:
        try:
            defs.append(_tool_json_to_harmony_ts(td))
            defs.append("")
        except Exception:
            defs.append("// Failed to render tool; placeholder emitted.")
            defs.append("type unknown_tool = (_: any) => any;")
            defs.append("")

    return (
        "# Tools\n"
        "## functions\n\n"
        "namespace functions {\n\n"
        + "\n".join(defs).rstrip()
        + "\n\n}\n"
    )


def build_harmony_developer_message(tools_raw: List[Any]) -> str:
    """
    Developer message = instructions + tools.
    """
    return HARMONY_DEVELOPER_INSTRUCTIONS + "\n" + build_harmony_tools_namespace(tools_raw)
