"""
Structured response parser, JSON extractor, and schema validation engine.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from pydantic import ValidationError

from exocortex.schema import AgentDecision, StructuredStep
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import ToolRegistry


def extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a valid JSON object dictionary from model output text.
    
    Handles markdown code blocks (```json ... ```), surrounding text, and trailing commas.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # 1. Try parsing directly
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Extract from markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            data = json.loads(block_content)
            if isinstance(data, dict):
                return data
        except Exception:
            cleaned = block_content  # Fall through to repair logic on block content

    # 3. Find outer braces {...}
    brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
    if brace_match:
        candidate = brace_match.group(1).strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 4. Attempt heuristic repairs on candidate
        repaired = repair_malformed_json(candidate)
        if repaired is not None:
            return repaired

    # 5. Global repair attempt
    return repair_malformed_json(cleaned)


def repair_malformed_json(text: str) -> Optional[Dict[str, Any]]:
    """Apply heuristic repairs to malformed JSON strings."""
    # Find start brace
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    content = text[start_idx:]

    # Remove trailing commas before closing braces/brackets
    content = re.sub(r",\s*([}\]])", r"\1", content)

    # Replace single quotes with double quotes for keys/values
    # (avoid replacing apostrophes inside words by checking for quote preceded by whitespace or brace)
    content = re.sub(r"([{,\s])'([^']+)'(\s*[:,\}\]])", r'\1"\2"\3', content)

    # If missing closing brace, append
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces > close_braces:
        content += "}" * (open_braces - close_braces)

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def parse_and_validate_decision(
    raw_text: str,
    tool_registry: ToolRegistry,
) -> Tuple[Optional[AgentDecision], Optional[str]]:
    """
    Parse raw SLM output and strictly validate against schema and tool registry.
    
    Returns:
        (decision, error_message): decision is None if validation failed.
    """
    json_data = extract_json_from_text(raw_text)
    if json_data is None:
        return None, "Failed to extract valid JSON from SLM response."

    # Validate against Pydantic schema
    try:
        decision = AgentDecision.model_validate(json_data)
    except ValidationError as ve:
        return None, f"Schema validation error: {ve.errors()}"
    except Exception as e:
        return None, f"Invalid decision structure: {str(e)}"

    # Validate tool steps against ToolRegistry
    for step in decision.steps:
        tool = tool_registry.get_tool(step.tool)
        if not tool:
            return (
                None,
                f"Unknown tool requested: '{step.tool}'. Tool is not registered in ToolRegistry.",
            )

        # Validate arguments against tool parameter definition
        valid_args, arg_err = tool.validate_arguments(step.arguments)
        if not valid_args:
            return (
                None,
                f"Invalid arguments for tool '{step.tool}': {arg_err}",
            )

        # Ensure requires_confirmation is set if tool requires confirmation
        if tool.permission_level in (
            PermissionLevel.CONFIRMATION_REQUIRED,
            PermissionLevel.RESTRICTED,
        ):
            decision.requires_confirmation = True

    return decision, None
