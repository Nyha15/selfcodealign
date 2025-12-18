"""
Memory-aware tool calling orchestration (Letta-inspired).

This module is intentionally small and deterministic:
- Before any external tool call: load core memory and infer missing params
- After any tool response: persist stable signal back to core memory
- Learn workflows in a demo-safe way (no ML; just counters)

This does NOT implement MCP servers or expose MCP tools. It is a policy layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory.core import load_all_core_memory, memory_insert, memory_replace


ToolParams = Dict[str, Any]
ToolResult = Any


@dataclass(frozen=True)
class ToolCall:
    tool: str
    params: ToolParams


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def prepare_tool_call(tool: str, params: ToolParams) -> Tuple[ToolParams, Dict[str, Any]]:
    """
    Enforce the "Memory Read Policy":
    - Always load core memory immediately before the tool call
    - Use memory to infer missing parameters

    Returns (new_params, memory_snapshot).
    """
    mem = load_all_core_memory()
    cw = mem.get("current_work", {})
    lp = mem.get("learned_patterns", {})

    new_params = dict(params or {})

    # --- GitHub inference ---
    # Convention: PR number stored in current_work.id
    if tool.startswith("github_"):
        pr_number = _as_int(cw.get("id"))
        if pr_number is not None:
            # Most github MCP tools use `pullNumber` for PR-scoped operations.
            if "pullNumber" in new_params and new_params.get("pullNumber") is None:
                new_params["pullNumber"] = pr_number
            if "issue_number" in new_params and new_params.get("issue_number") is None:
                new_params["issue_number"] = pr_number

    # --- Linear inference ---
    if tool.startswith("linear_"):
        ticket = cw.get("related_linear_ticket")
        if ticket is not None:
            if "id" in new_params and new_params.get("id") is None:
                new_params["id"] = ticket
            if "issueId" in new_params and new_params.get("issueId") is None:
                new_params["issueId"] = ticket

    # --- Slack inference (simulated slack tool uses `channel`) ---
    if tool.startswith("slack_") or "slack" in tool:
        default_channel = (
            (lp.get("preferences") or {}).get("default_slack_channel")
            if isinstance(lp, dict)
            else None
        )
        if default_channel and "channel" in new_params and not new_params.get("channel"):
            new_params["channel"] = default_channel

    return new_params, mem


def _extract_github_pr_fields(result: ToolResult) -> Dict[str, Any]:
    """
    Best-effort extraction from GitHub MCP responses.
    We keep this tolerant because different MCP methods return different shapes.
    """
    if not isinstance(result, dict):
        return {}

    pr = result.get("pull_request") or result.get("pullRequest") or result
    if not isinstance(pr, dict):
        return {}

    # Common-ish fields across GitHub APIs
    pr_number = pr.get("number") or pr.get("pullNumber")
    title = pr.get("title")
    branch = pr.get("head", {}).get("ref") if isinstance(pr.get("head"), dict) else pr.get("headRefName")
    state = pr.get("state") or pr.get("status")
    merged = pr.get("merged") or pr.get("isMerged")

    out: Dict[str, Any] = {}
    if pr_number is not None:
        out["id"] = _as_int(pr_number) or pr_number
    if title is not None:
        out["title"] = _string(title)
    if branch is not None:
        out["branch"] = _string(branch)
    if state is not None:
        out["status"] = _string(state)
    if merged is True:
        out["status"] = "merged"
    return out


def _extract_linear_fields(result: ToolResult) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    # Linear MCP shapes vary; best effort.
    issue = result.get("issue") or result.get("data") or result
    if not isinstance(issue, dict):
        return {}

    out: Dict[str, Any] = {}
    if issue.get("id"):
        out["related_linear_ticket"] = issue.get("id")
    # Some APIs provide state name or type
    state = issue.get("state") or issue.get("status")
    if isinstance(state, dict) and state.get("name"):
        out["workflow_phase"] = state.get("name")
    elif isinstance(state, str):
        out["workflow_phase"] = state
    return out


def _extract_slack_fields(tool: str, params: ToolParams, result: ToolResult) -> Dict[str, Any]:
    # Our simulated slack tool returns a small JSON success response;
    # channel is best taken from params.
    out: Dict[str, Any] = {}
    if "channel" in (params or {}) and params.get("channel"):
        out["default_slack_channel"] = params.get("channel")
    return out


def _append_workflow_observation(trigger: str, action: str) -> None:
    """
    Record actions taken after a trigger into learned_patterns.workflows[0].observed_sequence.
    Deterministic: we only touch workflows[0], which is pre-seeded with trigger 'pr_merged'.
    """
    if trigger != "pr_merged":
        return
    memory_insert("learned_patterns", "workflows[0].observed_sequence", action)


def _bump_workflow_learning(trigger: str) -> None:
    """
    Deterministic workflow learning:
    - times_observed += 1
    - confidence = min(1.0, times_observed / 5.0)
    """
    if trigger != "pr_merged":
        return

    mem = load_all_core_memory()
    wf0 = ((mem.get("learned_patterns") or {}).get("workflows") or [{}])[0]
    times = wf0.get("times_observed") or 0
    times = int(times) + 1
    confidence = min(1.0, times / 5.0)

    memory_replace("learned_patterns", "workflows[0].times_observed", times)
    memory_replace("learned_patterns", "workflows[0].confidence", confidence)


def persist_after_tool_call(tool: str, params: ToolParams, result: ToolResult) -> None:
    """
    Enforce the "Memory Write Policy" after a successful tool call.

    Persist only stable, reusable info:
    - GitHub: PR number/title/branch/status -> current_work
    - Linear: ticket id -> current_work.related_linear_ticket; status -> workflow_state.phase
    - Slack: channel -> learned_patterns.preferences.default_slack_channel
    Also records action observations for workflow learning.
    """
    # Track "last_action" in workflow_state for demo clarity.
    memory_replace("workflow_state", "last_action", tool)

    if tool.startswith("github_"):
        pr = _extract_github_pr_fields(result)
        if "id" in pr:
            memory_replace("current_work", "id", pr["id"])
        if "title" in pr:
            memory_replace("current_work", "title", pr["title"])
        if "branch" in pr:
            memory_replace("current_work", "branch", pr["branch"])
        if "status" in pr:
            memory_replace("current_work", "status", pr["status"])
        _append_workflow_observation("pr_merged", "github")

    elif tool.startswith("linear_"):
        lin = _extract_linear_fields(result)
        if "related_linear_ticket" in lin:
            memory_replace("current_work", "related_linear_ticket", lin["related_linear_ticket"])
        if "workflow_phase" in lin:
            memory_replace("workflow_state", "phase", lin["workflow_phase"])
        _append_workflow_observation("pr_merged", "update_linear")

    elif tool.startswith("slack_") or "slack" in tool:
        slack = _extract_slack_fields(tool, params, result)
        if "default_slack_channel" in slack:
            memory_replace(
                "learned_patterns",
                "preferences.default_slack_channel",
                slack["default_slack_channel"],
            )
        _append_workflow_observation("pr_merged", "post_slack")


def propose_actions_for_trigger(trigger: str) -> List[ToolCall]:
    """
    Given a trigger (e.g. "pr_merged"), propose next actions from learned patterns.
    """
    mem = load_all_core_memory()
    workflows = (mem.get("learned_patterns") or {}).get("workflows") or []
    for wf in workflows:
        if isinstance(wf, dict) and wf.get("trigger") == trigger:
            seq = wf.get("observed_sequence") or []
            # Map learned action labels -> tool names (demo-friendly defaults).
            out: List[ToolCall] = []
            for a in seq:
                if a == "update_linear":
                    out.append(ToolCall(tool="linear_update_issue", params={"id": None}))
                elif a == "post_slack":
                    out.append(ToolCall(tool="slack_send_slack_update", params={"channel": None, "message": ""}))
            return out
    return []


def record_trigger_and_complete_sequence(trigger: str, executed_tools: List[str]) -> None:
    """
    Called when a trigger occurs and the agent subsequently executes a sequence.
    Deterministic learning: bump counters for the trigger and record the sequence.
    """
    if trigger != "pr_merged":
        return

    # Replace observed_sequence with the canonical action labels (not raw tool names)
    action_labels: List[str] = []
    for t in executed_tools:
        if t.startswith("linear_"):
            action_labels.append("update_linear")
        elif t.startswith("slack_") or "slack" in t:
            action_labels.append("post_slack")

    memory_replace("learned_patterns", "workflows[0].observed_sequence", action_labels)
    _bump_workflow_learning(trigger)


def call_tool_memory_aware(
    tool_name: str,
    tool_fn: Callable[..., ToolResult],
    **params: Any,
) -> ToolResult:
    """
    Convenience wrapper:
    - loads memory + infers missing params
    - calls the underlying tool function
    - persists stable info after success

    `tool_fn` is injected by the caller (agent runtime / MCP wrapper).
    """
    new_params, _mem = prepare_tool_call(tool_name, params)
    result = tool_fn(**new_params)
    persist_after_tool_call(tool_name, new_params, result)
    return result


