"""
Agent-internal, always-in-context memory.

This package provides a minimal Letta-inspired core memory layer backed by
human-readable JSON files committed in-repo under `.memory/blocks/`.
"""

from .core import load_all_core_memory, memory_insert, memory_replace
from .orchestrator import (
    ToolCall,
    call_tool_memory_aware,
    persist_after_tool_call,
    prepare_tool_call,
    propose_actions_for_trigger,
    record_trigger_and_complete_sequence,
)

__all__ = [
    "ToolCall",
    "call_tool_memory_aware",
    "load_all_core_memory",
    "memory_insert",
    "memory_replace",
    "persist_after_tool_call",
    "prepare_tool_call",
    "propose_actions_for_trigger",
    "record_trigger_and_complete_sequence",
]


