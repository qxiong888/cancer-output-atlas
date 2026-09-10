"""Single swap-point for the official Strands Agents SDK.

This file is the only module that imports strands.
If pip install strands-agents works, we use the real Agent and tool decorator
from https://strandsagents.com/ — we do not re-implement or fake those APIs.

If the import fails, the decorator degrades to a plain-function wrapper so the
deterministic pipeline and pytest still run. That fallback is not a Strands API.
Swap this one file to restore official Strands.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

STRANDS_AVAILABLE = False
STRANDS_VERSION = None
STRANDS_IMPORT_ERROR = None
Agent = None

try:
    from strands import Agent as _Agent
    from strands import tool as _tool

    try:
        from importlib.metadata import version

        STRANDS_VERSION = version("strands-agents")
    except Exception:
        STRANDS_VERSION = "strands"

    Agent = _Agent
    tool = _tool
    STRANDS_AVAILABLE = True
except Exception as exc:
    STRANDS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    def tool(func=None, **_kwargs):
        """Passthrough decorator. Not a Strands API — see module docstring."""

        def wrap(fn):
            return fn

        return func if func is not None else wrap


def build_agent(*, tools, system_prompt, model=None):
    """Construct an official strands.Agent. Requires strands-agents.

    Default `coa run` does not call this. It invokes tools as functions so the
    atlas can run unattended without Bedrock or other model credentials.
    """
    if not STRANDS_AVAILABLE or Agent is None:
        raise RuntimeError(
            "official strands-agents is not importable; "
            f"pipeline tools still work. detail={STRANDS_IMPORT_ERROR}"
        )
    kwargs = {
        "tools": tools,
        "system_prompt": system_prompt,
        "callback_handler": None,
        "name": "Cancer Output Atlas",
        "description": (
            "Classify and link public cancer research outputs. "
            "Never invent GEO/SRA/DOI/NCT IDs. Skip on fetch failure."
        ),
    }
    if model:
        kwargs["model"] = model
    return Agent(**kwargs)


FIND_SYSTEM_PROMPT = (
    "You find reusable public cancer research outputs on a baked graph. "
    "Always call find_public_outputs with the user goal. "
    "Never invent GEO, SRA, DOI, or NCT identifiers. "
    "Public metadata only."
)


def find_loop_model():
    """Official strands Model that always calls the find tool. No Bedrock/AgentCore."""
    if not STRANDS_AVAILABLE:
        raise RuntimeError(
            "official strands-agents is not importable; "
            f"detail={STRANDS_IMPORT_ERROR}"
        )
    import json
    import uuid

    from strands.models import Model

    class FindLoopModel(Model):
        def __init__(self) -> None:
            self.config = {"model_id": "coa-find-loop"}

        def update_config(self, **model_config: Any) -> None:
            self.config.update(model_config)

        def get_config(self) -> dict[str, Any]:
            return self.config

        async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
            if False:
                yield {}
            raise NotImplementedError("structured_output is not used for find")

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            last = messages[-1] if messages else {}
            content = last.get("content") or []
            has_tool_result = any(isinstance(b, dict) and "toolResult" in b for b in content)
            yield {"messageStart": {"role": "assistant"}}
            if has_tool_result or not tool_specs:
                yield {"contentBlockStart": {"start": {}}}
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": "Ranked public outputs on the baked graph."}
                    }
                }
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "end_turn"}}
                return
            name = tool_specs[0]["name"]
            goal = ""
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                for block in msg.get("content") or []:
                    if isinstance(block, dict) and block.get("text"):
                        goal = str(block["text"])
            tool_id = "tooluse_" + uuid.uuid4().hex[:16]
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"name": name, "toolUseId": tool_id}}
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": json.dumps({"goal": goal})}}
                }
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}

    return FindLoopModel()


def build_find_agent(*, tools, system_prompt: str | None = None):
    """Construct strands.Agent for /api/find. Uses FindLoopModel, not Bedrock."""
    if not STRANDS_AVAILABLE or Agent is None:
        raise RuntimeError(
            "official strands-agents is not importable; "
            f"detail={STRANDS_IMPORT_ERROR}"
        )
    return Agent(
        tools=tools,
        system_prompt=system_prompt or FIND_SYSTEM_PROMPT,
        callback_handler=None,
        name="Cancer Output Atlas",
        description=(
            "Find public cancer research outputs on a baked graph. "
            "Never invent GEO/SRA/DOI/NCT IDs."
        ),
        model=find_loop_model(),
    )
