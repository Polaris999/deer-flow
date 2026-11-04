"""Skill allowed-tools is assembly-time surface authority when a skill is active.

When slash activation or in-context skill load makes a skill policy-active and
that skill declares ``allowed-tools``, ``SkillToolPolicyMiddleware`` binds only
those schemas (plus framework discovery helpers). Guardrails and authorization
remain a second line for writes and RBAC denials.

These regressions drive the real ``build_middlewares`` (lead) and
``build_subagent_runtime_middlewares`` chains through ``create_agent``, activate a
synthetic skill via a real slash command, and assert both model binding and
actual ToolNode execution on sync and async paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from deerflow.agents.lead_agent import agent as lead_agent_module
from deerflow.agents.middlewares.durable_context_middleware import DurableContextMiddleware
from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware
from deerflow.agents.middlewares.skill_tool_policy_middleware import SkillToolPolicyMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares
from deerflow.agents.thread_state import ThreadState
from deerflow.authz.rbac import RbacAuthorizationProvider
from deerflow.config.app_config import AppConfig
from deerflow.config.authorization_config import AuthorizationConfig, AuthorizationProviderConfig
from deerflow.config.loop_detection_config import LoopDetectionConfig
from deerflow.config.model_config import ModelConfig
from deerflow.config.safety_finish_reason_config import SafetyFinishReasonConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.skills_config import SkillsConfig
from deerflow.config.subagents_config import SubagentsAppConfig
from deerflow.config.token_budget_config import TokenBudgetConfig

_PARSE_CALLS: list[str] = []
_WRITE_CALLS: list[str] = []
_DEVICE_CALLS: list[str] = []


@tool
def parse_document(path: str) -> str:
    """Parse a document and return its text."""
    _PARSE_CALLS.append(path)
    return "parsed"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    _WRITE_CALLS.append(path)
    return "written"


@tool
def read_file(path: str) -> str:
    """Read a file from the skills container."""
    return "file body"


@tool
def control_device(device_id: str, action: str) -> str:
    """Control an IoT device (not part of meeting workflow)."""
    _DEVICE_CALLS.append(f"{device_id}:{action}")
    return "ok"


class _RecordingModel(GenericFakeChatModel):
    bound_tool_names: list[list[str]] = Field(default_factory=list)

    def __init__(self, responses: list[AIMessage]):
        super().__init__(messages=iter(responses))

    def bind_tools(self, tools: Any, **kwargs: Any):
        self.bound_tool_names.append([getattr(candidate, "name", "") for candidate in tools])
        return self


class _NoopMiddleware:
    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "<noop-test-middleware>"


def _make_app_config(skills_root: Path) -> AppConfig:
    return AppConfig(
        models=[
            ModelConfig(
                name="safe-model",
                display_name="safe-model",
                description=None,
                use="langchain_openai:ChatOpenAI",
                model="safe-model",
                supports_thinking=False,
                supports_vision=False,
            )
        ],
        sandbox=SandboxConfig(use="test"),
        loop_detection=LoopDetectionConfig(enabled=False),
        skills=SkillsConfig(path=str(skills_root)),
        safety_finish_reason=SafetyFinishReasonConfig(enabled=False),
        subagents=SubagentsAppConfig(token_budget=TokenBudgetConfig(enabled=False)),
    )


def _make_skill(skills_root: Path, name: str, allowed_tools: list[str] | None) -> None:
    skill_dir = skills_root / "custom" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        "description: Demo skill for tool authority regressions",
        "license: MIT",
    ]
    if allowed_tools is None:
        pass
    elif not allowed_tools:
        lines.append("allowed-tools: []")
    else:
        lines.append("allowed-tools:")
        lines.extend(f"  - {entry}" for entry in allowed_tools)
    lines.extend(["---", "", "# Demo", "", "Use the parser."])
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("\n".join(lines), encoding="utf-8")


def _lead_middlewares(monkeypatch, app_config: AppConfig) -> list:
    monkeypatch.setattr(lead_agent_module, "build_lead_runtime_middlewares", lambda *, app_config, lazy_init=True: [])
    monkeypatch.setattr(lead_agent_module, "_create_summarization_middleware", lambda **_kwargs: None)
    monkeypatch.setattr(lead_agent_module, "_create_todo_list_middleware", lambda is_plan_mode: None)
    monkeypatch.setattr(lead_agent_module, "TitleMiddleware", lambda *, app_config, extensions: _NoopMiddleware())
    monkeypatch.setattr(lead_agent_module, "MemoryMiddleware", lambda *args, **kwargs: _NoopMiddleware())
    return lead_agent_module.build_middlewares(
        {"configurable": {"is_plan_mode": False, "subagent_enabled": False}},
        model_name="safe-model",
        app_config=app_config,
    )


def _subagent_middlewares(app_config: AppConfig) -> list:
    return build_subagent_runtime_middlewares(app_config=app_config, model_name="safe-model")


def _skill_authority_slice(middlewares: list) -> list:
    start = next(i for i, middleware in enumerate(middlewares) if isinstance(middleware, SkillActivationMiddleware))
    end = next(i for i in range(start + 1, len(middlewares)) if isinstance(middlewares[i], DurableContextMiddleware))
    return middlewares[start:end]


def _scripted_model(*, call_device: bool = False) -> _RecordingModel:
    tool_name = "control_device" if call_device else "parse_document"
    tool_args = {"device_id": "lamp-1", "action": "on"} if call_device else {"path": "report.pdf"}
    return _RecordingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": "tool-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )


class _Journal:
    def __init__(self) -> None:
        self.middleware_calls: list[dict] = []
        self.skill_usages: list[Any] = []

    def record_middleware(self, *args, **kwargs):
        self.middleware_calls.append(kwargs)

    def record_skill_usage(self, *args, **kwargs):
        self.skill_usages.append(args[0] if args else kwargs)


def _invoke(graph, *, asynchronous: bool) -> tuple[dict, _Journal]:
    journal = _Journal()
    graph_input = {"messages": [HumanMessage(content="/demo-skill parse the report", id="msg-1")]}
    context = {"__run_journal": journal}
    config = {"recursion_limit": 20}
    if asynchronous:
        result = asyncio.run(graph.ainvoke(graph_input, context=context, config=config))
    else:
        result = graph.invoke(graph_input, context=context, config=config)
    return result, journal


@pytest.mark.parametrize("builder", ["lead", "subagent"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_active_skill_allowed_tools_filters_unlisted_tools(monkeypatch, tmp_path, builder, asynchronous):
    """Restrictive allowed-tools hides business tools not declared by the active skill."""
    allowed = ["write_file"]
    _make_skill(tmp_path, "demo-skill", allowed)
    app_config = _make_app_config(tmp_path)
    middlewares = _lead_middlewares(monkeypatch, app_config) if builder == "lead" else _subagent_middlewares(app_config)
    slice_ = _skill_authority_slice(middlewares)
    assert any(isinstance(m, SkillToolPolicyMiddleware) for m in slice_)

    model = _scripted_model()
    graph = create_agent(
        model=model,
        tools=[read_file, parse_document, write_file, control_device],
        middleware=slice_,
        state_schema=ThreadState,
    )

    _PARSE_CALLS.clear()
    _DEVICE_CALLS.clear()
    result, journal = _invoke(graph, asynchronous=asynchronous)

    bound = set(model.bound_tool_names[0])
    assert "parse_document" not in bound
    assert "control_device" not in bound
    assert "write_file" in bound
    assert "read_file" in bound
    assert _PARSE_CALLS == []
    assert result["messages"][-1].content == "done"
    assert [usage.get("name") for usage in journal.skill_usages if isinstance(usage, dict)] == ["demo-skill"]


@pytest.mark.parametrize("builder", ["lead", "subagent"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_active_skill_blocks_unlisted_tool_execution(monkeypatch, tmp_path, builder, asynchronous):
    """Calls to tools removed from the model surface are refused at execution time."""
    _make_skill(
        tmp_path,
        "demo-skill",
        [
            "list_available_meeting_rooms",
            "create_booking",
            "cancel_booking",
            "list_bookings",
        ],
    )
    app_config = _make_app_config(tmp_path)
    middlewares = _lead_middlewares(monkeypatch, app_config) if builder == "lead" else _subagent_middlewares(app_config)
    slice_ = _skill_authority_slice(middlewares)

    model = _scripted_model(call_device=True)
    graph = create_agent(
        model=model,
        tools=[read_file, control_device],
        middleware=slice_,
        state_schema=ThreadState,
    )

    _DEVICE_CALLS.clear()
    result, _journal = _invoke(graph, asynchronous=asynchronous)

    assert _DEVICE_CALLS == []
    blocked = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "tool-call-1"]
    assert len(blocked) == 1
    assert blocked[0].status == "error"
    assert "not allowed by the active skill policy" in blocked[0].content


@pytest.mark.parametrize("asynchronous", [False, True])
def test_skill_without_allowed_tools_does_not_filter(monkeypatch, tmp_path, asynchronous):
    """Skills that omit allowed-tools keep the agent's full authorized tool set."""
    _make_skill(tmp_path, "demo-skill", None)
    app_config = _make_app_config(tmp_path)
    middlewares = _lead_middlewares(monkeypatch, app_config)
    slice_ = _skill_authority_slice(middlewares)

    model = _scripted_model()
    graph = create_agent(
        model=model,
        tools=[read_file, parse_document, write_file],
        middleware=slice_,
        state_schema=ThreadState,
    )

    _PARSE_CALLS.clear()
    result, _journal = _invoke(graph, asynchronous=asynchronous)

    assert {"read_file", "parse_document", "write_file"} <= set(model.bound_tool_names[0])
    assert _PARSE_CALLS == ["report.pdf"]
    assert result["messages"][-1].content == "done"


def test_allowed_tools_metadata_cannot_grant_rbac_denied_tool(monkeypatch, tmp_path):
    """Declaring a tool in allowed-tools never bypasses genuine Agent authorization."""
    from deerflow.agents.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares

    provider = RbacAuthorizationProvider(roles={"guest": {"tools": {"allow": ["read_file", "parse_document"]}}})
    _make_skill(tmp_path, "demo-skill", ["write_file"])
    app_config = _make_app_config(tmp_path)
    app_config.authorization = AuthorizationConfig(
        enabled=True,
        provider=AuthorizationProviderConfig(use="unused:Provider"),
    )

    monkeypatch.setattr(lead_agent_module, "_create_summarization_middleware", lambda **_kwargs: None)
    monkeypatch.setattr(lead_agent_module, "_create_todo_list_middleware", lambda is_plan_mode: None)
    monkeypatch.setattr(lead_agent_module, "TitleMiddleware", lambda *, app_config, extensions: _NoopMiddleware())
    monkeypatch.setattr(lead_agent_module, "MemoryMiddleware", lambda *args, **kwargs: _NoopMiddleware())

    runtime_middlewares = build_lead_runtime_middlewares(
        app_config=app_config,
        authorization_provider=provider,
    )
    full_chain = _lead_middlewares(monkeypatch, app_config)
    slice_ = _skill_authority_slice(full_chain)

    model = _RecordingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "out.txt", "content": "x"},
                        "id": "denied-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = create_agent(
        model=model,
        tools=[read_file, parse_document, write_file],
        middleware=[*runtime_middlewares, *slice_],
        state_schema=ThreadState,
    )

    _WRITE_CALLS.clear()
    journal = SimpleNamespace(
        record_middleware=lambda *args, **kwargs: None,
        record_skill_usage=lambda *args, **kwargs: None,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="/demo-skill write the report", id="msg-1")]},
        context={"__run_journal": journal, "user_role": "guest", "thread_id": "t-authority-test"},
        config={"recursion_limit": 20, "configurable": {"thread_id": "t-authority-test"}},
    )

    assert _WRITE_CALLS == []
    blocked = [message for message in result["messages"] if isinstance(message, ToolMessage) and message.tool_call_id == "denied-call"]
    assert len(blocked) == 1
    assert blocked[0].status == "error"
    assert "authz.denied" in blocked[0].content
