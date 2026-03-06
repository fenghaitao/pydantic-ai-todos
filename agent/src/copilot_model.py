"""
PydanticAI Model implementation for GitHub Copilot SDK.

Uses the SDK's native `tools` session config so the model issues real tool
calls rather than relying on prompt-injected JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

try:
    from copilot import CopilotClient, PermissionHandler
    from copilot.types import SessionConfig, Tool, ToolInvocation, ToolResult
except ImportError as e:
    raise ImportError(
        "Please install github-copilot-sdk to use CopilotModel: "
        "pip install github-copilot-sdk"
    ) from e

log = logging.getLogger("copilot_model")


def _find_copilot_binary() -> str:
    import os
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "copilot", "bin", "copilot")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    path_binary = shutil.which("copilot")
    if path_binary:
        return path_binary
    raise RuntimeError("Copilot CLI not found. Install github-copilot-sdk or provide cli_path.")


def _make_sdk_tools(function_tools: list[Any], tool_call_queue: asyncio.Queue) -> list[Tool]:
    """
    Wrap pydantic-ai function tools as SDK Tool objects.

    When the Copilot CLI calls back via JSON-RPC (tool.call), the handler
    puts (tool_name, args) into the queue and returns a placeholder.
    pydantic-ai owns the actual execution — we surface it as a ToolCallPart.
    """
    sdk_tools: list[Tool] = []
    for ft in function_tools:
        def make_handler(name: str):
            async def handler(invocation: ToolInvocation) -> ToolResult:
                log.warning("[SDK TOOL] handler called: %s args=%s", name, invocation.get("arguments") if isinstance(invocation, dict) else invocation)
                args = invocation.get("arguments") if isinstance(invocation, dict) else invocation.arguments
                tool_call_queue.put_nowait((name, args))
                # Block until pydantic-ai has processed the tool call and put the result back
                # We return a placeholder — the real result goes through pydantic-ai's loop
                return {"textResultForLlm": "Tool executed successfully.", "resultType": "success"}
            return handler

        sdk_tools.append(Tool(
            name=ft.name,
            description=ft.description or "",
            handler=make_handler(ft.name),
            parameters=ft.parameters_json_schema,
        ))
    return sdk_tools


@dataclass
class CopilotStreamedResponse(StreamedResponse):
    _session: Any = field(repr=False)
    _user_prompt: str = field(repr=False)
    _timeout: float = field(repr=False)
    _model_name_value: str = field(repr=False)
    _output_tools: list[Any] = field(repr=False)
    _function_tools: list[Any] = field(repr=False)
    _allow_text_output: bool = field(repr=False)
    _tool_call_queue: asyncio.Queue = field(repr=False)
    _timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc), repr=False)

    @property
    def model_name(self) -> str:
        return self._model_name_value

    @property
    def provider_name(self) -> str:
        return "copilot"

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        event_queue: asyncio.Queue = asyncio.Queue()

        def on_event(event):
            event_queue.put_nowait(event)

        unsubscribe = self._session.on(on_event)
        try:
            log.warning("[STREAM] sending prompt: %r", self._user_prompt[:200])
            await self._session.send({"prompt": self._user_prompt})
            full_content = ""
            deadline = asyncio.get_event_loop().time() + self._timeout

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise RuntimeError(f"Timeout after {self._timeout}s")

                # Poll both the tool call queue and the event queue
                try:
                    tool_name, args = self._tool_call_queue.get_nowait()
                    log.warning("[STREAM] native tool call intercepted: %s args=%s", tool_name, args)
                    part = ToolCallPart(tool_name=tool_name, args=args)
                    yield self._parts_manager.handle_part(vendor_part_id=None, part=part)
                    return
                except asyncio.QueueEmpty:
                    pass

                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    # Check tool queue again before looping
                    try:
                        tool_name, args = self._tool_call_queue.get_nowait()
                        log.warning("[STREAM] native tool call (post-timeout): %s args=%s", tool_name, args)
                        part = ToolCallPart(tool_name=tool_name, args=args)
                        yield self._parts_manager.handle_part(vendor_part_id=None, part=part)
                        return
                    except asyncio.QueueEmpty:
                        pass
                    continue

                etype = event.type.value if hasattr(event.type, 'value') else str(event.type)
                log.warning("[STREAM] event: %s data=%s", etype, str(getattr(event, 'data', ''))[:200])

                if etype == "assistant.message":
                    full_content = event.data.content
                elif etype == "session.idle":
                    log.warning("[STREAM] idle. content=%r", full_content[:300] if full_content else "")
                    break
                elif etype == "session.error":
                    raise RuntimeError(f"Session error: {event.data.message}")

            # Final check for tool calls that arrived just before idle
            try:
                tool_name, args = self._tool_call_queue.get_nowait()
                log.warning("[STREAM] native tool call (post-idle): %s args=%s", tool_name, args)
                part = ToolCallPart(tool_name=tool_name, args=args)
                yield self._parts_manager.handle_part(vendor_part_id=None, part=part)
                return
            except asyncio.QueueEmpty:
                pass

            # No tool call — plain text response
            part = TextPart(content=full_content)
            yield self._parts_manager.handle_part(vendor_part_id=None, part=part)
        finally:
            unsubscribe()


@dataclass
class CopilotModel(Model):
    cli_path: str | None = None
    model_name_value: str = "gpt-4"
    working_directory: str | None = None
    timeout: float = 300.0

    _client: Optional[Any] = field(default=None, repr=False)
    _session_lock: Optional[asyncio.Lock] = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return self.model_name_value

    @property
    def system(self) -> str:
        return "copilot"

    def __post_init__(self):
        if self.cli_path is None:
            self.cli_path = _find_copilot_binary()
        self._session_lock = asyncio.Lock()

    async def _ensure_client(self):
        if self._client is None:
            client_opts = {"cli_path": self.cli_path} if self.cli_path else {}
            self._client = CopilotClient(client_opts)
            await self._client.start()

    async def _create_session(
        self,
        system_prompt: str | None,
        model_settings: ModelSettings | None,
        sdk_tools: list[Any] | None = None,
    ) -> Any:
        await self._ensure_client()
        session_config: SessionConfig = {
            "on_permission_request": PermissionHandler.approve_all,
        }
        model = self.model_name_value
        if model_settings and "copilot_model" in model_settings:
            model = model_settings["copilot_model"]
        session_config["model"] = model

        if model_settings and "copilot_provider" in model_settings:
            session_config["provider"] = model_settings["copilot_provider"]
        if system_prompt:
            session_config["system_message"] = {"mode": "append", "content": system_prompt}
        working_dir = self.working_directory
        if model_settings and "copilot_working_directory" in model_settings:
            working_dir = model_settings["copilot_working_directory"]
        if working_dir:
            session_config["working_directory"] = working_dir
        if sdk_tools:
            session_config["tools"] = sdk_tools

        log.warning("[SESSION] model=%s tools=%s", model, [t.name for t in (sdk_tools or [])])
        return await self._client.create_session(session_config)

    def _convert_messages_to_prompt(self, messages: list[ModelMessage]) -> tuple[str | None, str]:
        """Convert pydantic-ai messages to a system prompt + user prompt string.

        NOTE: We do NOT inject any JSON-format instructions here — the SDK's
        native tool calling handles function dispatch via JSON-RPC callbacks.
        """
        system_prompt = None
        user_parts = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt = part.content
                    elif isinstance(part, UserPromptPart):
                        user_parts.append(str(part.content))
                    elif isinstance(part, ToolReturnPart):
                        user_parts.append(f"Tool {part.tool_name} returned: {part.content}")
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        user_parts.append(f"Assistant: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        user_parts.append(f"Assistant called tool {part.tool_name} with args: {part.args}")
        return system_prompt, "\n".join(user_parts) if user_parts else ""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        timeout = self.timeout
        if model_settings and "copilot_timeout" in model_settings:
            timeout = model_settings["copilot_timeout"]

        tool_call_queue: asyncio.Queue = asyncio.Queue()
        sdk_tools = _make_sdk_tools(list(model_request_parameters.function_tools or []), tool_call_queue)

        session = await self._create_session(system_prompt, model_settings, sdk_tools or None)
        try:
            response = await session.send_and_wait({"prompt": user_prompt}, timeout=timeout)
        finally:
            try:
                await session.destroy()
            except Exception:
                pass

        if not response:
            raise RuntimeError("No response from Copilot agent")

        # Check if a native tool call was intercepted
        if not tool_call_queue.empty():
            tool_name, args = tool_call_queue.get_nowait()
            log.warning("[REQUEST] native tool call: %s args=%s", tool_name, args)
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)], usage=RequestUsage())

        content = response.data.content if hasattr(response, 'data') else str(response)
        log.warning("[REQUEST] text response: %r", content[:200])
        return ModelResponse(parts=[TextPart(content=content)], usage=RequestUsage())

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[CopilotStreamedResponse]:
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        timeout = self.timeout
        if model_settings and "copilot_timeout" in model_settings:
            timeout = model_settings["copilot_timeout"]

        tool_call_queue: asyncio.Queue = asyncio.Queue()
        sdk_tools = _make_sdk_tools(list(model_request_parameters.function_tools or []), tool_call_queue)

        await self._ensure_client()
        session = await self._create_session(system_prompt, model_settings, sdk_tools or None)
        try:
            yield CopilotStreamedResponse(
                model_request_parameters=model_request_parameters,
                _session=session,
                _user_prompt=user_prompt,
                _timeout=timeout,
                _model_name_value=self.model_name_value,
                _output_tools=list(model_request_parameters.output_tools or []),
                _function_tools=list(model_request_parameters.function_tools or []),
                _allow_text_output=model_request_parameters.allow_text_output,
                _tool_call_queue=tool_call_queue,
            )
        finally:
            try:
                await session.destroy()
            except Exception:
                pass

    async def cleanup(self):
        if self._client:
            try:
                await self._client.stop()
            except Exception:
                pass
            self._client = None

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, *args):
        await self.cleanup()
