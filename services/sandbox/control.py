from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aiohttp import ClientSession, ClientTimeout


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    status: str
    result: dict[str, Any] | None = None
    cause: str = ""


class ProducerControlAdapter(Protocol):
    async def execute(
        self,
        *,
        endpoint_context: dict[str, Any],
        action_id: str,
        action: str,
        parameters: dict[str, Any],
    ) -> ActionOutcome: ...

    async def close(self) -> None: ...


class HttpProducerControlAdapter:
    """Forward producer actions to a deployment-provided business endpoint.

    The interface specification deliberately does not prescribe the robot-side
    URL.  The template is therefore deployment configuration, and may use
    ``{ue_ipv4_address}``, ``{agent_id}``, and ``{binding_ref}`` placeholders.
    """

    def __init__(self, endpoint_template: str, timeout_seconds: float = 15.0) -> None:
        self.endpoint_template = endpoint_template.strip()
        self.timeout_seconds = timeout_seconds
        self._http: ClientSession | None = None

    async def execute(
        self,
        *,
        endpoint_context: dict[str, Any],
        action_id: str,
        action: str,
        parameters: dict[str, Any],
    ) -> ActionOutcome:
        if not self.endpoint_template:
            return ActionOutcome("FAILED", cause="producer-control-endpoint-unconfigured")
        try:
            url = self.endpoint_template.format_map(endpoint_context)
        except (KeyError, ValueError):
            return ActionOutcome("FAILED", cause="producer-control-endpoint-invalid")

        if self._http is None:
            self._http = ClientSession(
                timeout=ClientTimeout(total=self.timeout_seconds)
            )
        payload = {
            "action_id": action_id,
            "compute_service_session_id": endpoint_context["compute_service_session_id"],
            "compute_instance_id": endpoint_context["compute_instance_id"],
            "binding_ref": endpoint_context["binding_ref"],
            "agent_id": endpoint_context["agent_id"],
            "action": action,
            "parameters": parameters,
        }
        try:
            async with self._http.post(url, json=payload) as response:
                body = await response.json(content_type=None)
                if response.status >= 400 or not isinstance(body, dict):
                    return ActionOutcome("FAILED", cause="producer-control-rejected")
        except Exception:
            return ActionOutcome("FAILED", cause="producer-control-unavailable")

        status = str(body.get("status", "COMPLETED")).upper()
        if status not in {"RUNNING", "COMPLETED", "FAILED", "CANCELLED"}:
            return ActionOutcome("FAILED", cause="producer-control-invalid-response")
        result = body.get("result")
        if result is not None and not isinstance(result, dict):
            return ActionOutcome("FAILED", cause="producer-control-invalid-response")
        cause = str(body.get("cause", ""))
        return ActionOutcome(status, result, cause)

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None
