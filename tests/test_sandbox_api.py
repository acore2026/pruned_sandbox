from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from services.video.rtc_server import ApiError
from services.sandbox.api import SandboxApi, canonical_digest
from services.sandbox.control import ActionOutcome
from services.sandbox.main import (
    RUNTIME_KEY,
    VideoRuntime,
    create_management_app,
    create_user_app,
    management_health,
)
from services.video.config import VideoSettings


@web.middleware
async def contract_errors(request: web.Request, handler):
    try:
        return await handler(request)
    except ApiError as exc:
        return web.json_response(
            {"error": {"code": exc.code, "message": exc.message}},
            status=exc.status,
        )


class FakeMediaEngine:
    def __init__(self) -> None:
        self.bindings: set[str] = set()
        self.targets: dict[str, str | None] = {}
        self.closed_connections: list[str] = []

    def ensure_binding(self, binding_ref: str) -> None:
        self.bindings.add(binding_ref)

    def set_recognition_target(self, binding_ref: str, prompt: str | None) -> None:
        self.targets[binding_ref] = prompt

    async def create_standard_media_connection(
        self, binding_ref: str, role: str, offer_sdp: str, state_callback
    ):
        await state_callback(role, True, "")
        return SimpleNamespace(role=role, id=f"{binding_ref}:{role}"), {
            "type": "answer",
            "sdp": offer_sdp.replace("offer", "answer"),
        }

    async def close_standard_media_connection(self, binding_ref: str, handle) -> None:
        self.closed_connections.append(handle.id)

    async def close_binding(self, binding_ref: str) -> None:
        self.bindings.discard(binding_ref)

    async def search_object(self, binding_ref: str, prompt: str, timeout_ms: int):
        self.assert_bound(binding_ref)
        return [{"label": prompt, "confidence": 0.94}]

    def assert_bound(self, binding_ref: str) -> None:
        if binding_ref not in self.bindings:
            raise RuntimeError("binding is closed")


class FakeProducerControlAdapter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ActionOutcome(
            "COMPLETED", {"accepted": True, "executed": True}, ""
        )

    async def close(self) -> None:
        return None


class SandboxPlaneIsolationTest(IsolatedAsyncioTestCase):
    async def test_management_and_user_routes_are_isolated(self) -> None:
        settings = VideoSettings.from_env()
        runtime = VideoRuntime.build(settings)
        management_paths = {
            route.resource.canonical for route in create_management_app(runtime).router.routes()
        }
        user_paths = {
            route.resource.canonical for route in create_user_app(runtime).router.routes()
        }

        self.assertIn(
            "/management/v1/compute-session-bindings:bind", management_paths
        )
        self.assertNotIn("/v1/media-connections", management_paths)
        self.assertIn("/v1/media-connections", user_paths)
        self.assertIn("/v1/audio-control-actions", user_paths)
        self.assertNotIn(
            "/management/v1/compute-session-bindings:bind", user_paths
        )
        await runtime.close()

    async def test_management_health_counts_only_active_bindings(self) -> None:
        runtime = SimpleNamespace(
            api=SimpleNamespace(
                bindings={
                    "active": SimpleNamespace(state="BOUND"),
                    "history": SimpleNamespace(state="UNBOUND"),
                }
            )
        )
        request = SimpleNamespace(app={RUNTIME_KEY: runtime})
        response = await management_health(request)
        payload = json.loads(response.body)
        self.assertEqual(1, payload["bindings"])
        self.assertEqual(2, payload["binding_records"])


class SandboxContractApiTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = FakeMediaEngine()
        self.control_adapter = FakeProducerControlAdapter()
        self.api = SandboxApi(
            self.engine, "secret", "", self.control_adapter
        )
        app = web.Application(middlewares=[contract_errors])
        self.api.register_routes(app)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.configuration = {
            "compute_service_session_id": "css-1",
            "connection_parameters": {
                "transport": "WEBRTC",
                "media_connections_path": "/v1/media-connections",
                "recognition_target_path_template": "/v1/recognition-targets/{compute_service_session_id}",
            },
            "participants_network_facts": [
                {
                    "role": "consumer", "agent_id": "glasses",
                    "up_session_id": "ups-glasses", "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.11", "dnn": "internet",
                    "snssai": "1-010203", "generation": "1",
                },
                {
                    "role": "producer", "agent_id": "dog",
                    "up_session_id": "ups-dog", "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.12", "dnn": "internet",
                    "snssai": "1-010203", "generation": "1",
                },
            ],
        }
        self.bind_body = {
            "activation_idempotency_key": "activate-1",
            "owner_ref": "ca/css-1",
            "binding_ref": "binding-1",
            "compute_service_session_id": "css-1",
            "compute_instance_id": "ci-1",
            "configuration": self.configuration,
            "configuration_digest": canonical_digest(self.configuration),
            "expected_binding_absent": True,
        }
        self.context = {
            "compute_service_session_id": "css-1",
            "compute_instance_id": "ci-1",
            "binding_ref": "binding-1",
            "role": "consumer",
            "agent_id": "glasses",
        }

    async def asyncTearDown(self) -> None:
        await self.api.close()
        await self.client.close()

    async def bind(self):
        return await self.client.post(
            "/management/v1/compute-session-bindings:bind",
            json=self.bind_body,
            headers={"Authorization": "Bearer secret"},
        )

    async def test_binding_media_recognition_control_and_unbind(self) -> None:
        response = await self.bind()
        self.assertEqual(200, response.status, await response.text())
        binding = await response.json()
        self.assertEqual("BOUND", binding["state"])
        self.assertEqual("activate-1", binding["activation_idempotency_key"])

        response = await self.client.post(
            "/v1/media-connections",
            json={
                "request_id": "media-1",
                "computing_context": self.context,
                "offer": {"type": "offer", "sdp": "v=0 offer"},
            },
        )
        media = await response.json()
        self.assertEqual(201, response.status, media)
        self.assertEqual("answer", media["answer"]["type"])

        producer_context = dict(self.context, role="producer", agent_id="dog")
        response = await self.client.post(
            "/v1/media-connections",
            json={
                "request_id": "media-producer-1",
                "computing_context": producer_context,
                "offer": {"type": "offer", "sdp": "v=0 offer"},
            },
        )
        self.assertEqual(201, response.status, await response.text())

        response = await self.client.get(
            "/management/v1/compute-session-bindings/binding-1/media-state",
            headers={"Authorization": "Bearer secret"},
        )
        media_state = await response.json()
        self.assertTrue(media_state["consumer_connected"])
        self.assertTrue(media_state["producer_connected"])

        response = await self.client.put(
            "/v1/recognition-targets/css-1",
            json={
                "request_id": "target-1",
                "computing_context": self.context,
                "input": {"type": "TEXT", "text": "寻找红色玩偶", "language": "zh"},
            },
        )
        target = await response.json()
        self.assertEqual(200, response.status, target)
        self.assertEqual("red doll", target["target"]["prompt"])
        self.assertEqual("red doll", self.engine.targets["binding-1"])

        response = await self.client.post(
            "/v1/control-actions",
            json={
                "request_id": "action-1",
                "computing_context": self.context,
                "action": "movement",
                "input": {"type": "STRUCTURED"},
                "parameters": {"direction": "forward"},
            },
        )
        action = await response.json()
        self.assertEqual(202, response.status, action)
        self.assertEqual("movement", action["normalized_action"])
        completed = await self._wait_for_action(action["action_id"])
        self.assertEqual("COMPLETED", completed["status"])
        self.assertTrue(completed["result"]["executed"])

        response = await self.client.post(
            "/v1/control-actions",
            json={
                "request_id": "action-text-left-1",
                "computing_context": self.context,
                "input": {"type": "TEXT", "text": "向左", "language": "zh"},
            },
        )
        text_action = await response.json()
        self.assertEqual(202, response.status, text_action)
        self.assertEqual("movement", text_action["normalized_action"])
        self.assertEqual({"direction": "left"}, text_action["normalized_parameters"])
        await self._wait_for_action(text_action["action_id"])
        self.assertEqual(
            {"direction": "left"}, self.control_adapter.calls[-1]["parameters"]
        )
        self.assertEqual("10.60.0.12", self.control_adapter.calls[0]["endpoint_context"]["ue_ipv4_address"])

        response = await self.client.post(
            "/v1/control-actions",
            json={
                "request_id": "search-1",
                "computing_context": self.context,
                "action": "search_object",
                "input": {"type": "STRUCTURED"},
                "parameters": {"query": "cup", "timeout_ms": 1000},
            },
        )
        search = await response.json()
        self.assertEqual(202, response.status, search)
        search = await self._wait_for_action(search["action_id"])
        self.assertEqual("COMPLETED", search["status"])
        self.assertEqual("cup", search["result"]["matches"][0]["label"])
        self.assertEqual("red doll", self.engine.targets["binding-1"])

        response = await self.client.post(
            "/management/v1/compute-session-bindings/binding-1:unbind",
            json={
                "owner_ref": "ca/css-1",
                "unbind_idempotency_key": "unbind-1",
                "cause": "released",
            },
            headers={"Authorization": "Bearer secret"},
        )
        unbound = await response.json()
        self.assertEqual("UNBOUND", unbound["state"])
        self.assertEqual("activate-1", unbound["activation_idempotency_key"])

    async def _wait_for_action(self, action_id: str) -> dict:
        for _ in range(20):
            response = await self.client.get(f"/v1/control-actions/{action_id}")
            payload = await response.json()
            if payload["status"] not in {"ACCEPTED", "RUNNING"}:
                return payload
            await asyncio.sleep(0.01)
        self.fail(f"action {action_id} did not finish")

    async def test_management_auth_and_idempotency_conflict(self) -> None:
        response = await self.client.get(
            "/management/v1/compute-session-bindings/missing"
        )
        self.assertEqual(401, response.status)

        self.assertEqual(200, (await self.bind()).status)
        self.assertEqual(200, (await self.bind()).status)
        changed = dict(self.bind_body)
        changed["compute_instance_id"] = "ci-2"
        response = await self.client.post(
            "/management/v1/compute-session-bindings:bind",
            json=changed,
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(409, response.status)
        self.assertEqual("idempotency-conflict", (await response.json())["error"]["code"])
