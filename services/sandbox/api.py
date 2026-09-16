from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout, FormData, web

from services.intent.classifier import RuleIntentClassifier

from services.video.rtc_server import (
    ApiError,
    OrangeVideoServer,
    StandardMediaConnection,
)

from .control import HttpProducerControlAdapter, ProducerControlAdapter


LOGGER = logging.getLogger("sandbox.service")


@dataclass(slots=True)
class BindingRecord:
    activation_key: str
    owner_ref: str
    binding_ref: str
    session_id: str
    instance_id: str
    configuration: dict[str, Any]
    configuration_digest: str
    participants: dict[str, str]
    participant_facts: dict[str, dict[str, Any]]
    state: str = "BOUND"
    initialized: bool = True
    cause: str = ""
    revision: int = 1
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    connected: dict[str, bool] = field(
        default_factory=lambda: {"producer": False, "consumer": False}
    )


@dataclass(slots=True)
class StoredOperation:
    digest: str
    response: dict[str, Any]


@dataclass(slots=True)
class MediaResource:
    request_id: str
    context: dict[str, str]
    media_connection_id: str
    digest: str
    answer: dict[str, str]
    handle: StandardMediaConnection
    closed: bool = False


@dataclass(slots=True)
class RecognitionTarget:
    request_id: str
    context: dict[str, str]
    revision: int
    label: str
    prompt: str


@dataclass(slots=True)
class ControlAction:
    request_id: str
    context: dict[str, str]
    action_id: str
    status: str
    normalized_action: str | None
    normalized_parameters: dict[str, Any]
    digest: str
    control_triggered: bool | None = True
    intent: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    cause: str = ""
    transcription: dict[str, Any] | None = None


class SandboxApi:
    """Stable CMF/N6 contract backed directly by the Python media engine."""

    def __init__(
        self,
        media_engine: OrangeVideoServer,
        management_token: str,
        intent_url: str,
        control_adapter: ProducerControlAdapter | None = None,
    ) -> None:
        self.media_engine = media_engine
        self.management_token = management_token.strip()
        self.intent = RuleIntentClassifier()
        self.intent_url = intent_url.strip()
        self.control_adapter = control_adapter or HttpProducerControlAdapter("")
        self.http: ClientSession | None = None
        self.lock = asyncio.Lock()
        self.bindings: dict[str, BindingRecord] = {}
        self.activation_ops: dict[str, StoredOperation] = {}
        self.unbind_ops: dict[str, StoredOperation] = {}
        self.media: dict[str, MediaResource] = {}
        self.media_ops: dict[str, StoredOperation] = {}
        self.media_pending: set[tuple[str, str]] = set()
        self.targets: dict[str, RecognitionTarget] = {}
        self.target_ops: dict[str, StoredOperation] = {}
        self.actions: dict[str, ControlAction] = {}
        self.control_ops: dict[str, StoredOperation] = {}
        self.action_tasks: dict[str, asyncio.Task[None]] = {}

    async def close(self) -> None:
        tasks = list(self.action_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.action_tasks.clear()
        await self.control_adapter.close()
        if self.http is not None:
            await self.http.close()
            self.http = None

    def register_routes(self, app: web.Application) -> None:
        self.register_management_routes(app)
        self.register_user_routes(app)

    def register_management_routes(self, app: web.Application) -> None:
        app.add_routes(
            [
                web.post(
                    "/management/v1/compute-session-bindings:bind",
                    self.bind,
                ),
                web.get(
                    "/management/v1/compute-session-bindings/{binding_ref}",
                    self.get_binding,
                ),
                web.get(
                    "/management/v1/compute-session-bindings/{binding_ref}/media-state",
                    self.get_media_state,
                ),
                web.post(
                    "/management/v1/compute-session-bindings/{binding_ref}:unbind",
                    self.unbind,
                ),
            ]
        )

    def register_user_routes(self, app: web.Application) -> None:
        app.add_routes(
            [
                web.post("/v1/media-connections", self.create_media_connection),
                web.delete(
                    "/v1/media-connections/{media_connection_id}",
                    self.delete_media_connection,
                ),
                web.put(
                    "/v1/recognition-targets/{compute_service_session_id}",
                    self.update_recognition_target,
                ),
                web.get(
                    "/v1/recognition-targets/{compute_service_session_id}",
                    self.get_recognition_target,
                ),
                web.post("/v1/control-actions", self.create_control_action),
                web.post(
                    "/v1/audio-control-actions",
                    self.create_audio_control_action,
                ),
                web.get(
                    "/v1/control-actions/{action_id}",
                    self.get_control_action,
                ),
            ]
        )

    def _authorize_management(self, request: web.Request) -> None:
        if self.management_token and request.headers.get("Authorization") != (
            f"Bearer {self.management_token}"
        ):
            raise ApiError(401, "unauthorized", "valid management token is required")

    async def bind(self, request: web.Request) -> web.Response:
        self._authorize_management(request)
        payload = await _json_object(request)
        required = (
            "activation_idempotency_key",
            "owner_ref",
            "binding_ref",
            "compute_service_session_id",
            "compute_instance_id",
            "configuration",
            "configuration_digest",
        )
        _require_strings(payload, required[:-2] + ("configuration_digest",))
        if payload.get("expected_binding_absent") is not True:
            raise ApiError(
                400,
                "invalid-request",
                "expected_binding_absent must be true",
            )
        configuration = payload.get("configuration")
        if not isinstance(configuration, dict):
            raise ApiError(400, "invalid-configuration", "configuration must be an object")
        if configuration.get("compute_service_session_id") != payload[
            "compute_service_session_id"
        ]:
            raise ApiError(
                400,
                "invalid-configuration",
                "configuration identifies another session",
            )
        _connection_parameters(configuration)
        expected_digest = canonical_digest(configuration)
        if payload["configuration_digest"] != expected_digest:
            raise ApiError(
                400,
                "configuration-digest-mismatch",
                "configuration digest does not match canonical JSON",
            )
        participant_facts = _participant_facts(configuration)
        participants = {
            role: str(facts["agent_id"])
            for role, facts in participant_facts.items()
        }
        op_key = f"{payload['owner_ref']}\0{payload['activation_idempotency_key']}"
        operation_digest = canonical_digest(payload)
        async with self.lock:
            previous = self.activation_ops.get(op_key)
            if previous is not None:
                _verify_retry(previous, operation_digest)
                return web.json_response(previous.response)
            if payload["binding_ref"] in self.bindings:
                raise ApiError(409, "binding-conflict", "binding_ref is already in use")
            record = BindingRecord(
                activation_key=payload["activation_idempotency_key"],
                owner_ref=payload["owner_ref"],
                binding_ref=payload["binding_ref"],
                session_id=payload["compute_service_session_id"],
                instance_id=payload["compute_instance_id"],
                configuration=configuration,
                configuration_digest=expected_digest,
                participants=participants,
                participant_facts=participant_facts,
            )
            self.media_engine.ensure_binding(record.binding_ref)
            self.bindings[record.binding_ref] = record
            response = _binding_response(record)
            self.activation_ops[op_key] = StoredOperation(operation_digest, response)
        return web.json_response(response)

    async def get_binding(self, request: web.Request) -> web.Response:
        self._authorize_management(request)
        activation_key = request.query.get("activation_idempotency_key", "")
        if not activation_key:
            raise ApiError(
                400,
                "invalid-request",
                "activation_idempotency_key is required",
            )
        async with self.lock:
            record = self.bindings.get(request.match_info["binding_ref"])
            if record is None or record.activation_key != activation_key:
                raise ApiError(404, "binding-not-found", "binding does not exist")
            response = _binding_response(record)
        return web.json_response(response)

    async def get_media_state(self, request: web.Request) -> web.Response:
        self._authorize_management(request)
        async with self.lock:
            record = self.bindings.get(request.match_info["binding_ref"])
            if record is None:
                raise ApiError(404, "binding-not-found", "binding does not exist")
            response = _media_state_response(record)
        return web.json_response(response)

    async def unbind(self, request: web.Request) -> web.Response:
        self._authorize_management(request)
        payload = await _json_object(request)
        _require_strings(payload, ("owner_ref", "unbind_idempotency_key", "cause"))
        binding_ref = request.match_info["binding_ref"]
        op_key = f"{payload['owner_ref']}\0{payload['unbind_idempotency_key']}"
        operation_digest = canonical_digest(
            {"binding_ref": binding_ref, "input": payload}
        )
        async with self.lock:
            previous = self.unbind_ops.get(op_key)
            if previous is not None:
                _verify_retry(previous, operation_digest)
                return web.json_response(previous.response)
            record = self.bindings.get(binding_ref)
            if record is None:
                raise ApiError(404, "binding-not-found", "binding does not exist")
            if record.owner_ref != payload["owner_ref"]:
                raise ApiError(403, "owner-mismatch", "binding belongs to another owner")
            record.state = "UNBINDING"

        try:
            await self.media_engine.close_binding(binding_ref)
        except Exception as exc:
            raise ApiError(503, "unbind-incomplete", str(exc)) from exc

        async with self.lock:
            record.state = "UNBOUND"
            record.initialized = False
            record.cause = payload["cause"]
            record.connected = {"producer": False, "consumer": False}
            record.revision += 1
            record.observed_at = _now()
            for media in self.media.values():
                if media.context["binding_ref"] == binding_ref:
                    media.closed = True
            self.targets.pop(binding_ref, None)
            for action in self.actions.values():
                if (
                    action.context["binding_ref"] == binding_ref
                    and action.status in {"ACCEPTED", "RUNNING"}
                ):
                    action.status = "CANCELLED"
                    action.cause = payload["cause"]
                    task = self.action_tasks.get(action.action_id)
                    if task is not None:
                        task.cancel()
            response = _binding_response(record)
            self.unbind_ops[op_key] = StoredOperation(operation_digest, response)
        return web.json_response(response)

    async def create_media_connection(self, request: web.Request) -> web.Response:
        payload = await _json_object(request)
        _require_strings(payload, ("request_id",))
        context = _context(payload.get("computing_context"))
        offer = payload.get("offer")
        if (
            not isinstance(offer, dict)
            or offer.get("type") != "offer"
            or not isinstance(offer.get("sdp"), str)
            or not offer["sdp"].strip()
        ):
            raise ApiError(400, "invalid-request", "a complete SDP Offer is required")
        operation_digest = canonical_digest(payload)
        op_key = f"{context['binding_ref']}\0{context['role']}\0{payload['request_id']}"
        role_key = (context["binding_ref"], context["role"])
        async with self.lock:
            previous = self.media_ops.get(op_key)
            if previous is not None:
                _verify_retry(previous, operation_digest)
                return web.json_response(previous.response, status=201)
            self._validate_context(context)
            if role_key in self.media_pending or any(
                not item.closed
                and item.context["binding_ref"] == context["binding_ref"]
                and item.context["role"] == context["role"]
                for item in self.media.values()
            ):
                raise ApiError(
                    409,
                    "media-connection-exists",
                    "role already has an active media connection",
                )
            self.media_pending.add(role_key)

        async def state_callback(role: str, connected: bool, cause: str) -> None:
            await self._observe_media_state(
                context["binding_ref"], role, connected, cause
            )

        try:
            handle, answer = await self.media_engine.create_standard_media_connection(
                context["binding_ref"],
                context["role"],
                offer["sdp"],
                state_callback,
            )
        except ApiError:
            raise
        except ValueError as exc:
            raise ApiError(400, "invalid-sdp", str(exc)) from exc
        except Exception as exc:
            raise ApiError(503, "media-setup-failed", str(exc)) from exc
        finally:
            async with self.lock:
                self.media_pending.discard(role_key)

        media_connection_id = f"media-connection-{uuid4().hex}"
        response = {
            "request_id": payload["request_id"],
            "computing_context": context,
            "media_connection_id": media_connection_id,
            "answer": answer,
        }
        resource = MediaResource(
            request_id=payload["request_id"],
            context=context,
            media_connection_id=media_connection_id,
            digest=operation_digest,
            answer=answer,
            handle=handle,
        )
        async with self.lock:
            self.media[media_connection_id] = resource
            self.media_ops[op_key] = StoredOperation(operation_digest, response)
        return web.json_response(response, status=201)

    async def delete_media_connection(self, request: web.Request) -> web.Response:
        media_connection_id = request.match_info["media_connection_id"]
        async with self.lock:
            resource = self.media.get(media_connection_id)
            if resource is None:
                raise ApiError(
                    404,
                    "media-connection-not-found",
                    "media connection does not exist",
                )
            if resource.closed:
                return web.Response(status=204)
        await self.media_engine.close_standard_media_connection(
            resource.context["binding_ref"], resource.handle
        )
        async with self.lock:
            resource.closed = True
            self._set_media_state_locked(
                resource.context["binding_ref"],
                resource.context["role"],
                False,
                "",
            )
        return web.Response(status=204)

    async def update_recognition_target(self, request: web.Request) -> web.Response:
        payload = await _json_object(request)
        _require_strings(payload, ("request_id",))
        context = _context(payload.get("computing_context"))
        input_value = payload.get("input")
        if (
            not isinstance(input_value, dict)
            or input_value.get("type") != "TEXT"
            or not isinstance(input_value.get("text"), str)
            or not input_value["text"].strip()
        ):
            raise ApiError(400, "invalid-request", "non-empty TEXT input is required")
        if (
            context["role"] != "consumer"
            or context["compute_service_session_id"]
            != request.match_info["compute_service_session_id"]
        ):
            raise ApiError(
                409,
                "binding-mismatch",
                "recognition target requires matching consumer context",
            )
        operation_digest = canonical_digest(payload)
        op_key = f"{context['binding_ref']}\0{payload['request_id']}"
        async with self.lock:
            previous = self.target_ops.get(op_key)
            if previous is not None:
                _verify_retry(previous, operation_digest)
                return web.json_response(previous.response)
            self._validate_context(context)

        label, prompt = await self._recognition_target(input_value["text"])
        self.media_engine.set_recognition_target(context["binding_ref"], prompt)
        async with self.lock:
            previous_target = self.targets.get(context["binding_ref"])
            revision = previous_target.revision + 1 if previous_target else 1
            target = RecognitionTarget(
                request_id=payload["request_id"],
                context=context,
                revision=revision,
                label=label,
                prompt=prompt,
            )
            self.targets[context["binding_ref"]] = target
            response = _target_response(target)
            self.target_ops[op_key] = StoredOperation(operation_digest, response)
        return web.json_response(response)

    async def get_recognition_target(self, request: web.Request) -> web.Response:
        session_id = request.match_info["compute_service_session_id"]
        async with self.lock:
            target = next(
                (
                    item
                    for item in self.targets.values()
                    if item.context["compute_service_session_id"] == session_id
                ),
                None,
            )
            if target is None:
                raise ApiError(
                    404,
                    "recognition-target-not-set",
                    "recognition target is not set",
                )
            response = _target_response(target)
        return web.json_response(response)

    async def create_control_action(self, request: web.Request) -> web.Response:
        raise ApiError(
            410,
            "control-actions-disabled",
            "Sandbox no longer sends machine-dog control actions",
        )

    async def create_audio_control_action(self, request: web.Request) -> web.Response:
        """Compatibility endpoint for bound sessions; it has no control side effect."""
        fields, audio, filename, content_type = await self._audio_upload(request)
        request_id = fields.get("request_id", "").strip()
        try:
            context_value = json.loads(fields.get("computing_context", ""))
        except (TypeError, ValueError) as exc:
            raise ApiError(
                400,
                "invalid-request",
                "computing_context must be a JSON object",
            ) from exc
        context = _context(context_value)
        async with self.lock:
            self._validate_context(context)
        if context["role"] != "consumer":
            raise ApiError(
                409,
                "binding-mismatch",
                "audio control action must use the bound consumer context",
            )
        if not request_id:
            raise ApiError(400, "invalid-request", "request_id is required")
        if not self.media_engine.settings.asr_url:
            raise ApiError(503, "asr-unavailable", "ASR endpoint is not configured")

        if self.http is None:
            self.http = ClientSession(timeout=ClientTimeout(total=120))
        upload = FormData()
        upload.add_field("request_id", request_id)
        upload.add_field("language", fields.get("language", "zh"))
        upload.add_field(
            "file",
            audio,
            filename=filename,
            content_type=content_type or "application/octet-stream",
        )
        try:
            async with self.http.post(
                self.media_engine.settings.asr_url,
                data=upload,
            ) as response:
                asr_body = await response.json(content_type=None)
                if response.status != 200:
                    raise ApiError(502, "asr-failed", "ASR transcription failed")
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(503, "asr-unavailable", "ASR service is unavailable") from exc
        transcript = asr_body.get("text") if isinstance(asr_body, dict) else None
        if not isinstance(transcript, str) or not transcript.strip():
            raise ApiError(422, "empty-transcription", "ASR returned no usable text")
        intent = asr_body.get("intent") if isinstance(asr_body, dict) else None
        if not isinstance(intent, dict):
            raise ApiError(502, "asr-failed", "ASR returned an invalid intent")
        response = {
            "request_id": request_id,
            "text": transcript.strip(),
            "intent": intent,
        }
        LOGGER.info(
            "runtime audio recognized request_id=%s text=%s intent=%s",
            request_id,
            json.dumps(transcript.strip(), ensure_ascii=False),
            json.dumps(intent, ensure_ascii=False, separators=(",", ":")),
        )
        return web.json_response(response)

    async def _create_control_action(
        self,
        payload: dict[str, Any],
        *,
        operation_digest: str | None = None,
        transcription: dict[str, Any] | None = None,
    ) -> web.Response:
        _require_strings(payload, ("request_id",))
        context = _context(payload.get("computing_context"))
        operation_digest = operation_digest or canonical_digest(payload)
        op_key = f"{context['binding_ref']}\0{payload['request_id']}"
        async with self.lock:
            previous = self.control_ops.get(op_key)
            if previous is not None:
                _verify_retry(previous, operation_digest)
                action_id = previous.response["action_id"]
                return web.json_response(
                    _action_response(self.actions[action_id]), status=202
                )
            binding = self._validate_context(context)
            if context["role"] != "consumer":
                raise ApiError(
                    409,
                    "binding-mismatch",
                    "control action must use the bound consumer context",
                )
            _validate_action_target(binding, payload.get("target"))

        action, parameters = await self._control_action(payload)
        _validate_action_role(action, payload.get("target"))
        control = ControlAction(
            request_id=payload["request_id"],
            context=context,
            action_id=f"action-{uuid4().hex}",
            status="ACCEPTED",
            normalized_action=action,
            normalized_parameters=parameters,
            result=None,
            cause="",
            digest=operation_digest,
            control_triggered=True,
            transcription=transcription,
        )
        response = _action_response(control)
        async with self.lock:
            self.actions[control.action_id] = control
            self.control_ops[op_key] = StoredOperation(operation_digest, response)
            task = asyncio.create_task(
                self._execute_control_action(control, binding),
                name=f"control-action:{control.action_id}",
            )
            self.action_tasks[control.action_id] = task
            task.add_done_callback(
                lambda _: self.action_tasks.pop(control.action_id, None)
            )
        return web.json_response(response, status=202)

    async def _audio_upload(
        self, request: web.Request
    ) -> tuple[dict[str, str], bytes, str, str]:
        if not request.content_type.startswith("multipart/"):
            raise ApiError(415, "unsupported-media-type", "multipart/form-data is required")
        reader = await request.multipart()
        fields: dict[str, str] = {}
        audio: bytes | None = None
        filename = ""
        content_type = ""
        async for field in reader:
            if field.name != "file" or not field.filename:
                fields[str(field.name)] = await field.text()
                continue
            if audio is not None:
                raise ApiError(400, "invalid-request", "only one audio file is allowed")
            filename = Path(field.filename).name
            if Path(filename).suffix.lower() not in {
                ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"
            }:
                raise ApiError(400, "invalid-audio", "unsupported audio type")
            content_type = field.headers.get("Content-Type", "")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = await field.read_chunk(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > self.media_engine.settings.audio_max_upload_bytes:
                    raise ApiError(413, "audio-too-large", "audio upload exceeds configured limit")
                chunks.append(chunk)
            audio = b"".join(chunks)
        if not audio:
            raise ApiError(400, "invalid-audio", "non-empty file field is required")
        return fields, audio, filename, content_type

    async def get_control_action(self, request: web.Request) -> web.Response:
        raise ApiError(
            410,
            "control-actions-disabled",
            "Sandbox no longer sends machine-dog control actions",
        )

    async def _execute_control_action(
        self,
        control: ControlAction,
        binding: BindingRecord,
    ) -> None:
        async with self.lock:
            if control.status == "CANCELLED":
                return
            control.status = "RUNNING"
        try:
            if control.normalized_action == "search_object":
                query = str(
                    control.normalized_parameters.get("query")
                    or ""
                ).strip()
                if not query:
                    raise ApiError(422, "invalid-control-action", "query is required")
                # TEXT actions are normalized by _control_action; STRUCTURED
                # queries are already caller-provided model prompts.
                prompt = query
                timeout_ms = control.normalized_parameters.get("timeout_ms", 10000)
                if (
                    not isinstance(timeout_ms, int)
                    or isinstance(timeout_ms, bool)
                    or not 1 <= timeout_ms <= 120000
                ):
                    raise ApiError(
                        422,
                        "invalid-control-action",
                        "timeout_ms must be an integer between 1 and 120000",
                    )
                matches = await self.media_engine.search_object(
                    binding.binding_ref, prompt, timeout_ms
                )
                status = "COMPLETED"
                result: dict[str, Any] | None = {
                    "query": prompt,
                    "matches": matches,
                }
                cause = ""
            else:
                endpoint_context = dict(binding.participant_facts["producer"])
                endpoint_context.update(
                    {
                        "binding_ref": binding.binding_ref,
                        "compute_service_session_id": binding.session_id,
                        "compute_instance_id": binding.instance_id,
                    }
                )
                outcome = await self.control_adapter.execute(
                    endpoint_context=endpoint_context,
                    action_id=control.action_id,
                    action=control.normalized_action,
                    parameters=control.normalized_parameters,
                )
                status, result, cause = (
                    outcome.status,
                    outcome.result,
                    outcome.cause,
                )
        except asyncio.CancelledError:
            return
        except ApiError as exc:
            status, result, cause = "FAILED", None, exc.code
        except Exception:
            status, result, cause = "FAILED", None, "action-execution-failed"

        async with self.lock:
            if control.status != "CANCELLED":
                control.status = status
                control.result = result
                control.cause = cause

    async def _observe_media_state(
        self,
        binding_ref: str,
        role: str,
        connected: bool,
        cause: str,
    ) -> None:
        async with self.lock:
            self._set_media_state_locked(binding_ref, role, connected, cause)

    def _set_media_state_locked(
        self,
        binding_ref: str,
        role: str,
        connected: bool,
        cause: str,
    ) -> None:
        record = self.bindings.get(binding_ref)
        if record is None or role not in record.connected:
            return
        if record.connected[role] == connected and record.cause == cause:
            return
        record.connected[role] = connected
        record.cause = cause
        record.revision += 1
        record.observed_at = _now()

    def _validate_context(self, context: dict[str, str]) -> BindingRecord:
        record = self.bindings.get(context["binding_ref"])
        if record is None or record.state != "BOUND" or not record.initialized:
            raise ApiError(409, "binding-mismatch", "active binding does not exist")
        if (
            context["compute_service_session_id"] != record.session_id
            or context["compute_instance_id"] != record.instance_id
            or record.participants.get(context["role"]) != context["agent_id"]
        ):
            raise ApiError(
                409,
                "binding-mismatch",
                "computing_context does not match the binding",
            )
        return record

    async def _recognition_target(self, text: str) -> tuple[str, str]:
        classified = await self._classify(text)
        if classified["intent"] != "find_object" or not classified["argument"]:
            raise ApiError(
                422,
                "invalid-recognition-target",
                "text cannot be resolved to a visual target",
            )
        label = text.strip()
        prompt = classified["argument"]
        i18n = classified.get("argument_i18n")
        if isinstance(i18n, dict):
            label = str(i18n.get("zh") or label)
            prompt = str(i18n.get("en") or prompt)
        return label, prompt

    async def _control_action(
        self,
        payload: dict[str, Any],
        classified: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        input_value = payload.get("input")
        if not isinstance(input_value, dict):
            raise ApiError(400, "invalid-request", "input must be an object")
        input_type = input_value.get("type")
        if input_type == "STRUCTURED":
            action = payload.get("action")
            parameters = payload.get("parameters")
            if action not in {"movement", "grab", "search_object"}:
                raise ApiError(422, "invalid-control-action", "unsupported action")
            if not isinstance(parameters, dict):
                raise ApiError(400, "invalid-request", "parameters are required")
            return action, parameters
        if input_type == "TEXT" and isinstance(input_value.get("text"), str):
            classified = classified or await self._classify(input_value["text"])
            intent = classified["intent"]
            action = "search_object" if intent == "find_object" else intent
            if action not in {"movement", "grab", "search_object"}:
                raise ApiError(
                    422,
                    "invalid-control-action",
                    "text action is ambiguous",
                )
            argument = classified["argument"] or input_value["text"]
            if action == "movement":
                direction = _normalize_direction(argument)
                if direction is None:
                    raise ApiError(
                        422,
                        "invalid-control-action",
                        "movement text cannot be resolved to a direction",
                    )
                return action, {"direction": direction}
            if action == "grab":
                return action, {"object": argument}
            return action, {"query": argument}
        raise ApiError(
            400,
            "invalid-request",
            "input.type must be TEXT or STRUCTURED",
        )

    async def _classify(self, text: str) -> dict[str, Any]:
        if self.intent_url:
            try:
                if self.http is None:
                    self.http = ClientSession(timeout=ClientTimeout(total=10))
                async with self.http.post(self.intent_url, json={"text": text}) as response:
                    if response.status == 200:
                        payload = await response.json()
                        intent = payload.get("intent") or payload.get("scene")
                        argument = payload.get("normalized_argument") or payload.get(
                            "argument"
                        )
                        if isinstance(intent, str) and isinstance(argument, str):
                            return {
                                "intent": intent,
                                "argument": argument,
                                "argument_i18n": payload.get(
                                    "normalized_argument_i18n"
                                ),
                            }
            except Exception:
                # The deterministic rules keep the N6 API usable during model
                # warm-up; health endpoints still expose the Intent failure.
                pass
        fallback = self.intent.classify(text)
        return {
            "intent": fallback.intent,
            "argument": fallback.argument,
            "argument_i18n": fallback.argument_i18n,
        }


def _normalize_direction(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return {
        "forward": "forward",
        "move_forward": "forward",
        "back": "backward",
        "backward": "backward",
        "move_back": "backward",
        "left": "left",
        "turn_left": "left",
        "right": "right",
        "turn_right": "right",
        "wave": "wave",
    }.get(normalized)


def _classification_from_public_intent(intent: dict[str, Any]) -> dict[str, str]:
    public_name = str(intent.get("intent") or "other").strip().lower()
    scene = {
        "security patrol": "patrol",
        "movement": "movement",
        "find object": "find_object",
        "grab": "grab",
        "other": "other",
    }.get(public_name, "other")
    if scene == "patrol":
        argument = str(intent.get("area") or "")
    elif scene == "movement":
        argument = str(intent.get("direction") or "")
    else:
        argument = str(intent.get("object") or "")
    return {"intent": scene, "argument": argument}


def _public_intent(classified: dict[str, Any], backend: str) -> dict[str, Any]:
    scene = str(classified.get("intent") or "other")
    argument = str(classified.get("argument") or "")
    result: dict[str, Any] = {
        "executor": "robot dog" if scene != "other" else None,
        "intent": {
            "patrol": "security patrol",
            "movement": "movement",
            "find_object": "find object",
            "grab": "grab",
            "other": "other",
        }.get(scene, scene),
    }
    if scene == "patrol":
        result["area"] = argument.removesuffix("区域").strip()
    elif scene == "movement":
        result["direction"] = argument
    elif scene in {"find_object", "grab"}:
        result["object"] = argument
    result["matched"] = scene != "other"
    result["backend"] = backend
    return result


async def _json_object(request: web.Request) -> dict[str, Any]:
    try:
        payload = json.loads(
            await request.text(),
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except Exception as exc:
        raise ApiError(400, "invalid-request", "valid JSON object is required") from exc
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid-request", "JSON body must be an object")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> Any:
    raise ValueError(f"non-canonical JSON number: {value}")


def _require_strings(payload: dict[str, Any], names: tuple[str, ...]) -> None:
    missing = [
        name
        for name in names
        if not isinstance(payload.get(name), str) or not payload[name].strip()
    ]
    if missing:
        raise ApiError(
            400,
            "invalid-request",
            f"required string fields are missing: {', '.join(missing)}",
        )


def _context(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ApiError(400, "invalid-request", "computing_context is required")
    fields = (
        "compute_service_session_id",
        "compute_instance_id",
        "binding_ref",
        "role",
        "agent_id",
    )
    _require_strings(value, fields)
    if value["role"] not in {"producer", "consumer"}:
        raise ApiError(400, "invalid-role", "role must be producer or consumer")
    return {name: value[name] for name in fields}


def _participant_facts(
    configuration: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    values = configuration.get("participants_network_facts")
    if not isinstance(values, list):
        raise ApiError(
            400,
            "invalid-configuration",
            "participants_network_facts must be an array",
        )
    result: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ApiError(400, "invalid-configuration", "participant must be an object")
        role = item.get("role")
        agent_id = item.get("agent_id")
        required_strings = (
            "agent_id",
            "up_session_id",
            "ue_ipv4_address",
            "dnn",
            "snssai",
        )
        generation = item.get("generation")
        if (
            role not in {"producer", "consumer"}
            or any(
                not isinstance(item.get(name), str) or not item[name].strip()
                for name in required_strings
            )
            or not isinstance(item.get("pdu_session_id"), int)
            or isinstance(item.get("pdu_session_id"), bool)
            or item["pdu_session_id"] <= 0
            or not (
                (isinstance(generation, int) and not isinstance(generation, bool) and generation >= 0)
                or (isinstance(generation, str) and generation.isdigit())
            )
            or role in result
        ):
            raise ApiError(
                400,
                "invalid-configuration",
                "producer and consumer participants must be unique",
            )
        result[role] = dict(item)
    if set(result) != {"producer", "consumer"}:
        raise ApiError(
            400,
            "invalid-configuration",
            "producer and consumer participants are required",
        )
    return result


def _connection_parameters(configuration: dict[str, Any]) -> None:
    value = configuration.get("connection_parameters")
    if not isinstance(value, dict):
        raise ApiError(
            400,
            "invalid-configuration",
            "connection_parameters must be an object",
        )
    if (
        value.get("transport") != "WEBRTC"
        or value.get("media_connections_path") != "/v1/media-connections"
        or value.get("recognition_target_path_template")
        != "/v1/recognition-targets/{compute_service_session_id}"
    ):
        raise ApiError(
            400,
            "invalid-configuration",
            "dog-vision requires the standard WEBRTC and recognition paths",
        )


def canonical_digest(value: Any) -> str:
    _reject_noncanonical_numbers(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_noncanonical_numbers(value: Any) -> None:
    if isinstance(value, float):
        raise ApiError(400, "invalid-request", "floating-point values are not allowed")
    if isinstance(value, dict):
        for nested in value.values():
            _reject_noncanonical_numbers(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_noncanonical_numbers(nested)


def _verify_retry(operation: StoredOperation, digest: str) -> None:
    if operation.digest != digest:
        raise ApiError(
            409,
            "idempotency-conflict",
            "idempotency key was used with different content",
        )


def _binding_response(record: BindingRecord) -> dict[str, Any]:
    return {
        "activation_idempotency_key": record.activation_key,
        "state": record.state,
        "binding_ref": record.binding_ref,
        "compute_service_session_id": record.session_id,
        "compute_instance_id": record.instance_id,
        "configuration_digest": record.configuration_digest,
        "initialized": record.initialized,
        "cause": record.cause,
    }


def _media_state_response(record: BindingRecord) -> dict[str, Any]:
    return {
        "compute_service_session_id": record.session_id,
        "compute_instance_id": record.instance_id,
        "binding_ref": record.binding_ref,
        "revision": str(record.revision),
        "producer_connected": record.connected["producer"],
        "consumer_connected": record.connected["consumer"],
        "observed_at": record.observed_at.isoformat().replace("+00:00", "Z"),
        "cause": record.cause,
    }


def _target_response(target: RecognitionTarget) -> dict[str, Any]:
    return {
        "request_id": target.request_id,
        "computing_context": target.context,
        "status": "APPLIED",
        "target_revision": str(target.revision),
        "target": {"label": target.label, "prompt": target.prompt},
    }


def _action_response(action: ControlAction) -> dict[str, Any]:
    response: dict[str, Any] = {
        "request_id": action.request_id,
        "computing_context": action.context,
        "action_id": action.action_id,
        "status": action.status,
        "normalized_action": action.normalized_action,
        "normalized_parameters": action.normalized_parameters,
        "cause": action.cause,
        "control_triggered": action.control_triggered,
    }
    if action.result is not None:
        response["result"] = action.result
    if action.transcription is not None:
        response["transcription"] = action.transcription
    if action.intent is not None:
        response["intent"] = action.intent
    return response


def _validate_action_target(record: BindingRecord, target: Any) -> None:
    if target is None:
        return
    if not isinstance(target, dict) or target.get("role") not in {
        "producer",
        "sandbox",
    }:
        raise ApiError(409, "binding-mismatch", "control target is invalid")
    if target["role"] == "producer" and target.get("agent_id") != record.participants[
        "producer"
    ]:
        raise ApiError(
            409,
            "binding-mismatch",
            "control target does not match the producer",
        )
    if target["role"] == "sandbox" and target.get("agent_id"):
        raise ApiError(
            409,
            "binding-mismatch",
            "sandbox target must not contain agent_id",
        )


def _validate_action_role(action: str, target: Any) -> None:
    if target is None:
        return
    expected = "sandbox" if action == "search_object" else "producer"
    if target["role"] != expected:
        raise ApiError(
            409,
            "binding-mismatch",
            f"{action} must target {expected}",
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)
