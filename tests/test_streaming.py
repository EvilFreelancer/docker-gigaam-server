import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, Mock, patch

from starlette.requests import ClientDisconnect

from tests._support import ApiTestCase, RESULT


class StreamingTests(ApiTestCase):
    def test_response_start_failure_removes_upload_before_body_iteration(self):
        response = self.api._stream_response(
            self.engine, "/tmp/sample.wav", "gigaam-v3-e2e-rnnt", False
        )
        unlink = Mock()

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                raise OSError("client disconnected before response body")

        with patch.object(self.api.os, "unlink", unlink):
            with self.assertRaises(ClientDisconnect):
                asyncio.run(
                    response(
                        {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}},
                        receive,
                        send,
                    )
                )
        unlink.assert_called_once_with("/tmp/sample.wav")

    def test_stream_returns_openai_sse_event_types_and_done_marker(self):
        response = self.post_transcription({"model": "gigaam", "stream": "true"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        self.assertEqual([event["type"] for event in events], [
            "transcript.text.delta", "transcript.text.done"
        ])
        self.assertEqual(events[-1]["text"], "hello world")
        self.assertEqual(set(events[-1]), {"type", "text"})
        self.assertIn("data: [DONE]", response.text)

    def test_stream_emits_delta_before_transcription_finishes(self):
        finished = threading.Event()

        def transcribe(
            audio_path, model_name, word_timestamps=False, on_segment=None, cancel_event=None
        ):
            on_segment(RESULT.segments[0])
            time.sleep(0.15)
            finished.set()
            return RESULT

        with (
            patch.object(self.api, "engine", self.engine),
            patch.object(self.api, "_save_upload", new=AsyncMock(return_value="/tmp/sample.wav")),
            patch.object(self.api.os, "unlink"),
            patch.object(self.engine, "transcribe", side_effect=transcribe),
        ):
            with self.client.stream(
                "POST",
                "/v1/audio/transcriptions",
                data={"model": "gigaam", "stream": "true"},
                files={"file": ("sample.wav", b"fake audio", "audio/wav")},
            ) as response:
                self.assertEqual(response.status_code, 200)
                lines = iter(response.iter_lines())
                while True:
                    line = next(lines)
                    if line.startswith("data: {"):
                        event = json.loads(line[6:])
                        break
                self.assertEqual(event["type"], "transcript.text.delta")
                self.assertFalse(finished.is_set())

    def test_stream_decode_error_is_sanitized_as_an_openai_sse_error(self):
        with (
            patch.object(self.api, "engine", self.engine),
            patch.object(self.api, "_save_upload", new=AsyncMock(return_value="/tmp/sample.wav")),
            patch.object(self.api.os, "unlink"),
            patch.object(
                self.engine,
                "transcribe",
                side_effect=self.api.AudioDecodeError("private decoder details"),
            ),
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"model": "gigaam", "stream": "true"},
                files={"file": ("sample.wav", b"fake audio", "audio/wav")},
            )
        errors = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(errors[0]["error"]["type"], "invalid_request_error")
        self.assertEqual(errors[0]["error"]["param"], "file")
        self.assertNotIn("private decoder details", response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_client_disconnect_signals_asr_worker_cancellation(self):
        worker_cancelled = threading.Event()
        unlink = Mock()

        def transcribe(audio_path, model_name, word_timestamps=False, on_segment=None, cancel_event=None):
            on_segment(RESULT.segments[0])
            if cancel_event.wait(timeout=2):
                worker_cancelled.set()
            return RESULT

        with (
            patch.object(self.api, "engine", self.engine),
            patch.object(self.api, "_save_upload", new=AsyncMock(return_value="/tmp/sample.wav")),
            patch.object(self.api.os, "unlink", unlink),
            patch.object(self.engine, "transcribe", side_effect=transcribe),
        ):
            with self.client.stream(
                "POST",
                "/v1/audio/transcriptions",
                data={"model": "gigaam", "stream": "true"},
                files={"file": ("sample.wav", b"fake audio", "audio/wav")},
            ) as response:
                self.assertEqual(response.status_code, 200)
                lines = iter(response.iter_lines())
                while True:
                    line = next(lines)
                    if line.startswith("data: {"):
                        self.assertEqual(json.loads(line[6:])["type"], "transcript.text.delta")
                        break
        self.assertTrue(worker_cancelled.wait(timeout=1))
        unlink.assert_called_once_with("/tmp/sample.wav")
