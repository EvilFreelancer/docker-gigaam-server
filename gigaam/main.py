"""OpenAI-compatible speech-to-text API on top of GigaAM.

Implements the surface the official clients use:

    POST /v1/audio/transcriptions   multipart, response_format json|text|srt|vtt|verbose_json,
                                    timestamp_granularities[], stream=true (SSE)
    POST /v1/audio/translations     rejected with an OpenAI-shaped error (GigaAM does not translate)
    GET  /v1/models, /v1/models/{id}
    GET  /health

Errors always come back in the OpenAI envelope so client-side error handling works unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from asr import AsrEngine, Transcription

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("gigaam.api")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


CHECKPOINTS = [
    name.strip()
    for name in os.getenv("GIGAAM_MODELS", "v3_e2e_rnnt").split(",")
    if name.strip()
]
DEVICE = os.getenv("GIGAAM_DEVICE", "cuda")
CACHE_DIR = os.getenv("GIGAAM_CACHE_DIR", "/models")
API_KEY = os.getenv("GIGAAM_API_KEY", "").strip()
LANGUAGE = os.getenv("GIGAAM_LANGUAGE", "ru").strip()
MAX_FILE_MB = float(os.getenv("GIGAAM_MAX_FILE_MB", "1024"))
OWNED_BY = os.getenv("GIGAAM_OWNED_BY", "salute-developers")

RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}

engine: Optional[AsrEngine] = None
_started_at = int(time.time())


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = AsrEngine(
        checkpoints=CHECKPOINTS,
        device=DEVICE,
        fp16_encoder=_env_bool("GIGAAM_FP16", True),
        cache_dir=CACHE_DIR,
        batch_size=int(os.getenv("GIGAAM_BATCH_SIZE", "8")),
        max_chunk_s=float(os.getenv("GIGAAM_MAX_CHUNK_S", "22")),
        max_gap_s=float(os.getenv("GIGAAM_VAD_MAX_GAP_S", "0.6")),
        vad_threshold=float(os.getenv("GIGAAM_VAD_THRESHOLD", "0.5")),
        vad_min_silence_ms=int(os.getenv("GIGAAM_VAD_MIN_SILENCE_MS", "300")),
        vad_speech_pad_ms=int(os.getenv("GIGAAM_VAD_SPEECH_PAD_MS", "200")),
    )
    yield


app = FastAPI(title="GigaAM ASR", version="1.0.0", docs_url="/docs", lifespan=lifespan)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def _error(
    status: int,
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


class ApiError(HTTPException):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error", param: Optional[str] = None):
        super().__init__(status_code=status, detail=message)
        self.err_type = err_type
        self.param = param


@app.exception_handler(ApiError)
async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return _error(exc.status_code, str(exc.detail), exc.err_type, exc.param)


@app.exception_handler(HTTPException)
async def _http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    param = ".".join(str(part) for part in first.get("loc", [])[1:]) or None
    return _error(400, first.get("msg", "invalid request"), param=param)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def _require_engine() -> AsrEngine:
    if engine is None:
        raise ApiError(503, "model is still loading", err_type="server_error")
    return engine


def _check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise ApiError(
            401,
            "Incorrect API key provided.",
            err_type="invalid_request_error",
            param=None,
        )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _model_card(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": _started_at,
        "owned_by": OWNED_BY,
    }


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)) -> dict:
    _check_auth(authorization)
    api = _require_engine()
    return {"object": "list", "data": [_model_card(m) for m in api.models]}


@app.get("/v1/models/{model_id:path}")
def retrieve_model(model_id: str, authorization: Optional[str] = Header(None)) -> dict:
    _check_auth(authorization)
    api = _require_engine()
    resolved = api.resolve(model_id)
    if resolved is None:
        raise ApiError(404, f"The model '{model_id}' does not exist", param="model")
    return _model_card(resolved)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if engine is not None else "loading",
        "models": list(engine.models) if engine else [],
        "device": DEVICE,
    }


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _timestamp(seconds: float, millis_sep: str = ",") -> str:
    hours, rest = divmod(max(seconds, 0.0), 3600)
    minutes, secs = divmod(rest, 60)
    whole = int(secs)
    millis = int(round((secs - whole) * 1000))
    if millis == 1000:  # rounding carry
        whole += 1
        millis = 0
    return f"{int(hours):02d}:{int(minutes):02d}:{whole:02d}{millis_sep}{millis:03d}"


def _srt(result: Transcription) -> str:
    lines = []
    for number, segment in enumerate(result.segments, start=1):
        lines.append(str(number))
        lines.append(f"{_timestamp(segment.start)} --> {_timestamp(segment.end)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)


def _vtt(result: Transcription) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        lines.append(f"{_timestamp(segment.start, '.')} --> {_timestamp(segment.end, '.')}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)


def _compression_ratio(text: str) -> float:
    """Whisper reports this per segment; approximate it so the field is not a lie."""
    import zlib

    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return round(len(raw) / len(zlib.compress(raw)), 3)


def _verbose_json(result: Transcription, granularities: Iterable[str]) -> dict:
    wanted = set(granularities) or {"segment"}
    payload = {
        "task": "transcribe",
        "language": LANGUAGE,
        "duration": result.duration,
        "text": result.text,
        "usage": _usage(result),
    }
    if "segment" in wanted:
        payload["segments"] = [
            {
                "id": index,
                "seek": int(segment.start * 100),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": _compression_ratio(segment.text),
                "no_speech_prob": 0.0,
            }
            for index, segment in enumerate(result.segments)
        ]
    if "word" in wanted:
        payload["words"] = [
            {"word": word.text, "start": word.start, "end": word.end}
            for segment in result.segments
            for word in segment.words
        ]
    return payload


def _usage(result: Transcription) -> dict:
    """OpenAI bills duration in whole seconds and LiteLLM parses the field as an int -
    a fractional value there fails validation on the proxy side."""
    return {"type": "duration", "seconds": int(round(result.duration))}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_response(result: Transcription) -> StreamingResponse:
    """SSE in the shape openai-python expects for streamed transcriptions."""

    def events():
        for index, segment in enumerate(result.segments):
            delta = segment.text if index == 0 else " " + segment.text
            yield _sse({"type": "transcript.text.delta", "delta": delta, "logprobs": []})
        yield _sse(
            {
                "type": "transcript.text.done",
                "text": result.text,
                "logprobs": [],
                "usage": _usage(result),
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------


async def _save_upload(upload: UploadFile) -> str:
    limit = int(MAX_FILE_MB * 1024 * 1024)
    suffix = Path(upload.filename or "audio").suffix or ".bin"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    written = 0
    try:
        while True:
            block = await upload.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > limit:
                raise ApiError(
                    413,
                    f"File is too large, the limit is {MAX_FILE_MB:g} MB.",
                    param="file",
                )
            handle.write(block)
    except BaseException:
        handle.close()
        os.unlink(handle.name)
        raise
    handle.close()
    if written == 0:
        os.unlink(handle.name)
        raise ApiError(400, "Uploaded file is empty.", param="file")
    return handle.name


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
    stream: bool = Form(False),
    granularities: Optional[List[str]] = Form(None, alias="timestamp_granularities[]"),
    granularities_plain: Optional[List[str]] = Form(None, alias="timestamp_granularities"),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    api = _require_engine()

    if response_format not in RESPONSE_FORMATS:
        raise ApiError(
            400,
            f"Invalid value: '{response_format}'. Supported values are: "
            + ", ".join(sorted(RESPONSE_FORMATS))
            + ".",
            param="response_format",
        )

    resolved = api.resolve(model)
    if resolved is None:
        raise ApiError(404, f"The model '{model}' does not exist", param="model")

    wanted = set(granularities or []) | set(granularities_plain or [])
    unknown = wanted - {"word", "segment"}
    if unknown:
        raise ApiError(
            400,
            f"Invalid value: '{sorted(unknown)[0]}'. Supported values are: word, segment.",
            param="timestamp_granularities[]",
        )
    want_words = "word" in wanted and response_format == "verbose_json"

    audio_path = await _save_upload(file)
    started = time.monotonic()
    try:
        result = await asyncio.to_thread(
            api.transcribe, audio_path, resolved, want_words
        )
    except RuntimeError as exc:  # ffmpeg failure on a non-audio upload
        LOG.warning("decode failed for %s: %s", file.filename, exc)
        raise ApiError(
            400,
            "Audio file could not be decoded. Supported formats: "
            "flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm.",
            param="file",
        ) from exc
    finally:
        os.unlink(audio_path)

    LOG.info(
        "%s: %.1fs audio, %d segments in %.2fs (%s)",
        resolved,
        result.duration,
        len(result.segments),
        time.monotonic() - started,
        response_format,
    )

    if stream:
        return _stream_response(result)
    if response_format == "text":
        return PlainTextResponse(result.text + "\n")
    if response_format == "srt":
        return PlainTextResponse(_srt(result), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(_vtt(result), media_type="text/vtt")
    if response_format == "verbose_json":
        return JSONResponse(_verbose_json(result, wanted))
    return JSONResponse(
        {"text": result.text, "usage": _usage(result)}
    )


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """GigaAM transcribes, it does not translate into English - say so instead of
    silently returning a transcript the caller would read as an English translation."""
    _check_auth(authorization)
    raise ApiError(
        400,
        "Translation is not supported by this endpoint. "
        "Use POST /v1/audio/transcriptions for speech-to-text.",
        param="model",
    )
