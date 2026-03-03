# app/proxy.py
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator
import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# ── Shared client (module-level singleton) ──────────────────────────────────
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client(timeout_s: float = 600.0) -> httpx.AsyncClient:
    """Return a long-lived shared AsyncClient with connection pooling."""
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
        # Double-check after acquiring lock
        if _client is not None and not _client.is_closed:
            return _client
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=10.0),
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=80,
                keepalive_expiry=30,
            ),
            follow_redirects=False,
            http2=False,
        )
        return _client


async def close_client() -> None:
    """Call on shutdown to cleanly close the shared client."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


def _filter_headers(headers: httpx.Headers) -> dict[str, str]:
    out = {}
    for k, v in headers.items():
        if k.lower() in HOP_BY_HOP_HEADERS:
            continue
        out[k] = v
    return out


import json
from typing import AsyncIterator, Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


def _openai_error(
    message: str,
    *,
    type_: str = "api_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
):
    # Matches the general OpenAI error envelope shape
    err = {"message": message, "type": type_, "param": param, "code": code}
    return {"error": err}


def _status_for_httpx_exc(exc: Exception) -> int:
    # Choose status codes that are common/expected for proxies
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return 504
    if isinstance(exc, httpx.TimeoutException):
        return 504
    if isinstance(exc, httpx.RequestError):
        # DNS failure, connection refused, network unreachable, etc.
        return 502
    return 500


async def proxy_json_or_stream(
    request: Request,
    upstream_url: str,
    *,
    body: bytes | None = None,
    is_stream: bool | None = None,
    timeout_s: float = 600.0,
) -> Response:
    """
    Proxy a request to an upstream OpenAI-compatible server (e.g., vLLM).

    Goals:
    - Pass through upstream responses verbatim when possible.
    - Never leak internal stacktraces to logs via unhandled exceptions.
    - Return OpenAI-style JSON errors for non-stream.
    - For stream, emit SSE error frames + [DONE].
    """
    if body is None:
        body = await request.body()

    headers = dict(request.headers)
    headers.pop("host", None)

    # Determine streaming mode (if not provided by caller)
    if is_stream is None:
        try:
            j = json.loads(body.decode("utf-8") or "{}")
            is_stream = bool(j.get("stream", False))
        except Exception:
            is_stream = False

    client = await _get_client(timeout_s)

    if is_stream:
        async def gen() -> AsyncIterator[bytes]:
            resp: httpx.Response | None = None
            try:
                req = client.build_request(
                    "POST",
                    upstream_url,
                    content=body,
                    headers=headers,
                )
                resp = await client.send(req, stream=True)

                # If upstream immediately returns an error status, you can either
                # pass through bytes (may not be SSE) or convert to SSE error.
                # Converting is usually friendlier for OpenAI-style streaming clients.
                if resp.status_code >= 400:
                    try:
                        raw = await resp.aread()
                        # Try to keep upstream error message if it is JSON
                        msg = None
                        try:
                            parsed = json.loads(raw.decode("utf-8"))
                            if isinstance(parsed, dict) and "error" in parsed:
                                # upstream already OpenAI-like
                                data = parsed
                            else:
                                data = _openai_error("Upstream error", type_="upstream_error")
                        except Exception:
                            data = _openai_error("Upstream error", type_="upstream_error")

                        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")
                    finally:
                        yield b"data: [DONE]\n\n"
                    return

                async for chunk in resp.aiter_bytes():
                    yield chunk

            except Exception as e:
                status = _status_for_httpx_exc(e)
                # Keep message short; don’t dump internal reprs
                if isinstance(e, httpx.TimeoutException):
                    msg = "Upstream request timed out"
                    code = "upstream_timeout"
                    type_ = "timeout_error"
                elif isinstance(e, httpx.RequestError):
                    msg = "Upstream connection error"
                    code = "upstream_connection_error"
                    type_ = "api_error"
                else:
                    msg = "Internal proxy error"
                    code = "proxy_internal_error"
                    type_ = "api_error"

                data = _openai_error(msg, type_=type_, code=code)
                # Streaming clients expect SSE frames, not plain JSON
                yield f"data: {json.dumps(data)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            finally:
                if resp is not None:
                    await resp.aclose()

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- non-stream path ----
    try:
        r = await client.post(upstream_url, content=body, headers=headers)
        # Pass through upstream content and status
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_filter_headers(r.headers),
        )

    except Exception as e:
        status = _status_for_httpx_exc(e)

        if isinstance(e, httpx.TimeoutException):
            payload = _openai_error(
                "Upstream request timed out",
                type_="timeout_error",
                code="upstream_timeout",
            )
        elif isinstance(e, httpx.RequestError):
            payload = _openai_error(
                "Upstream connection error",
                type_="api_error",
                code="upstream_connection_error",
            )
        else:
            payload = _openai_error(
                "Internal proxy error",
                type_="api_error",
                code="proxy_internal_error",
            )

        return JSONResponse(status_code=status, content=payload)


async def proxy_multipart(
    request: Request,
    upstream_url: str,
    *,
    timeout_s: float = 600.0,
) -> Response:
    """
    Proxy multipart/form-data requests (e.g. OpenAI audio transcription).
    Reads the form, forwards fields + files to upstream.
    """
    client = await _get_client(timeout_s)

    # Copy headers, but let httpx rebuild Content-Type for multipart boundary
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    try:
        form = await request.form()

        data: dict[str, str] = {}
        files: list[tuple[str, tuple[str, bytes, str]]] = []

        # form.multi_items() preserves repeated keys (e.g. timestamp_granularities[])
        for key, value in form.multi_items():
            # Starlette UploadFile has .filename and .content_type
            if hasattr(value, "filename"):
                upload = value
                content = await upload.read()
                files.append(
                    (
                        key,
                        (
                            upload.filename or "upload",
                            content,
                            upload.content_type or "application/octet-stream",
                        ),
                    )
                )
            else:
                data[key] = str(value)

        r = await client.post(upstream_url, data=data, files=files, headers=headers)

        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_filter_headers(r.headers),
        )

    except Exception as e:
        status = _status_for_httpx_exc(e)
        payload = _openai_error(
            "Upstream request timed out" if status == 504 else
            ("Upstream connection error" if status == 502 else "Internal proxy error"),
            type_="api_error",
            code="upstream_timeout" if status == 504 else
                 ("upstream_connection_error" if status == 502 else "proxy_internal_error"),
        )
        return JSONResponse(status_code=status, content=payload)