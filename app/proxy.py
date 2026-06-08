# app/proxy.py
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from .metrics import endpoint_label, track_proxy

logger = logging.getLogger(__name__)

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


async def _get_client() -> httpx.AsyncClient:
    """Return a long-lived shared AsyncClient with connection pooling."""
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
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
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in HOP_BY_HOP_HEADERS:
            continue
        out[k] = v
    return out


def _asgi_headers(headers: dict[str, str]) -> list[tuple[bytes, bytes]]:
    return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]


def _openai_error(
    message: str,
    *,
    type_: str = "api_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
):
    err = {"message": message, "type": type_, "param": param, "code": code}
    return {"error": err}


def _status_for_httpx_exc(exc: Exception) -> int:
    if isinstance(exc, asyncio.TimeoutError):
        return 504
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return 504
    if isinstance(exc, httpx.TimeoutException):
        return 504
    if isinstance(exc, httpx.RequestError):
        return 502
    return 500


def _payload_for_exc(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return _openai_error(
            "Upstream request timed out",
            type_="timeout_error",
            code="upstream_timeout",
        )
    if isinstance(exc, httpx.RequestError):
        return _openai_error(
            "Upstream connection error",
            type_="api_error",
            code="upstream_connection_error",
        )
    return _openai_error(
        "Internal proxy error",
        type_="api_error",
        code="proxy_internal_error",
    )


def _deadline_after(timeout_s: float) -> float:
    return asyncio.get_running_loop().time() + timeout_s


def _time_left(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError()
    return remaining


class DownstreamDisconnected(Exception):
    """Raised when the downstream HTTP client disconnects."""


async def _listen_for_disconnect(receive) -> None:
    """
    Wait until ASGI tells us the client disconnected.

    Important: this should only be used after the request body has already been
    consumed by the route/request parser. In your routes that is already true:
    - JSON endpoints call request.body() before proxying
    - multipart endpoints call request.form() before proxying
    """
    while True:
        message = await receive()
        mtype = message["type"]

        if mtype == "http.disconnect":
            raise DownstreamDisconnected()

        # Ignore any remaining request-body events defensively.
        if mtype == "http.request":
            continue


async def _await_or_disconnect(awaitable, disconnect_task: asyncio.Task, deadline: float):
    """
    Await one upstream operation, but abort immediately if downstream disconnects.
    """
    work_task = asyncio.create_task(awaitable)
    try:
        done, _ = await asyncio.wait(
            {work_task, disconnect_task},
            timeout=_time_left(deadline),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            work_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await work_task
            raise asyncio.TimeoutError()

        if disconnect_task in done:
            work_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await work_task

            exc = disconnect_task.exception()
            if exc is None:
                raise DownstreamDisconnected()
            raise exc

        return await work_task

    finally:
        if not work_task.done():
            work_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work_task


async def _read_all_or_disconnect(
    resp: httpx.Response,
    disconnect_task: asyncio.Task,
    deadline: float,
) -> bytes:
    chunks = bytearray()
    ait = resp.aiter_raw()

    while True:
        try:
            chunk = await _await_or_disconnect(ait.__anext__(), disconnect_task, deadline)
        except StopAsyncIteration:
            break
        chunks.extend(chunk)

    return bytes(chunks)


async def _send_upstream_or_abort(
    client: httpx.AsyncClient,
    req: httpx.Request,
    disconnect_task: asyncio.Task,
    deadline: float,
) -> httpx.Response:
    """
    Send the upstream request, aborting if downstream disconnects.

    This handles the race where the downstream disconnects before
    ``client.send()`` has returned a response object.  After
    ``client.send()`` returns, the normal ``finally`` block in
    ``_ProxyResponse.__call__`` already closes ``resp``.  Before that,
    we briefly wait for ``resp`` and close it if it appears, so the
    upstream vLLM connection is explicitly closed.
    """
    send_task = asyncio.create_task(client.send(req, stream=True))

    done, _ = await asyncio.wait(
        {send_task, disconnect_task},
        timeout=_time_left(deadline),
        return_when=asyncio.FIRST_COMPLETED,
    )

    if not done:
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send_task
        raise asyncio.TimeoutError()

    if disconnect_task in done:
        # Downstream disconnected while send was still in-flight.
        # The request may already have been sent to vLLM, which is now
        # processing it.  Grab the response (up to 3 s) and close it so
        # the underlying TCP connection is torn down and vLLM detects it.
        # If send_task later raises an httpx error, treat it as a
        # downstream disconnect (the client is already gone).
        try:
            resp = await asyncio.wait_for(send_task, timeout=3.0)
        except asyncio.TimeoutError:
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task
        except Exception:
            # Upstream failed while downstream was already gone.
            # Treat this as downstream disconnect, not proxy failure.
            pass
        else:
            with contextlib.suppress(Exception):
                await resp.aclose()

        exc = disconnect_task.exception()
        if exc is None:
            raise DownstreamDisconnected()
        raise exc

    return send_task.result()


class _ProxyResponse(Response):
    media_type = None

    def __init__(
        self,
        *,
        request: Request,
        upstream_url: str,
        headers: dict[str, str],
        body: bytes | None = None,
        data: dict[str, str] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        is_stream: bool = False,
        timeout_s: float = 600.0,
        model: str = "",
    ) -> None:
        super().__init__(content=b"", status_code=200)
        self._request = request
        self._upstream_url = upstream_url
        self._headers = headers
        self._body = body
        self._data = data
        self._files = files
        self._is_stream = is_stream
        self._timeout_s = timeout_s
        self._model = model

    async def _send_error_response(self, scope, receive, send, exc: Exception) -> None:
        status = _status_for_httpx_exc(exc)
        payload = _payload_for_exc(exc)
        body = json.dumps(payload).encode("utf-8")

        headers = _asgi_headers({
            "content-type": "application/json",
            "content-length": str(len(body)),
        })

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        })
        self._response_started = True

        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })

    async def _safe_empty_response(self, send, status: int) -> None:
        """
        Send a minimal response to satisfy the ASGI contract when we have not
        yet started a response. Used when the client already disconnected —
        uvicorn will discard the bytes but we avoid the
        'ASGI callable returned without starting response' error.
        """
        if self._response_started:
            return
        try:
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-length", b"0")],
            })
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
        except Exception:
            # Client truly gone; uvicorn will log the closed socket, but we
            # have satisfied the contract from our side.
            pass
        finally:
            self._response_started = True

    async def _send_buffered_response(
        self,
        scope,
        receive,
        send,
        resp: httpx.Response,
        disconnect_task: asyncio.Task,
        deadline: float,
    ) -> None:
        content = await _read_all_or_disconnect(resp, disconnect_task, deadline)
        headers = _asgi_headers(_filter_headers(resp.headers))

        await send({
            "type": "http.response.start",
            "status": resp.status_code,
            "headers": headers,
        })
        self._response_started = True  # ← mark as started right after http.response.start

        await send({
            "type": "http.response.body",
            "body": content,
            "more_body": False,
        })


    async def _send_streaming_response(
        self,
        send,
        resp: httpx.Response,
        disconnect_task: asyncio.Task,
        deadline: float,
    ) -> None:
        headers = _asgi_headers(_filter_headers(resp.headers))

        await send(
            {
                "type": "http.response.start",
                "status": resp.status_code,
                "headers": headers,
            }
        )
        self._response_started = True

        ait = resp.aiter_raw()

        while True:
            try:
                chunk = await _await_or_disconnect(ait.__anext__(), disconnect_task, deadline)
            except StopAsyncIteration:
                break

            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                }
            )

        await send(
            {
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            }
        )
        
    async def __call__(self, scope, receive, send) -> None:
        client = await _get_client()
        deadline = _deadline_after(self._timeout_s)
        disconnect_task = asyncio.create_task(_listen_for_disconnect(receive))
        resp: httpx.Response | None = None
        self._response_started = False  # ← NEW: track ASGI response lifecycle

        async with track_proxy(self._upstream_url, model=self._model) as _ctx:
            try:
                req = client.build_request(
                    "POST",
                    self._upstream_url,
                    content=self._body,
                    data=self._data,
                    files=self._files,
                    headers=self._headers,
                )

                resp = await _send_upstream_or_abort(
                    client,
                    req,
                    disconnect_task,
                    deadline,
                )

                # Non-stream: buffer whole upstream body before sending anything.
                if not self._is_stream:
                    await self._send_buffered_response(scope, receive, send, resp, disconnect_task, deadline)
                    _ctx["status"] = resp.status_code
                    return

                # Stream: if upstream already errored, buffer and return that error.
                if resp.status_code >= 400:
                    await self._send_buffered_response(scope, receive, send, resp, disconnect_task, deadline)
                    _ctx["status"] = resp.status_code
                    return

                await self._send_streaming_response(send, resp, disconnect_task, deadline)
                _ctx["status"] = resp.status_code

            except DownstreamDisconnected:
                # Client disconnected before we finished. If we hadn't started a
                # response yet, satisfy the ASGI contract with a 499 so uvicorn
                # does NOT synthesize a 500 "ASGI callable returned without
                # starting response".
                logger.info(
                    "proxy: downstream disconnected (url=%s, response_started=%s)",
                    self._upstream_url, self._response_started,
                )
                await self._safe_empty_response(send, status=499)
                _ctx["status"] = 499
                return

            except OSError as exc:
                # Socket-level error talking to the downstream client. If we have
                # not started a response, still try to satisfy the ASGI contract.
                logger.warning(
                    "proxy: OSError during proxy (url=%s, response_started=%s): %s",
                    self._upstream_url, self._response_started, exc,
                )
                await self._safe_empty_response(send, status=499)
                _ctx["status"] = 499
                return

            except Exception as exc:
                # Real error (upstream timeout, connection error, proxy bug, ...).
                # These must be propagated to the client whenever possible.
                logger.exception("proxy failed: %s", self._upstream_url)
                _ctx["error"] = type(exc).__name__
                _ctx["status"] = _status_for_httpx_exc(exc)

                if not self._response_started:
                    try:
                        await self._send_error_response(scope, receive, send, exc)
                    except OSError:
                        # Client is already gone — still satisfy ASGI contract.
                        await self._safe_empty_response(send, status=500)
                    except Exception:
                        logger.exception("proxy: failed to send error response")
                        await self._safe_empty_response(send, status=500)
                else:
                    # Mid-stream failure: we already sent http.response.start with
                    # a 2xx status, so we cannot change the status code now. Best
                    # we can do is cleanly close the body. Log so ops can see it.
                    logger.error(
                        "proxy: error after response started (url=%s): %s",
                        self._upstream_url, exc,
                    )
                    with contextlib.suppress(Exception):
                        await send({
                            "type": "http.response.body",
                            "body": b"",
                            "more_body": False,
                        })
                return

            finally:
                disconnect_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await disconnect_task

                if resp is not None:
                    with contextlib.suppress(Exception):
                        await resp.aclose()


async def proxy_json_or_stream(
    request: Request,
    upstream_url: str,
    *,
    body: bytes | None = None,
    is_stream: bool | None = None,
    timeout_s: float = 600.0,
    model: str = "",
):
    """
    Return an ASGI response object that proxies to an upstream OpenAI-compatible server.

    Key behavior:
    - downstream disconnect aborts upstream request for BOTH stream and non-stream
    - non-stream buffers upstream fully before replying
    - stream passes bytes through directly after upstream headers
    """
    if body is None:
        body = await request.body()

    headers = dict(request.headers)
    headers.pop("host", None)

    if is_stream is None:
        try:
            j = json.loads(body.decode("utf-8") or "{}")
            is_stream = bool(j.get("stream", False))
        except Exception:
            is_stream = False

    return _ProxyResponse(
        request=request,
        upstream_url=upstream_url,
        headers=headers,
        body=body,
        is_stream=is_stream,
        timeout_s=timeout_s,
        model=model,
    )


async def proxy_get(
    request: Request,
    upstream_url: str,
    *,
    timeout_s: float = 30.0,
    model: str = "",
):
    """
    Proxy a GET request to upstream and return the buffered response.

    Used for endpoints like /metrics that don't need streaming or JSON
    request bodies.
    """
    client = await _get_client()
    headers = dict(request.headers)
    headers.pop("host", None)

    async with track_proxy(upstream_url, model=model) as _ctx:
        try:
            resp = await client.get(upstream_url, headers=headers, timeout=timeout_s)
            _ctx["status"] = resp.status_code
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=_filter_headers(resp.headers),
            )
        except Exception as exc:
            _ctx["error"] = type(exc).__name__
            _ctx["status"] = _status_for_httpx_exc(exc)
            raise


async def proxy_multipart(
    request: Request,
    upstream_url: str,
    *,
    timeout_s: float = 600.0,
    model: str = "",
):
    """
    Proxy multipart/form-data requests (e.g. OpenAI audio transcription).

    We parse the form before returning the proxy response so that later disconnect
    listening can safely use the ASGI receive channel without interfering with form parsing.
    """
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    form = await request.form()

    data: dict[str, str] = {}
    files: list[tuple[str, tuple[str, bytes, str]]] = []

    for key, value in form.multi_items():
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

    return _ProxyResponse(
        request=request,
        upstream_url=upstream_url,
        headers=headers,
        data=data,
        files=files,
        is_stream=False,
        timeout_s=timeout_s,
        model=model,
    )