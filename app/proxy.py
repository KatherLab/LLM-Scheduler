# app/proxy.py
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator
import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

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
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=80,
                keepalive_expiry=30,
            ),
            follow_redirects=False,
            http2=False,  # vLLM serves HTTP/1.1
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


async def proxy_json_or_stream(
    request: Request,
    upstream_url: str,
    *,
    body: bytes | None = None,
    is_stream: bool | None = None,
    timeout_s: float = 600.0,
) -> Response:
    """
    Proxy a request to upstream vLLM.

    If `body` and `is_stream` are provided, we skip re-reading/re-parsing
    the request (saves one full JSON parse + body read).
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

    client = await _get_client(timeout_s)

    if is_stream:
        async def gen() -> AsyncIterator[bytes]:
            try:
                req = client.build_request(
                    "POST", upstream_url, content=body, headers=headers,
                )
                resp = await client.send(req, stream=True)
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()
            except Exception as e:
                error_data = json.dumps(
                    {"error": {"message": str(e), "type": "proxy_error"}}
                )
                yield f"data: {error_data}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        r = await client.post(upstream_url, content=body, headers=headers)
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_filter_headers(r.headers),
        )
