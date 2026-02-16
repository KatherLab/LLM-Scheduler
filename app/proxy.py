from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator, Optional, Tuple
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
    timeout_s: float = 600.0,
) -> Response:
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        # Try to detect stream=true (OpenAI style)
        is_stream = False
        try:
            j = json.loads(body.decode("utf-8") or "{}")
            is_stream = bool(j.get("stream", False))
        except Exception:
            is_stream = False

        if is_stream:
            async def gen() -> AsyncIterator[bytes]:
                async with client.stream("POST", upstream_url, content=body, headers=headers) as r:
                    # forward status? For SSE, we stream regardless; errors still stream body.
                    async for chunk in r.aiter_bytes():
                        yield chunk
            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            r = await client.post(upstream_url, content=body, headers=headers)
            return Response(content=r.content, status_code=r.status_code, headers=_filter_headers(r.headers))
