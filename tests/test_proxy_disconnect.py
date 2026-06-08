"""Tests for proxy disconnect handling and upstream cancellation.

Scenarios covered (see individual docstrings):
  1. Disconnect before upstream response headers
  2. Disconnect after upstream response headers but before body completion (streaming)
  3. Buffered non-stream response disconnect during body read
  4. send_task never returns after disconnect (timeout)
  5. send_task raises httpx.RequestError after downstream disconnect
"""
from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest

from app.proxy import _send_upstream_or_abort, _ProxyResponse, DownstreamDisconnected


# ======================================================================
# ASGI helpers
# ======================================================================


class ControlledReceive:
    """ASGI ``receive`` callable that delivers messages on demand."""

    def __init__(self):
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    async def __call__(self):
        return await self._queue.get()

    def send_disconnect(self):
        self._queue.put_nowait({"type": "http.disconnect"})


class CaptureSend:
    """ASGI ``send`` callable that records every message sent."""

    def __init__(self):
        self.messages: list[dict] = []

    async def __call__(self, message: dict):
        self.messages.append(message)

    @property
    def started(self) -> bool:
        return any(m["type"] == "http.response.start" for m in self.messages)

    @property
    def response_status(self) -> int | None:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return m.get("status")
        return None


# ======================================================================
# Helper coroutines for controlled asyncio tasks
# ======================================================================


async def _raise_disconnect():
    """Coroutine that raises DownstreamDisconnected when awaited."""
    raise DownstreamDisconnected()


async def _never():
    """Coroutine that never completes (for tasks we want to remain pending)."""
    await asyncio.Event().wait()


# ======================================================================
# Shared mocks
# ======================================================================


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.send = AsyncMock()
    client.build_request = MagicMock(return_value=MagicMock(spec=httpx.Request))
    return client


# ======================================================================
# Test 1 — disconnect before upstream response headers
# ======================================================================


@pytest.mark.asyncio
async def test_disconnect_before_upstream_response_headers():
    """Test 1: disconnect fires before ``client.send()`` returns.

    ``_send_upstream_or_abort`` should wait for the send task (up to 3 s),
    call ``resp.aclose()``, and raise ``DownstreamDisconnected``.
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    req = MagicMock(spec=httpx.Request)
    mock_resp = AsyncMock(spec=httpx.Response)

    # send_task is delayed — controlled by an event
    send_ready = asyncio.Event()

    async def _delayed_send(*args, **kwargs):
        await send_ready.wait()
        return mock_resp

    client.send.side_effect = _delayed_send

    # disconnect_task fires immediately
    disconnect_task = asyncio.create_task(_raise_disconnect())
    await asyncio.sleep(0)  # let it finish

    deadline = asyncio.get_running_loop().time() + 10.0

    # Start _send_upstream_or_abort
    func_task = asyncio.create_task(
        _send_upstream_or_abort(client, req, disconnect_task, deadline),
    )
    await asyncio.sleep(0)

    # Now let send_task complete
    send_ready.set()

    with pytest.raises(DownstreamDisconnected):
        await func_task

    mock_resp.aclose.assert_called_once()


# ======================================================================
# Test 2 — streaming disconnect during body
# ======================================================================


@pytest.mark.asyncio
async def test_streaming_disconnect_during_body():
    """Test 2: downstream disconnects while upstream body is streaming.

    ``_send_upstream_or_abort`` returns normally, but
    ``_send_streaming_response`` is interrupted by the disconnect.
    The outer ``finally`` block in ``__call__`` must call
    ``resp.aclose()``.
    """
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/event-stream"}

    receive_ctrl = ControlledReceive()

    async def _chunks():
        yield b'data: {"content":"hello"}\n\n'
        # Trigger disconnect while the proxy waits for the next chunk
        receive_ctrl.send_disconnect()
        await asyncio.Event().wait()  # never yield again
        yield b"data: [DONE]\n\n"

    mock_resp.aiter_raw = _chunks

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.send.return_value = mock_resp

    send_capture = CaptureSend()

    with patch("app.proxy._get_client", return_value=mock_client):
        resp_obj = _ProxyResponse(
            request=MagicMock(),
            upstream_url="http://upstream:8000/v1/chat/completions",
            headers={"content-type": "application/json"},
            body=b'{"model":"t","stream":true}',
            is_stream=True,
            timeout_s=10.0,
            model="t",
        )
        await resp_obj(
            scope={"type": "http"},
            receive=receive_ctrl,
            send=send_capture,
        )

    # Outer finally must have closed the upstream response
    mock_resp.aclose.assert_called_once()

    # The response headers were sent (we got the first chunk)
    assert send_capture.started
    assert send_capture.response_status == 200


# ======================================================================
# Test 3 — buffered disconnect during body read
# ======================================================================


@pytest.mark.asyncio
async def test_buffered_disconnect_during_body():
    """Test 3: downstream disconnects while buffering the upstream body.

    ``_ProxyResponse.__call__`` is in the buffered (non-stream) path.
    ``_send_buffered_response`` → ``_read_all_or_disconnect`` gets
    interrupted.  The finally block must still call ``resp.aclose()``.
    """
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}

    receive_ctrl = ControlledReceive()

    async def _chunks():
        yield b'{"first": true}'
        receive_ctrl.send_disconnect()
        await asyncio.Event().wait()
        yield b'"second": true}'

    mock_resp.aiter_raw = _chunks

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.send.return_value = mock_resp

    send_capture = CaptureSend()

    with patch("app.proxy._get_client", return_value=mock_client):
        resp_obj = _ProxyResponse(
            request=MagicMock(),
            upstream_url="http://upstream:8000/v1/chat/completions",
            headers={"content-type": "application/json"},
            body=b'{"model":"t","stream":false}',
            is_stream=False,
            timeout_s=10.0,
            model="t",
        )
        await resp_obj(
            scope={"type": "http"},
            receive=receive_ctrl,
            send=send_capture,
        )

    # Finally block closed the upstream response
    mock_resp.aclose.assert_called_once()

    # Response was started by _safe_empty_response with 499
    assert send_capture.response_status == 499


# ======================================================================
# Test 4 — send_task never returns after disconnect
# ======================================================================


@pytest.mark.asyncio
async def test_send_task_timeout_after_disconnect():
    """Test 4: send_task never returns after downstream disconnect.

    ``_send_upstream_or_abort`` waits up to 3 s for the send to
    finish.  After the timeout it cancels the send task and raises
    ``DownstreamDisconnected``.  No exception should leak from the
    cancelled task.
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    req = MagicMock(spec=httpx.Request)

    # send_task hangs forever
    async def _hung_send(*args, **kwargs):
        await asyncio.Event().wait()
        raise RuntimeError("should never get here")

    client.send.side_effect = _hung_send

    disconnect_task = asyncio.create_task(_raise_disconnect())
    await asyncio.sleep(0)  # let it finish

    deadline = asyncio.get_running_loop().time() + 10.0

    func_task = asyncio.create_task(
        _send_upstream_or_abort(client, req, disconnect_task, deadline),
    )

    with pytest.raises(DownstreamDisconnected):
        await func_task

    # The cancelled send_task must not leave a dangling exception.
    # (we give the event loop a tick so the cancellation propagates)
    await asyncio.sleep(0)


# ======================================================================
# Test 5 — send_task raises httpx.RequestError after disconnect
# ======================================================================


@pytest.mark.asyncio
async def test_send_upstream_error_after_disconnect():
    """Test 5: send_task raises ``httpx.RequestError`` after disconnect.

    The upstream failed while the downstream was already gone.
    ``_send_upstream_or_abort`` should treat this as a downstream
    disconnect (499) rather than a proxy error (502).
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    req = MagicMock(spec=httpx.Request)

    async def _failing_send(*args, **kwargs):
        raise httpx.ConnectError("upstream refused connection")

    client.send.side_effect = _failing_send

    disconnect_task = asyncio.create_task(_raise_disconnect())
    await asyncio.sleep(0)  # let it finish

    deadline = asyncio.get_running_loop().time() + 10.0

    with pytest.raises(DownstreamDisconnected):
        await _send_upstream_or_abort(client, req, disconnect_task, deadline)
