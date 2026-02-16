from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
import httpx

from .models import Endpoint

def choose_ready_endpoint(db: Session, model: str) -> Optional[Endpoint]:
    # Simple round-robin-ish: pick newest READY endpoint
    eps = db.execute(
        select(Endpoint).where(Endpoint.model == model, Endpoint.state == "READY").order_by(Endpoint.id.desc())
    ).scalars().all()
    return eps[0] if eps else None

async def health_check_endpoint(host: str, port: int, timeout_s: float = 2.0) -> tuple[bool, str | None]:
    url = f"http://{host}:{port}/health"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as c:
            r = await c.get(url)
            if r.status_code == 200:
                return True, None
            return False, f"health status {r.status_code}"
    except Exception as e:
        return False, str(e)
