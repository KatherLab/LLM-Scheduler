# app/utils.py
from datetime import datetime, timezone


def ensure_utc(dt: datetime | None) -> datetime:
    """
    Ensure a datetime is timezone-aware (UTC).
    If None, returns current UTC time.
    If naive, assumes UTC and attaches tzinfo.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
