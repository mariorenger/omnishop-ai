"""Error helpers and a correlation id for request tracing (observability §12)."""
from __future__ import annotations
import uuid
from fastapi import HTTPException


def bad_request(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)


def unauthorized(msg: str = "not authenticated") -> HTTPException:
    return HTTPException(status_code=401, detail=msg)


def forbidden(msg: str = "forbidden") -> HTTPException:
    return HTTPException(status_code=403, detail=msg)


def not_found(msg: str = "not found") -> HTTPException:
    return HTTPException(status_code=404, detail=msg)


def payment_required(msg: str) -> HTTPException:
    return HTTPException(status_code=402, detail=msg)


def new_correlation_id() -> str:
    return uuid.uuid4().hex
