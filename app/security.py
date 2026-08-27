"""Auth primitives with zero native deps: PBKDF2 password hashing + HS256 JWT.

We deliberately avoid bcrypt/PyJWT to keep the image slim and build-reliable.
The `AuthProvider` boundary (ADR-003) means this can be swapped for Keycloak/OIDC
later without touching callers.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from .config import config

# --- password hashing (PBKDF2-HMAC-SHA256) ---------------------------------
_PBKDF2_ROUNDS = 200_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:  # noqa: BLE001
        return False


# --- JWT (HS256) ------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(user_id: str, ttl_s: int = 7 * 24 * 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": str(user_id), "iat": round(time.time(), 3), "exp": int(time.time()) + ttl_s}
    seg = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    sig = hmac.new(config.APP_SECRET.encode(), seg.encode(), hashlib.sha256).digest()
    return f"{seg}.{_b64(sig)}"


def decode_token(token: str) -> Optional[dict]:
    """Return the verified payload ({sub, iat, exp}) if valid, else None."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        seg = f"{header_b64}.{payload_b64}"
        expected = hmac.new(config.APP_SECRET.encode(), seg.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64d(sig_b64)):
            return None
        payload = json.loads(_b64d(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:  # noqa: BLE001
        return None


def verify_token(token: str) -> Optional[str]:
    """Return the user id (sub) if valid, else None."""
    p = decode_token(token)
    return str(p["sub"]) if p else None


# --- lightweight symmetric encryption for stored credentials (R-17) ---------
# OAuth tokens etc. must never be stored plaintext. This is a simple app-secret
# derived Fernet-like scheme using HMAC + XOR keystream (AES would need a dep).
# For production, swap for KMS/OpenBao (documented in ADR/risk register).
def encrypt_secret(plaintext: str) -> bytes:
    key = hashlib.sha256(config.APP_SECRET.encode()).digest()
    nonce = os.urandom(16)
    keystream = b""
    counter = 0
    while len(keystream) < len(plaintext.encode()):
        keystream += hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
        counter += 1
    ct = bytes(a ^ b for a, b in zip(plaintext.encode(), keystream))
    mac = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    return nonce + mac + ct


def decrypt_secret(blob: bytes) -> str:
    key = hashlib.sha256(config.APP_SECRET.encode()).digest()
    nonce, mac, ct = blob[:16], blob[16:48], blob[48:]
    expected = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, mac):
        raise ValueError("credential MAC mismatch")
    keystream = b""
    counter = 0
    while len(keystream) < len(ct):
        keystream += hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(a ^ b for a, b in zip(ct, keystream)).decode()
