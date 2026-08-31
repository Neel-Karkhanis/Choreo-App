"""Anonymous, device-scoped identity.

Replaces src/api/auth.py's account system: there are no accounts, no
passwords, no email. Every request is scoped to a device by a UUIDv4
"owner_id" — today that IS the whole identity, but the field stays named
generically (not "user_id") because a real account id may sit behind it
later, and every persisted row/record/sidecar keeps carrying owner_id as
an explicit field for exactly that reason (see server.py's DEVICES AND
DATA ISOLATION note).

The device id travels two ways:
  - a signed, HttpOnly cookie the server sets and reads (itsdangerous,
    the same SESSION_SECRET-backed signing auth.py used to sign session
    cookies). This is the durable, tamper-evident copy: a client cannot
    read it (HttpOnly) or forge one that verifies (signed), so the only
    way to present a valid cookie for device X is to already have been
    handed one for device X.
  - an X-Choreo-Device-Id header the client sends when it has no cookie
    (Safari's ITP having evicted it, or a non-browser client). This one
    is NOT signed — it can't be, the client itself originates it from its
    own IndexedDB mirror — so it carries no more trust than "whoever
    holds this 128-bit random UUID gets this device's data". That is the
    same security model a share link or capability URL already has, and
    it is an acceptable trade for an app with no accounts: the id is
    unguessable and never enumerable, so losing it only loses
    convenience, never someone else's data.

get_owner_id is the dependency every route in server.py takes, in the
exact spot auth.get_current_user used to sit. It only extracts and
validates; it never mints an id (that's issue_device, called only from
POST /api/device) and it never falls back to a shared/default owner — a
request with neither a valid cookie nor a valid header is a 400, full
stop.

BEARER CREDENTIAL, NOT JUST AN IDENTIFIER: POST /api/device's
requested_device_id (see issue_device) lets a caller hand in ANY
syntactically valid UUIDv4 and get it cookied — the server has no way to
verify the caller actually held that id before, only that it's
well-formed. That is deliberate (it is what lets a device recover its
cookie from its own IndexedDB mirror after Safari evicts the cookie —
see server.py's DeviceRequest), but the honest description of the
mechanism is that owner_id is a bearer credential, exactly like the
X-Choreo-Device-Id header above: whoever presents it gets cookied for
it, full stop, with no check of "is this really yours". Treat it
accordingly everywhere it flows: never log it (see owner_key below for
the redacted stand-in to use in logs/keys instead), never put it in a
URL or query string (POST body / cookie / header only), and never
include it in a Phase 4 export file.

CRITICAL: owner_id ends up as a raw filesystem path segment all over
server.py (tracks/<owner_id>/, cache/stems/<owner_id>/, data/grids/
<owner_id>/, ...). _is_canonical_uuid4 is the entire defense against
path traversal via that value, so it is deliberately not a regex or a
length check: uuid.UUID(value, version=4) parses, then str(parsed) is
compared back against the original string. Anything that isn't already
in that one canonical form — extra characters, wrong case, '../', an
absolute path, anything — fails either the parse or the round-trip and
is rejected before any route body ever runs.
"""

import hashlib
import os
import uuid

from fastapi import Cookie, Header, HTTPException, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

DEVICE_COOKIE_NAME = "choreo_device"
DEVICE_HEADER_NAME = "X-Choreo-Device-Id"
DEVICE_COOKIE_MAX_AGE_SECONDS = 400 * 24 * 60 * 60  # 400 days

_SECRET = os.environ.get("SESSION_SECRET")
if not _SECRET:
    raise RuntimeError(
        "SESSION_SECRET is not set. The device-id cookie is cryptographically "
        "signed with it, so a missing (or default, or shared-across-deploys) "
        "secret must never silently ship. Set it in your .env (see "
        ".env.example) or the process environment before starting the "
        "server."
    )

_serializer = URLSafeTimedSerializer(_SECRET, salt="choreo-device")


def _is_canonical_uuid4(value: str) -> bool:
    """True iff `value` is EXACTLY the canonical string form of a UUIDv4.

    See the module docstring's CRITICAL note — this is the whole path
    traversal defense for owner_id, so it stays a parse-and-round-trip
    check rather than a pattern match that something unexpected could
    slip past.
    """
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value, version=4)
    except (ValueError, AttributeError, TypeError):
        return False
    return str(parsed) == value


def make_device_cookie(device_id: str) -> str:
    return _serializer.dumps(device_id)


def _unsign_cookie(value: str) -> str:
    try:
        device_id = _serializer.loads(value, max_age=DEVICE_COOKIE_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired) as exc:
        raise HTTPException(
            status_code=400, detail="Device id cookie is invalid or expired"
        ) from exc
    if not _is_canonical_uuid4(device_id):
        # Can only happen if something we ourselves signed stops being a
        # canonical UUID (it never will), or the signing secret was reused
        # for other cookie data — fail loudly rather than trust it anyway.
        raise HTTPException(status_code=400, detail="Device id cookie is malformed")
    return device_id


def _try_unsign_cookie(value: str | None) -> str | None:
    """Like _unsign_cookie, but returns None on any failure instead of
    raising. For issue_device, which needs to check "is there already a
    valid cookie on this request" without turning an absent, expired, or
    stale-secret cookie into a hard failure of the bootstrap call itself —
    POST /api/device has to succeed and hand back SOME id even when the
    incoming cookie is garbage.
    """
    if value is None:
        return None
    try:
        return _unsign_cookie(value)
    except HTTPException:
        return None


def get_owner_id(
    device_cookie: str | None = Cookie(default=None, alias=DEVICE_COOKIE_NAME),
    device_header: str | None = Header(default=None, alias=DEVICE_HEADER_NAME),
) -> str:
    """The FastAPI dependency every route in server.py takes.

    Cookie wins when both are present (see the module docstring on why
    it's the more-trusted of the two). Missing or malformed in both
    places is a 400 — never a silent fallback to a shared/default owner.
    """
    if device_cookie is not None:
        return _unsign_cookie(device_cookie)
    if device_header is not None:
        if not _is_canonical_uuid4(device_header):
            raise HTTPException(status_code=400, detail="Device id header is malformed")
        return device_header
    raise HTTPException(status_code=400, detail="Missing device id")


def owner_key(owner_id: str) -> str:
    """A short, stable, one-way stand-in for owner_id, for anywhere an
    identifier derived from it would otherwise end up in a place raw
    owner_id must never appear — a Redis key, an RQ job id (both of which
    a worker's default logging prints on every job start/finish), a debug
    print. See jobs.py and analysis.py, which key job dedup, lock, and
    live-progress state off this instead of owner_id directly.

    One-way in practice, not just in name: recovering owner_id from this
    hash is as hard as a SHA-256 pre-image (infeasible), and even a full
    break of that would hand back nothing more useful than owner_id itself
    already is — this is not a substitute for keeping owner_id out of logs
    in the first place, just a way to keep OTHER identifiers correlatable
    across log lines without also being a bearer credential themselves:
    unlike owner_id, this value fails _is_canonical_uuid4 by construction,
    so leaking it cannot be used to claim a device's cookie or header.
    """
    return hashlib.sha256(owner_id.encode()).hexdigest()[:32]


def issue_device(
    response: Response,
    secure: bool,
    requested_device_id: str | None = None,
    existing_cookie: str | None = None,
) -> str:
    """Mint, adopt, or reaffirm a device id and set its signed cookie on
    `response`.

    Called only from POST /api/device — the frontend's one bootstrap call
    on startup, before anything else hits the API. Three cases, checked in
    this order:

      1. existing_cookie already unsigns to a valid device id: THAT id
         wins, full stop, regardless of what requested_device_id says.
         This is the reconciliation rule ("cookie wins if both exist and
         differ") — it has to live here, since the raw HttpOnly cookie
         value is never readable by the frontend JS that would otherwise
         have to do this comparison itself. It's also what makes calling
         this endpoint on every boot safe for a browser that already has
         a live cookie: a stale IndexedDB mirror (e.g. left over from
         before a manual cookie clear, or a device that was reassigned —
         see scripts/reassign_owner.py) can never silently swap out a
         session the cookie already authenticates.
      2. Otherwise, requested_device_id given: the frontend's IndexedDB
         mirror of an id whose cookie is gone (evicted, e.g. by Safari's
         7-day ITP sweep). Re-cookie that SAME id rather than minting a
         fresh one, so the device doesn't silently fork into two devices'
         worth of data.
      3. Otherwise: a real first visit. Mint a fresh UUIDv4 and cookie it.

    "Adopt" (case 2) means exactly what it sounds like and no more: this
    function does not and cannot verify the caller actually held
    requested_device_id before — it only checks the id is well-formed
    (a 400 on anything else, never a silent mint-fresh, so a corrupted
    local cache fails loudly instead of quietly orphaning whatever it
    used to point at) and cookies it. Presenting a valid-looking UUIDv4
    here is sufficient to take ownership of it — see the module
    docstring's BEARER CREDENTIAL note. The IndexedDB-recovery case is
    the only intended caller; nothing stops a request that skips the
    frontend entirely and just POSTs a UUID it invented or copied from
    somewhere, and that is accepted as the cost of having no accounts
    to check the id against. Case 1 (cookie wins) is not subject to this
    at all — a valid signed cookie IS the proof case 2 can't have.
    """
    reaffirmed = _try_unsign_cookie(existing_cookie)
    if reaffirmed is not None:
        device_id = reaffirmed
    elif requested_device_id is not None:
        if not _is_canonical_uuid4(requested_device_id):
            raise HTTPException(status_code=400, detail="Device id is malformed")
        device_id = requested_device_id
    else:
        device_id = str(uuid.uuid4())

    response.set_cookie(
        key=DEVICE_COOKIE_NAME,
        value=make_device_cookie(device_id),
        max_age=DEVICE_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure,
    )
    return device_id
