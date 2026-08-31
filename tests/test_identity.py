"""identity.get_owner_id is the entire trust boundary for owner_id, which
ends up as a raw filesystem path segment all over server.py
(tracks/<owner_id>/, cache/stems/<owner_id>/, data/grids/<owner_id>/...).
These tests exist because a regex or length check here would be a path
traversal vulnerability waiting to happen — see identity.py's module
docstring for why it's a uuid.UUID(..., version=4) parse-and-round-trip
check instead.

Every "malformed" case here is checked to raise BEFORE any filesystem
call could happen: get_owner_id and _is_canonical_uuid4 never touch disk,
so a passing test here is a guarantee the rejection happens at the
dependency, not somewhere deeper in a route body.
"""

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "api"))

import identity  # noqa: E402
from fastapi import HTTPException  # noqa: E402

# Deliberately contains hex letters (a-f), not just digits — a UUID made of
# digits alone would make the "wrong case" malformed-value check below a
# no-op (str.upper() on an all-digit string returns the same string).
VALID_UUID = "abcdefab-cdef-4bcd-9bcd-efabcdefabcd"

# Every one of these must be rejected by _is_canonical_uuid4 — not just the
# obviously-malicious ones. A round-trip check catches "parses as a UUID but
# isn't already canonical" as a side effect of catching "doesn't parse at
# all", and both matter equally here.
PATH_TRAVERSAL_VALUES = [
    "../../../etc/passwd",
    "..",
    "../",
    "..\\..\\windows\\system32",
]
ABSOLUTE_PATH_VALUES = [
    "/etc/passwd",
    "/",
    "C:\\Windows\\System32",
]
OTHER_MALFORMED_VALUES = [
    "",
    "not-a-uuid-at-all",
    "1",
    VALID_UUID.upper(),  # parses, but not the canonical lowercase form
    "{" + VALID_UUID + "}",  # braced form: uuid.UUID accepts it, str() won't
    "urn:uuid:" + VALID_UUID,  # URN form: uuid.UUID strips the prefix and accepts it too
    VALID_UUID + "x",  # trailing garbage
    " " + VALID_UUID,  # leading whitespace
    VALID_UUID + " ",  # trailing whitespace
    " " + VALID_UUID + " ",  # both
    "\t" + VALID_UUID,  # non-space whitespace
    VALID_UUID.replace("4", "1", 1),  # not a v4 UUID (wrong version nibble)
]


class TestIsCanonicalUuid4(unittest.TestCase):
    def test_accepts_a_canonical_uuid4(self):
        self.assertTrue(identity._is_canonical_uuid4(VALID_UUID))

    def test_rejects_path_traversal_values(self):
        for value in PATH_TRAVERSAL_VALUES:
            with self.subTest(value=value):
                self.assertFalse(identity._is_canonical_uuid4(value))

    def test_rejects_absolute_path_values(self):
        for value in ABSOLUTE_PATH_VALUES:
            with self.subTest(value=value):
                self.assertFalse(identity._is_canonical_uuid4(value))

    def test_rejects_other_malformed_values(self):
        for value in OTHER_MALFORMED_VALUES:
            with self.subTest(value=value):
                self.assertFalse(identity._is_canonical_uuid4(value))

    def test_rejects_non_string_input(self):
        self.assertFalse(identity._is_canonical_uuid4(None))
        self.assertFalse(identity._is_canonical_uuid4(12345))

    def test_rejects_forms_uuid_UUID_parses_but_that_are_not_canonical(self):
        """uuid.UUID() itself is lenient — it happily parses uppercase, a
        braced value, and a urn:uuid: value, normalizing all three to the
        same canonical object. The round-trip (str(parsed) == value) is
        what catches every one of these, since none of the four inputs
        below is already spelled the way str() would spell it back.

        Named explicitly (not just folded into OTHER_MALFORMED_VALUES
        above) because these are exactly the "parses fine, just isn't
        canonical" cases a naive `try: uuid.UUID(value); return True`
        implementation would wrongly accept.
        """
        self.assertFalse(identity._is_canonical_uuid4(VALID_UUID.upper()))
        self.assertFalse(identity._is_canonical_uuid4("{" + VALID_UUID + "}"))
        self.assertFalse(identity._is_canonical_uuid4("urn:uuid:" + VALID_UUID))
        self.assertFalse(identity._is_canonical_uuid4(" " + VALID_UUID))
        self.assertFalse(identity._is_canonical_uuid4(VALID_UUID + " "))
        self.assertFalse(identity._is_canonical_uuid4(" " + VALID_UUID + " "))


class TestGetOwnerIdHeaderPath(unittest.TestCase):
    """The X-Choreo-Device-Id header is unsigned, so it is validated purely
    by shape — this is the path a forged or malicious value would actually
    reach the server through.
    """

    def test_accepts_a_valid_header(self):
        self.assertEqual(
            identity.get_owner_id(device_cookie=None, device_header=VALID_UUID),
            VALID_UUID,
        )

    def test_rejects_missing_cookie_and_header(self):
        with self.assertRaises(HTTPException) as ctx:
            identity.get_owner_id(device_cookie=None, device_header=None)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_path_traversal_header_before_any_route_runs(self):
        for value in PATH_TRAVERSAL_VALUES:
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as ctx:
                    identity.get_owner_id(device_cookie=None, device_header=value)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_absolute_path_header(self):
        for value in ABSOLUTE_PATH_VALUES:
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as ctx:
                    identity.get_owner_id(device_cookie=None, device_header=value)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_other_malformed_headers(self):
        for value in OTHER_MALFORMED_VALUES:
            if value == "":
                # An empty header arrives as None (FastAPI/Starlette never
                # hands a dependency an empty-string header value distinct
                # from "absent") — covered by test_rejects_missing_* above.
                continue
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as ctx:
                    identity.get_owner_id(device_cookie=None, device_header=value)
                self.assertEqual(ctx.exception.status_code, 400)


class TestGetOwnerIdCookiePath(unittest.TestCase):
    """The cookie is signed — get_owner_id must unsign it and validate the
    result, and must never trust an unsigned or forged value even if it
    happens to look like a canonical UUID.
    """

    def test_accepts_a_cookie_this_server_signed(self):
        cookie = identity.make_device_cookie(VALID_UUID)
        self.assertEqual(
            identity.get_owner_id(device_cookie=cookie, device_header=None), VALID_UUID
        )

    def test_rejects_a_forged_unsigned_cookie_value(self):
        """Setting the cookie to the raw UUID (no signature) must not work —
        otherwise the signature buys nothing and anyone could set
        choreo_device directly to claim another device's id.
        """
        with self.assertRaises(HTTPException) as ctx:
            identity.get_owner_id(device_cookie=VALID_UUID, device_header=None)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_a_cookie_signed_with_a_different_secret(self):
        from itsdangerous import URLSafeTimedSerializer

        other_serializer = URLSafeTimedSerializer("a-different-secret", salt="choreo-device")
        forged = other_serializer.dumps(VALID_UUID)
        with self.assertRaises(HTTPException) as ctx:
            identity.get_owner_id(device_cookie=forged, device_header=None)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_garbage_cookie_value(self):
        with self.assertRaises(HTTPException) as ctx:
            identity.get_owner_id(device_cookie="not-even-a-signed-token", device_header=None)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_cookie_takes_precedence_over_header_when_both_present(self):
        cookie_owner = VALID_UUID
        header_owner = "22222222-2222-4222-8222-222222222222"
        cookie = identity.make_device_cookie(cookie_owner)
        self.assertEqual(
            identity.get_owner_id(device_cookie=cookie, device_header=header_owner),
            cookie_owner,
        )

    def test_an_invalid_cookie_is_not_silently_replaced_by_a_valid_header(self):
        """Malformed-in-one-place is a 400, never a quiet fallback to the
        other slot — see identity.get_owner_id's own docstring.
        """
        with self.assertRaises(HTTPException) as ctx:
            identity.get_owner_id(device_cookie="garbage", device_header=VALID_UUID)
        self.assertEqual(ctx.exception.status_code, 400)


class TestOwnerKey(unittest.TestCase):
    """owner_key is the redacted stand-in job/lock/progress keys (and any
    log line) use instead of the raw device id — see its own docstring for
    why. The property that actually matters here is the last one: leaking
    this value must not hand anyone a working credential.
    """

    def test_is_deterministic(self):
        self.assertEqual(identity.owner_key(VALID_UUID), identity.owner_key(VALID_UUID))

    def test_differs_between_owners(self):
        other = "22222222-2222-4222-8222-222222222222"
        self.assertNotEqual(identity.owner_key(VALID_UUID), identity.owner_key(other))

    def test_output_is_not_itself_a_usable_device_id(self):
        """Leaking owner_key(x) into a log must not hand out a value that
        would pass get_owner_id and grant access to x's data.
        """
        self.assertFalse(identity._is_canonical_uuid4(identity.owner_key(VALID_UUID)))


class _FakeResponse:
    """The one method identity.issue_device calls on a Response."""

    def __init__(self):
        self.cookies = []

    def set_cookie(self, **kwargs):
        self.cookies.append(kwargs)


class TestIssueDevice(unittest.TestCase):
    def test_mints_a_fresh_canonical_uuid_when_none_requested(self):
        response = _FakeResponse()
        device_id = identity.issue_device(response, secure=False)
        self.assertTrue(identity._is_canonical_uuid4(device_id))
        self.assertEqual(len(response.cookies), 1)
        self.assertEqual(response.cookies[0]["key"], identity.DEVICE_COOKIE_NAME)

    def test_adopts_a_valid_requested_id_instead_of_minting(self):
        response = _FakeResponse()
        device_id = identity.issue_device(response, secure=False, requested_device_id=VALID_UUID)
        self.assertEqual(device_id, VALID_UUID)

    def test_rejects_a_malformed_requested_id(self):
        response = _FakeResponse()
        for value in PATH_TRAVERSAL_VALUES + ABSOLUTE_PATH_VALUES:
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as ctx:
                    identity.issue_device(response, secure=False, requested_device_id=value)
                self.assertEqual(ctx.exception.status_code, 400)
        # A rejected request must not have set a cookie for the bad value.
        self.assertEqual(response.cookies, [])

    def test_issued_cookie_round_trips_through_get_owner_id(self):
        response = _FakeResponse()
        device_id = identity.issue_device(response, secure=False)
        signed_value = response.cookies[0]["value"]
        self.assertEqual(
            identity.get_owner_id(device_cookie=signed_value, device_header=None), device_id
        )

    def test_existing_valid_cookie_wins_over_a_differing_requested_id(self):
        """The reconciliation rule: a live cookie beats a stale IndexedDB
        mirror sent in the same request, since the cookie is the one thing
        neither the client nor an attacker could have forged.
        """
        cookie_owner = VALID_UUID
        mirror_owner = "22222222-2222-4222-8222-222222222222"
        cookie = identity.make_device_cookie(cookie_owner)

        response = _FakeResponse()
        device_id = identity.issue_device(
            response, secure=False, requested_device_id=mirror_owner, existing_cookie=cookie
        )
        self.assertEqual(device_id, cookie_owner, "cookie must win over a differing requested id")

    def test_existing_valid_cookie_wins_even_with_no_requested_id(self):
        """Calling POST /api/device on every boot (as the frontend does)
        must not swap out an already-live device just because the body
        happened to omit device_id.
        """
        cookie = identity.make_device_cookie(VALID_UUID)
        response = _FakeResponse()
        device_id = identity.issue_device(response, secure=False, existing_cookie=cookie)
        self.assertEqual(device_id, VALID_UUID)

    def test_invalid_existing_cookie_falls_back_to_requested_id(self):
        mirror_owner = "22222222-2222-4222-8222-222222222222"
        response = _FakeResponse()
        device_id = identity.issue_device(
            response, secure=False, requested_device_id=mirror_owner, existing_cookie="garbage"
        )
        self.assertEqual(device_id, mirror_owner)

    def test_no_cookie_and_no_requested_id_still_mints(self):
        response = _FakeResponse()
        device_id = identity.issue_device(response, secure=False, existing_cookie=None)
        self.assertTrue(identity._is_canonical_uuid4(device_id))


if __name__ == "__main__":
    unittest.main()
