"""Phase 5: a hard upload size cap and a basic per-IP rate limit on the two
endpoints that are now both expensive and reachable with no accounts behind
them — POST /api/import (disk writes + hashing) and GET .../analysis (can
trigger a multi-minute Demucs run on a cache miss). See server.py's own
comment above CHOREO_IMPORT_RATE_LIMIT for why per-IP, not per-device: an
owner_id is free to mint (no accounts — see identity.py's BEARER CREDENTIAL
note), so a per-device limit would cost an abuser nothing to route around.

Every route function here is called directly, matching every other test
file's own convention (see test_library.py's module docstring for why).
slowapi's @limiter.limit decorator needs a real starlette Request to read
the caller's address off even in that direct-call path — see _request.
RateLimitExceeded is a SIBLING of fastapi.HTTPException (both extend
starlette.exceptions.HTTPException, neither extends the other), so it's
asserted on by its own type here, not server.HTTPException.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "api"))

import server

TEST_OWNER_ID = "33333333-3333-4333-8333-333333333333"


class _Upload:
    """Matches test_library.py's own fake — read(size) chunks, since
    _read_upload_within_limit never does one unbounded read.
    """

    def __init__(self, filename, data):
        self.filename = filename
        self._data = data
        self._pos = 0

    async def read(self, size=-1):
        if size is None or size < 0:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def _request(client_ip="203.0.113.1", path="/api/import"):
    return Request({"type": "http", "path": path, "client": (client_ip, 0), "headers": []})


class RateLimitTestCase(unittest.TestCase):
    """Redirects every tree into a temp dir (import_song writes real
    files) and resets slowapi's storage before AND after every test — a
    hit recorded here must never leak into another test file's own call
    count (test_library.py resets independently, for the same reason).
    """

    def setUp(self):
        server.limiter.reset()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._tree_patches = {
            "CACHE_DIR": server.CACHE_DIR,
            "GRIDS_DIR": server.GRIDS_DIR,
            "LIBRARY_DIR": server.LIBRARY_DIR,
            "TRACKS_DIR": server.TRACKS_DIR,
        }
        server.CACHE_DIR = root / "cache" / "stems"
        server.GRIDS_DIR = root / "data" / "grids"
        server.LIBRARY_DIR = root / "data" / "library"
        server.TRACKS_DIR = root / "tracks"

    def tearDown(self):
        for name, value in self._tree_patches.items():
            setattr(server, name, value)
        self.tmp.cleanup()
        server.limiter.reset()

    def _import(self, filename, data, client_ip="203.0.113.1"):
        return asyncio.run(
            server.import_song(_request(client_ip), _Upload(filename, data), owner_id=TEST_OWNER_ID)
        )


class TestUploadSizeLimit(RateLimitTestCase):
    def test_rejects_an_upload_over_the_limit(self):
        original = server.CHOREO_MAX_UPLOAD_BYTES
        server.CHOREO_MAX_UPLOAD_BYTES = 1000
        try:
            with self.assertRaises(server.HTTPException) as ctx:
                self._import("big.mp3", b"x" * 2000)
            self.assertEqual(ctx.exception.status_code, 413)
        finally:
            server.CHOREO_MAX_UPLOAD_BYTES = original

    def test_accepts_an_upload_right_at_the_limit(self):
        original = server.CHOREO_MAX_UPLOAD_BYTES
        server.CHOREO_MAX_UPLOAD_BYTES = 1000
        try:
            result = self._import("ok.mp3", b"x" * 1000)
            self.assertTrue(result["created"])
        finally:
            server.CHOREO_MAX_UPLOAD_BYTES = original

    def test_rejects_a_payload_far_larger_than_the_limit_without_reading_it_all(self):
        """The real guarantee (bounded overread, not "eventually rejects
        after buffering everything") lives in _read_upload_within_limit's
        chunked loop; this just confirms a wildly oversized upload is
        still rejected rather than, say, silently truncated and accepted.
        """
        original = server.CHOREO_MAX_UPLOAD_BYTES
        server.CHOREO_MAX_UPLOAD_BYTES = 10
        try:
            with self.assertRaises(server.HTTPException) as ctx:
                self._import("huge.mp3", b"x" * (5 * 1024 * 1024))
            self.assertEqual(ctx.exception.status_code, 413)
        finally:
            server.CHOREO_MAX_UPLOAD_BYTES = original


class TestPerIpImportRateLimit(RateLimitTestCase):
    def test_the_same_ip_is_blocked_after_the_configured_number_of_requests(self):
        original = server.CHOREO_IMPORT_RATE_LIMIT
        server.CHOREO_IMPORT_RATE_LIMIT = "3/minute"
        try:
            for n in range(3):
                result = self._import(f"song{n}.mp3", f"bytes for song {n}".encode())
                self.assertTrue(result["created"])
            with self.assertRaises(RateLimitExceeded) as ctx:
                self._import("song4.mp3", b"one request too many")
            self.assertEqual(ctx.exception.status_code, 429)
        finally:
            server.CHOREO_IMPORT_RATE_LIMIT = original

    def test_a_different_ip_is_not_affected_by_another_ips_limit(self):
        original = server.CHOREO_IMPORT_RATE_LIMIT
        server.CHOREO_IMPORT_RATE_LIMIT = "1/minute"
        try:
            self._import("song.mp3", b"first ip's upload", client_ip="203.0.113.1")
            with self.assertRaises(RateLimitExceeded):
                self._import("song2.mp3", b"first ip again", client_ip="203.0.113.1")
            # A second, distinct address must still get its own full quota.
            result = self._import("song.mp3", b"second ip's upload", client_ip="198.51.100.7")
            self.assertTrue(result["created"])
        finally:
            server.CHOREO_IMPORT_RATE_LIMIT = original


class TestPerIpAnalysisRateLimit(RateLimitTestCase):
    def test_the_same_ip_is_blocked_after_the_configured_number_of_requests(self):
        """get_analysis is heavier to exercise for real (Demucs) — this
        only needs to prove the decorator is actually wired to it, so it
        drives the limit down to zero and confirms the very first call is
        rejected before any track/analysis lookup happens at all.
        """
        original = server.CHOREO_ANALYSIS_RATE_LIMIT
        server.CHOREO_ANALYSIS_RATE_LIMIT = "0/minute"
        try:
            with self.assertRaises(RateLimitExceeded) as ctx:
                asyncio.run(
                    server.get_analysis(
                        _request(path="/api/tracks/whatever/analysis"),
                        "whatever",
                        owner_id=TEST_OWNER_ID,
                    )
                )
            self.assertEqual(ctx.exception.status_code, 429)
        finally:
            server.CHOREO_ANALYSIS_RATE_LIMIT = original


if __name__ == "__main__":
    unittest.main()
