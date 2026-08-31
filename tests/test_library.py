"""The library is the app's memory of what songs exist.

Two invariants matter here and they pull in opposite directions:

  1. A song whose derived data is gone must still be LISTED, honestly, with
     its tapped grid intact — because that grid cannot be recomputed and
     hiding the row would look exactly like losing it.
  2. Re-importing the same file must be a no-op that RECONNECTS the song to
     everything it already had, not a fresh copy that orphans the grid.

Both hang on the md5 key, so most of these tests are really testing that the
key survives the thing being done to the file.

The endpoint functions are called directly rather than over HTTP — httpx (and
so starlette's TestClient) is not installed, and the routes are plain functions
anyway, so this exercises the real read/write path with no HTTP layer in the way.

Every route/helper that resolves a path now takes a device's identity —
either a raw owner_id (the plain helpers) or the same owner_id passed via
the `owner_id` kwarg (the routes, which take it via FastAPI's
Depends(identity.get_owner_id) in real traffic — a UUIDv4 string, not an
account). TEST_OWNER_ID below is one fixed fake device; setUp nests the
temp fixture directories one level deeper under it, so every
`self.tracks_dir`/`self.cache_dir`/etc. below still points exactly where the
code under test will actually read and write.
"""

import asyncio
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.requests import Request

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "api"))

import server

BEATS = [round(0.5 + i * 0.6, 3) for i in range(64)]
DOWNBEATS = list(range(0, 64, 4))
EIGHT_COUNTS = list(range(0, 64, 8))

TEST_OWNER_ID = "11111111-1111-4111-8111-111111111111"


class _Upload:
    """The attributes import_song touches on a Starlette UploadFile.

    read(size) matches UploadFile's real chunked-read signature — Phase 5's
    _read_upload_within_limit reads in fixed-size chunks rather than one
    unbounded call, so this fake has to actually chunk too, not just
    ignore the argument.
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


def _fake_request(path="/api/import"):
    """A minimal real starlette.requests.Request — slowapi's @limiter.limit
    decorator (now on import_song/get_analysis, see Phase 5) requires an
    actual Request instance (isinstance-checked) to read the caller's
    address off, even when the route is called directly rather than
    through the ASGI stack the way every test in this file does. Needs no
    real ASGI server behind it: Request only needs a scope dict shaped
    like one.
    """
    return Request({"type": "http", "path": path, "client": ("127.0.0.1", 0), "headers": []})


def _import(filename, data):
    return asyncio.run(
        server.import_song(_fake_request(), _Upload(filename, data), owner_id=TEST_OWNER_ID)
    )


class LibraryTestCase(unittest.TestCase):
    """Redirects every tree into a temp dir so no test can touch real data.

    server.TRACKS_DIR/CACHE_DIR/GRIDS_DIR/LIBRARY_DIR are patched to the
    BASE of each temp tree (matching what the real app points them at); the
    per-device subdirectory the app actually reads/writes is
    <base>/TEST_OWNER_ID/, which is what self.tracks_dir etc. below refer to.
    """

    def setUp(self):
        # import_song is now rate-limited per-IP (Phase 5); every test in
        # this file calls it through the same fake 127.0.0.1 request (see
        # _fake_request), so without a reset here the SUITE's cumulative
        # call count — not any one test's behavior — would eventually trip
        # CHOREO_IMPORT_RATE_LIMIT and start failing unrelated tests.
        server.limiter.reset()

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        base_cache_dir = root / "cache" / "stems"
        base_grids_dir = root / "data" / "grids"
        base_library_dir = root / "data" / "library"
        base_tracks_dir = root / "tracks"

        self.cache_dir = base_cache_dir / TEST_OWNER_ID
        self.grids_dir = base_grids_dir / TEST_OWNER_ID
        self.library_dir = base_library_dir / TEST_OWNER_ID
        self.tracks_dir = base_tracks_dir / TEST_OWNER_ID
        self.tracks_dir.mkdir(parents=True)

        self._patches = {
            "CACHE_DIR": server.CACHE_DIR,
            "GRIDS_DIR": server.GRIDS_DIR,
            "LIBRARY_DIR": server.LIBRARY_DIR,
            "TRACKS_DIR": server.TRACKS_DIR,
        }
        server.CACHE_DIR = base_cache_dir
        server.GRIDS_DIR = base_grids_dir
        server.LIBRARY_DIR = base_library_dir
        server.TRACKS_DIR = base_tracks_dir

    def tearDown(self):
        for name, value in self._patches.items():
            setattr(server, name, value)
        self.tmp.cleanup()

    # -- fixtures ---------------------------------------------------------

    def add_track(self, name="song.mp3", data=b"pretend this is an mp3"):
        path = self.tracks_dir / name
        path.write_bytes(data)
        return path

    def seed_stems(self, md5):
        d = self.cache_dir / md5
        d.mkdir(parents=True, exist_ok=True)
        (d / "analysis.json").write_text(json.dumps({"schema_version": 4, "duration": 180.5}))
        for stem in server.STEM_NAMES:
            (d / f"{stem}.wav").write_bytes(b"RIFF....derived audio")
        return d

    def seed_grid(self, md5):
        self.grids_dir.mkdir(parents=True, exist_ok=True)
        path = self.grids_dir / f"{md5}.json"
        path.write_text(
            json.dumps(
                {
                    "beats": BEATS,
                    "downbeats": DOWNBEATS,
                    "eight_counts": EIGHT_COUNTS,
                    "active": True,
                }
            )
        )
        return path

    def entry(self, md5):
        for song in server.list_library(owner_id=TEST_OWNER_ID)["songs"]:
            if song["md5"] == md5:
                return song
        return None


class TestTheThreeStates(LibraryTestCase):
    """Each state is reached by removing exactly one thing from a ready song."""

    def ready_song(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        self.seed_stems(md5)
        self.seed_grid(md5)
        return audio, md5

    def test_ready(self):
        _, md5 = self.ready_song()
        entry = self.entry(md5)
        self.assertEqual(entry["state"], "ready")
        self.assertTrue(entry["grid_present"])
        self.assertTrue(entry["stems_present"])
        self.assertTrue(entry["original_present"])

    def test_needs_tap_is_stems_without_a_grid(self):
        _, md5 = self.ready_song()
        (self.grids_dir / f"{md5}.json").unlink()

        entry = self.entry(md5)
        self.assertEqual(entry["state"], "needs_tap")
        self.assertFalse(entry["grid_present"])
        self.assertTrue(entry["stems_present"])

    def test_stems_evicted_is_the_source_file_going_missing(self):
        audio, md5 = self.ready_song()
        audio.unlink()

        entry = self.entry(md5)
        self.assertEqual(entry["state"], "stems_evicted")
        self.assertFalse(entry["original_present"])

    def test_an_evicted_entry_is_listed_not_hidden(self):
        audio, md5 = self.ready_song()
        audio.unlink()
        self.assertIsNotNone(self.entry(md5), "an evicted song vanished from the library")

    def test_an_evicted_entry_still_reports_its_grid_and_its_name(self):
        """The row's entire job is to say 'your tapped grid is still here'."""
        audio, md5 = self.ready_song()
        name = audio.name
        audio.unlink()

        entry = self.entry(md5)
        self.assertTrue(entry["grid_present"])
        self.assertEqual(entry["filename"], name, "nothing left to tell the user what to re-import")

    def test_listing_never_deletes_a_grid(self):
        audio, md5 = self.ready_song()
        audio.unlink()
        shutil.rmtree(self.cache_dir)
        for _ in range(3):
            server.list_library(owner_id=TEST_OWNER_ID)
        self.assertTrue((self.grids_dir / f"{md5}.json").exists())

    def test_missing_stems_alone_do_not_make_a_song_unopenable(self):
        """With the source in hand the app re-derives stems: slow, not dead."""
        _, md5 = self.ready_song()
        shutil.rmtree(self.cache_dir / md5)

        entry = self.entry(md5)
        self.assertEqual(entry["state"], "ready")
        self.assertFalse(entry["stems_present"], "the UI needs this to warn about the wait")

    def test_states_are_distinct(self):
        """Guards against a refactor that collapses two of them together."""
        ready = self.add_track("ready.mp3", b"ready song bytes")
        tap = self.add_track("tap.mp3", b"untapped song bytes")
        gone = self.add_track("gone.mp3", b"about to be evicted bytes")

        for audio in (ready, tap, gone):
            md5 = server._file_hash(audio)
            server._write_manifest(TEST_OWNER_ID, audio, md5, 100.0)
            self.seed_stems(md5)
        self.seed_grid(server._file_hash(ready))
        self.seed_grid(server._file_hash(gone))
        gone_md5 = server._file_hash(gone)
        gone.unlink()

        self.assertEqual(self.entry(server._file_hash(ready))["state"], "ready")
        self.assertEqual(self.entry(server._file_hash(tap))["state"], "needs_tap")
        self.assertEqual(self.entry(gone_md5)["state"], "stems_evicted")


class TestManifestsAreDurable(LibraryTestCase):
    def test_manifests_live_outside_the_cache(self):
        path = server._manifest_path(TEST_OWNER_ID, "deadbeef")
        self.assertFalse(
            (self.cache_dir.parent in path.parents),
            f"manifests must not live under the cache tree: {path}",
        )

    def test_manifest_survives_a_cache_wipe(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        self.seed_stems(md5)

        shutil.rmtree(self.cache_dir.parent)
        self.assertIsNotNone(self.entry(md5), "a cache wipe erased the library")

    def test_a_rename_keeps_one_entry_and_follows_the_new_name(self):
        audio = self.add_track("before.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        audio.rename(self.tracks_dir / "after.mp3")

        songs = server.list_library(owner_id=TEST_OWNER_ID)["songs"]
        self.assertEqual(len(songs), 1, "a rename split one song into two entries")
        self.assertEqual(songs[0]["filename"], "after.mp3")
        self.assertEqual(songs[0]["md5"], md5)


class TestMediaKind(LibraryTestCase):
    def test_kind_comes_from_the_extension(self):
        self.assertEqual(server._media_kind("clip.mp4"), "video")
        self.assertEqual(server._media_kind("clip.MOV"), "video")
        self.assertEqual(server._media_kind("song.mp3"), "audio")
        self.assertEqual(server._media_kind("song.wav"), "audio")

    def test_a_hand_marked_video_survives_re_analysis(self):
        """media_kind is the one field _write_manifest never reverts.

        Exercised here against a plain audio file marked "video" by hand,
        independent of TestVideoSource below (which covers the real path: a
        video file whose kind is derived from its own extension). If a later
        _write_manifest reverted a hand edit to "audio" the mark would be
        useless.
        """
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        record = json.loads(server._manifest_path(TEST_OWNER_ID, md5).read_text())
        record["media_kind"] = "video"
        server._manifest_path(TEST_OWNER_ID, md5).write_text(json.dumps(record))

        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)  # as a re-analysis would

        self.assertEqual(self.entry(md5)["media_kind"], "video")


class TestVideoSource(LibraryTestCase):
    """The real video path: a video file living in tracks/, not a hand-marked
    manifest. Covers exactly what TestMediaKind's docstring says it doesn't —
    _tracks_by_md5 recognizing a video extension, and a video track reaching
    the same three states an audio one does.
    """

    def test_a_video_source_reaches_ready_like_an_audio_one(self):
        video = self.add_track("clip.mp4", b"pretend this is an mp4")
        md5 = server._file_hash(video)
        server._write_manifest(TEST_OWNER_ID, video, md5, 12.0)
        self.seed_stems(md5)
        self.seed_grid(md5)

        entry = self.entry(md5)
        self.assertEqual(entry["state"], "ready")
        self.assertEqual(entry["media_kind"], "video")
        self.assertTrue(entry["original_present"])

    def test_track_file_resolves_a_video_id(self):
        video = self.add_track("clip.mp4", b"pretend this is an mp4")
        self.assertEqual(server._track_file(TEST_OWNER_ID, "clip"), video)

    def test_get_video_serves_the_source_file(self):
        video = self.add_track("clip.mp4", b"pretend this is an mp4")
        response = server.get_video("clip", owner_id=TEST_OWNER_ID)
        self.assertEqual(Path(response.path), video)

    def test_get_video_404s_for_an_audio_track(self):
        self.add_track("song.mp3", b"pretend this is an mp3")
        with self.assertRaises(server.HTTPException) as ctx:
            server.get_video("song", owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_resolve_analysis_audio_passes_through_an_audio_source(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        self.assertEqual(server._resolve_analysis_audio(TEST_OWNER_ID, audio, md5), audio)

    def test_resolve_analysis_audio_extracts_and_caches_a_video_source(self):
        video = self.add_track("clip.mp4", b"pretend this is an mp4")
        md5 = server._file_hash(video)

        fake_clip = MagicMock()

        def fake_write_audiofile(path, **kwargs):
            Path(path).write_bytes(b"RIFF....extracted audio")

        fake_clip.audio.write_audiofile.side_effect = fake_write_audiofile

        video_controls_module = types.ModuleType("video_controls")
        video_controls_module.load_video = MagicMock(return_value=fake_clip)
        with patch.dict(sys.modules, {"video_controls": video_controls_module}):
            wav_path = server._resolve_analysis_audio(TEST_OWNER_ID, video, md5)

            self.assertEqual(wav_path, self.cache_dir / md5 / "source_audio.wav")
            self.assertTrue(wav_path.exists())
            video_controls_module.load_video.assert_called_once_with(str(video))
            fake_clip.close.assert_called_once()

            # A second call must not re-extract: the wav is already cached.
            server._resolve_analysis_audio(TEST_OWNER_ID, video, md5)
            video_controls_module.load_video.assert_called_once()


class TestBackfillDoesNotLogTheDeviceId(LibraryTestCase):
    """_backfill_library prints a summary line whenever it writes something
    — the raw device id must never be part of it (see identity.owner_key's
    own docstring on why). Captures the actual stdout the function writes
    and asserts on it directly, rather than trusting the source doesn't
    regress silently.
    """

    def test_log_line_contains_the_hashed_key_not_the_raw_owner_id(self):
        import contextlib
        import io

        import identity

        audio = self.add_track()
        self.seed_stems(server._file_hash(audio))  # gives backfill something to write

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            written = server._backfill_library(TEST_OWNER_ID)
        output = captured.getvalue()

        self.assertEqual(written, 1, "test setup didn't actually trigger a backfill")
        self.assertNotIn(TEST_OWNER_ID, output)
        self.assertIn(identity.owner_key(TEST_OWNER_ID), output)


class TestBackfill(LibraryTestCase):
    def test_names_an_existing_analysis(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        self.seed_stems(md5)

        self.assertEqual(server._backfill_library(TEST_OWNER_ID), 1)
        entry = self.entry(md5)
        self.assertEqual(entry["filename"], "song.mp3")
        self.assertEqual(entry["duration"], 180.5)

    def test_registers_a_grid_that_nothing_can_name(self):
        """An orphan grid is unrecoverable user data; it must still show up."""
        self.seed_grid("f" * 32)
        self.assertEqual(server._backfill_library(TEST_OWNER_ID), 1)

        entry = self.entry("f" * 32)
        self.assertIsNotNone(entry, "an orphan grid was dropped from the library")
        self.assertEqual(entry["state"], "stems_evicted")
        self.assertTrue(entry["grid_present"])
        self.assertIsNone(entry["filename"])

    def test_skips_a_cache_dir_with_no_name_and_no_grid(self):
        """Nothing to show and nothing to lose — listing it is just noise."""
        self.seed_stems("a" * 32)
        self.assertEqual(server._backfill_library(TEST_OWNER_ID), 0)
        self.assertEqual(server.list_library(owner_id=TEST_OWNER_ID)["songs"], [])

    def test_is_idempotent(self):
        audio = self.add_track()
        self.seed_stems(server._file_hash(audio))
        self.assertEqual(server._backfill_library(TEST_OWNER_ID), 1)
        for _ in range(3):
            self.assertEqual(server._backfill_library(TEST_OWNER_ID), 0)
        self.assertEqual(len(server.list_library(owner_id=TEST_OWNER_ID)["songs"]), 1)

    def test_no_op_on_an_empty_install(self):
        self.assertEqual(server._backfill_library(TEST_OWNER_ID), 0)
        self.assertEqual(server.list_library(owner_id=TEST_OWNER_ID)["songs"], [])


class TestImport(LibraryTestCase):
    def test_round_trips_into_the_library(self):
        result = _import("New Song.mp3", b"brand new audio bytes")

        self.assertTrue(result["created"])
        self.assertEqual(result["id"], "New Song")
        self.assertTrue((self.tracks_dir / "New Song.mp3").exists())

        entry = self.entry(result["md5"])
        self.assertIsNotNone(entry)
        self.assertEqual(entry["state"], "needs_tap", "a fresh import has no grid yet")
        self.assertEqual(entry["filename"], "New Song.mp3")

    def test_reimporting_the_same_file_reconnects_the_existing_grid(self):
        """The documented recovery path out of stems_evicted."""
        data = b"the one song that matters"
        first = _import("song.mp3", data)
        md5 = first["md5"]
        self.seed_grid(md5)

        (self.tracks_dir / "song.mp3").unlink()
        self.assertEqual(self.entry(md5)["state"], "stems_evicted")

        again = _import("song.mp3", data)
        self.assertEqual(again["md5"], md5, "re-import produced a different key")

        entry = self.entry(md5)
        self.assertEqual(entry["state"], "ready")
        self.assertTrue(entry["grid_present"], "the recovered song lost its tapped grid")

    def test_reimport_does_not_duplicate_the_entry(self):
        data = b"identical bytes every time"
        _import("song.mp3", data)
        result = _import("song.mp3", data)

        self.assertFalse(result["created"])
        self.assertEqual(len(server.list_library(owner_id=TEST_OWNER_ID)["songs"]), 1)
        self.assertEqual(len(list(self.tracks_dir.iterdir())), 1)

    def test_a_name_collision_with_different_audio_keeps_both(self):
        first = _import("song.mp3", b"the first song")
        second = _import("song.mp3", b"a completely different song")

        self.assertNotEqual(first["md5"], second["md5"])
        self.assertEqual(second["filename"], "song (2).mp3")
        self.assertEqual(len(server.list_library(owner_id=TEST_OWNER_ID)["songs"]), 2)

    def test_rejects_a_path_traversal_filename(self):
        _import("../../evil.mp3", b"trying to escape tracks/")
        self.assertTrue((self.tracks_dir / "evil.mp3").exists())
        self.assertFalse((self.tracks_dir.parent / "evil.mp3").exists())

    def test_rejects_a_backslash_filename(self):
        """Path(...).name only strips '/' on this (POSIX) server — a
        backslash-based traversal payload survives it untouched and must
        be caught by an explicit check instead (see import_song's own
        comment on why: this module also runs under Windows in local dev).
        """
        with self.assertRaises(server.HTTPException) as ctx:
            _import("..\\..\\evil.mp3", b"backslash traversal attempt")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_a_leading_dot_filename(self):
        with self.assertRaises(server.HTTPException) as ctx:
            _import(".hidden.mp3", b"dotfile attempt")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_an_overlong_filename(self):
        long_name = ("a" * (server.MAX_FILENAME_LENGTH + 1)) + ".mp3"
        with self.assertRaises(server.HTTPException) as ctx:
            _import(long_name, b"absurdly long filename")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_accepts_a_filename_right_at_the_length_limit(self):
        name = ("a" * (server.MAX_FILENAME_LENGTH - 4)) + ".mp3"  # exactly MAX_FILENAME_LENGTH
        self.assertEqual(len(name), server.MAX_FILENAME_LENGTH)
        result = _import(name, b"right at the boundary")
        self.assertTrue(result["created"])

    def test_rejects_an_unsupported_type(self):
        with self.assertRaises(server.HTTPException) as ctx:
            _import("notes.txt", b"not audio")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_round_trips_a_video_into_the_library(self):
        result = _import("clip.mp4", b"pretend this is an mp4")

        self.assertTrue(result["created"])
        self.assertEqual(result["id"], "clip")
        self.assertTrue((self.tracks_dir / "clip.mp4").exists())

        entry = self.entry(result["md5"])
        self.assertIsNotNone(entry)
        self.assertEqual(entry["media_kind"], "video")
        self.assertEqual(entry["state"], "needs_tap", "a fresh import has no grid yet")

    def test_rejects_an_empty_file(self):
        with self.assertRaises(server.HTTPException) as ctx:
            _import("song.mp3", b"")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_an_import_over_the_storage_quota(self):
        original = server.CHOREO_MAX_STORAGE_BYTES
        server.CHOREO_MAX_STORAGE_BYTES = 10
        try:
            with self.assertRaises(server.HTTPException) as ctx:
                _import("song.mp3", b"this is well over ten bytes")
            self.assertEqual(ctx.exception.status_code, 413)
        finally:
            server.CHOREO_MAX_STORAGE_BYTES = original

    def test_rejects_an_import_over_the_track_count_limit(self):
        original = server.CHOREO_MAX_TRACKS
        server.CHOREO_MAX_TRACKS = 1
        try:
            _import("first.mp3", b"the first song")
            with self.assertRaises(server.HTTPException) as ctx:
                _import("second.mp3", b"a different song entirely")
            self.assertEqual(ctx.exception.status_code, 413)
        finally:
            server.CHOREO_MAX_TRACKS = original

    def test_reimporting_identical_bytes_is_exempt_from_the_quota(self):
        """A quota that blocked its own recovery path would be a trap."""
        data = b"the one song that matters"
        first = _import("song.mp3", data)

        original = server.CHOREO_MAX_STORAGE_BYTES
        server.CHOREO_MAX_STORAGE_BYTES = 0
        try:
            again = _import("song.mp3", data)
            self.assertEqual(again["md5"], first["md5"])
        finally:
            server.CHOREO_MAX_STORAGE_BYTES = original


class TestProjectExportImport(LibraryTestCase):
    """Phase 4: no account means no password-reset-style recovery, so a
    device can export its own project state to a file and import it back
    in — on the same device after data loss, or on a different one
    entirely. Both routes are keyed by md5 alone, deliberately independent
    of whether the source audio has ever been (re-)imported — see the
    module note above get_manual_grid_by_md5/import_project in server.py.
    """

    MD5 = "b" * 32

    def _grid_payload(self, beats=None):
        return server.ManualGrid(beats=beats or BEATS, downbeats=DOWNBEATS, eight_counts=EIGHT_COUNTS)

    # -- GET .../manual-grid (the export read path) ------------------------

    def test_get_by_md5_returns_null_when_nothing_exists(self):
        self.assertIsNone(
            server.get_manual_grid_by_md5(self.MD5, owner_id=TEST_OWNER_ID)["manual_grid"]
        )

    def test_get_by_md5_reads_a_grid_with_no_source_file_at_all(self):
        """The whole point: export must work on a stems_evicted-shaped row,
        not just a fully ready one."""
        self.seed_grid(self.MD5)
        stored = server.get_manual_grid_by_md5(self.MD5, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertEqual(stored["beats"], BEATS)
        self.assertEqual(stored["downbeats"], DOWNBEATS)
        self.assertEqual(stored["eight_counts"], EIGHT_COUNTS)

    def test_get_by_md5_rejects_a_malformed_md5(self):
        for value in ["..", "", "not-an-md5", "A" * 32, "../../etc/passwd"]:
            with self.subTest(value=value):
                with self.assertRaises(server.HTTPException) as ctx:
                    server.get_manual_grid_by_md5(value, owner_id=TEST_OWNER_ID)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_get_by_md5_is_isolated_per_device(self):
        self.seed_grid(self.MD5)
        other = "22222222-2222-4222-8222-222222222222"
        self.assertIsNone(server.get_manual_grid_by_md5(self.MD5, owner_id=other)["manual_grid"])

    # -- POST .../import (the restore path) ---------------------------------

    def test_import_creates_a_manifest_and_grid_with_no_source_file(self):
        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(
                md5=self.MD5, filename="My Song.mp3", media_kind="audio", duration=180.5
            ),
            manual_grid=self._grid_payload(),
        )
        result = server.import_project(payload, owner_id=TEST_OWNER_ID)
        self.assertEqual(result["md5"], self.MD5)
        self.assertTrue(result["manual_grid_imported"])

        entry = self.entry(self.MD5)
        self.assertIsNotNone(entry, "import must create a library entry with no source file present")
        self.assertEqual(entry["state"], "stems_evicted", "no source file was ever imported")
        self.assertEqual(entry["filename"], "My Song.mp3")
        self.assertEqual(entry["duration"], 180.5)
        self.assertTrue(entry["grid_present"])

        stored = server.get_manual_grid_by_md5(self.MD5, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertEqual(stored["beats"], BEATS)

    def test_import_without_a_grid_only_writes_the_manifest(self):
        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(md5=self.MD5, filename="No Grid Yet.mp3"),
            manual_grid=None,
        )
        result = server.import_project(payload, owner_id=TEST_OWNER_ID)
        self.assertFalse(result["manual_grid_imported"])

        entry = self.entry(self.MD5)
        self.assertIsNotNone(entry)
        self.assertFalse(entry["grid_present"])

    def test_import_overwrites_an_existing_grid_wholesale(self):
        self.seed_grid(self.MD5)
        newer = [round(b + 5, 3) for b in BEATS]
        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(md5=self.MD5, filename="song.mp3"),
            manual_grid=self._grid_payload(newer),
        )
        server.import_project(payload, owner_id=TEST_OWNER_ID)

        stored = server.get_manual_grid_by_md5(self.MD5, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertEqual(stored["beats"], newer, "import must replace, not merge with, an existing grid")

    def test_import_preserves_a_hand_marked_media_kind(self):
        """Same rule _write_manifest already follows for a live re-analysis
        — see TestMediaKind — now exercised through the import path too."""
        server._write_manifest_fields(TEST_OWNER_ID, self.MD5, "clip.mp3", "audio")
        record = json.loads(server._manifest_path(TEST_OWNER_ID, self.MD5).read_text())
        record["media_kind"] = "video"
        server._manifest_path(TEST_OWNER_ID, self.MD5).write_text(json.dumps(record))

        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(md5=self.MD5, filename="clip.mp3", media_kind="audio"),
        )
        server.import_project(payload, owner_id=TEST_OWNER_ID)

        self.assertEqual(self.entry(self.MD5)["media_kind"], "video")

    def test_import_rejects_a_mismatched_schema_version(self):
        payload = server.ProjectImport(
            schema_version=3,
            track=server.ProjectTrack(md5=self.MD5, filename="song.mp3"),
            manual_grid=self._grid_payload(),
        )
        with self.assertRaises(server.HTTPException) as ctx:
            server.import_project(payload, owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIsNone(self.entry(self.MD5), "a rejected import must not write anything")

    def test_import_rejects_a_malformed_md5(self):
        for value in ["..", "", "not-an-md5", "../../etc/passwd"]:
            with self.subTest(value=value):
                payload = server.ProjectImport(
                    schema_version=4,
                    track=server.ProjectTrack(md5=value, filename="song.mp3"),
                )
                with self.assertRaises(server.HTTPException) as ctx:
                    server.import_project(payload, owner_id=TEST_OWNER_ID)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_import_strips_a_path_traversal_filename_to_its_bare_name(self):
        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(md5=self.MD5, filename="../../etc/evil.mp3"),
        )
        server.import_project(payload, owner_id=TEST_OWNER_ID)
        self.assertEqual(self.entry(self.MD5)["filename"], "evil.mp3")

    def test_import_is_isolated_per_device(self):
        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(md5=self.MD5, filename="song.mp3"),
            manual_grid=self._grid_payload(),
        )
        server.import_project(payload, owner_id=TEST_OWNER_ID)

        other = "22222222-2222-4222-8222-222222222222"
        self.assertEqual(server.list_library(owner_id=other)["songs"], [])
        self.assertIsNone(server.get_manual_grid_by_md5(self.MD5, owner_id=other)["manual_grid"])

    # -- round trip -----------------------------------------------------------

    def test_export_then_import_round_trips_losslessly(self):
        """The actual contract: export a real grid, wipe it server-side
        (simulating total data loss on this device), import the exported
        data back in, and confirm every array is byte-identical — not just
        equal after some reasonable rounding.
        """
        audio = self.add_track("Round Trip.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 222.222)
        self.seed_grid(md5)

        # "Export": read exactly what the frontend's export would read.
        exported_grid = server.get_manual_grid_by_md5(md5, owner_id=TEST_OWNER_ID)["manual_grid"]
        exported_entry = self.entry(md5)

        # Total loss: delete the track, its grid, and its manifest — as if
        # this were a fresh device with only the export file in hand.
        server.delete_track(md5, owner_id=TEST_OWNER_ID)
        self.assertIsNone(self.entry(md5))

        payload = server.ProjectImport(
            schema_version=4,
            track=server.ProjectTrack(
                md5=md5,
                filename=exported_entry["filename"],
                media_kind=exported_entry["media_kind"],
                duration=exported_entry["duration"],
            ),
            manual_grid=server.ManualGrid(
                beats=exported_grid["beats"],
                downbeats=exported_grid["downbeats"],
                eight_counts=exported_grid["eight_counts"],
            ),
        )
        server.import_project(payload, owner_id=TEST_OWNER_ID)

        restored_grid = server.get_manual_grid_by_md5(md5, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertEqual(restored_grid["beats"], exported_grid["beats"])
        self.assertEqual(restored_grid["downbeats"], exported_grid["downbeats"])
        self.assertEqual(restored_grid["eight_counts"], exported_grid["eight_counts"])

        restored_entry = self.entry(md5)
        self.assertEqual(restored_entry["filename"], exported_entry["filename"])
        self.assertEqual(restored_entry["media_kind"], exported_entry["media_kind"])
        self.assertEqual(restored_entry["duration"], exported_entry["duration"])
        self.assertTrue(restored_entry["grid_present"])


class TestRename(LibraryTestCase):
    def test_renames_the_file_and_keeps_the_extension(self):
        audio = self.add_track("before.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        result = server.rename_track(md5, server.RenameTrack(name="after"), owner_id=TEST_OWNER_ID)

        self.assertEqual(result["filename"], "after.mp3")
        self.assertTrue((self.tracks_dir / "after.mp3").exists())
        self.assertFalse((self.tracks_dir / "before.mp3").exists())

    def test_keeps_the_same_md5_and_grid(self):
        """A rename must not fork the song into a second library entry."""
        audio = self.add_track("before.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        self.seed_grid(md5)

        server.rename_track(md5, server.RenameTrack(name="after"), owner_id=TEST_OWNER_ID)

        songs = server.list_library(owner_id=TEST_OWNER_ID)["songs"]
        self.assertEqual(len(songs), 1, "a rename split one song into two entries")
        self.assertEqual(songs[0]["md5"], md5)
        self.assertEqual(songs[0]["filename"], "after.mp3")
        self.assertTrue(songs[0]["grid_present"])

    def test_a_no_op_rename_to_its_own_name_succeeds(self):
        audio = self.add_track("song.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        result = server.rename_track(md5, server.RenameTrack(name="song"), owner_id=TEST_OWNER_ID)
        self.assertEqual(result["filename"], "song.mp3")

    def test_rejects_an_empty_name(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="   "), owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_a_path_traversal_name(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="../evil"), owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertTrue(audio.exists(), "the original file must be left alone on a rejected rename")

    def test_conflicts_with_an_existing_file(self):
        self.add_track("taken.mp3")
        audio = self.add_track("mine.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="taken"), owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(audio.exists(), "a rejected rename must not touch the source file")

    def test_404s_when_the_source_file_is_evicted(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        audio.unlink()

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="whatever"), owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_rejects_a_malformed_md5(self):
        for value in ["..", "", "not-an-md5", "a" * 31, "A" * 32, "../../etc/passwd"]:
            with self.subTest(value=value):
                with self.assertRaises(server.HTTPException) as ctx:
                    server.rename_track(value, server.RenameTrack(name="whatever"), owner_id=TEST_OWNER_ID)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_a_leading_dot_name(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name=".hidden"), owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertTrue(audio.exists())

    def test_rejects_an_overlong_name(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(
                md5, server.RenameTrack(name="a" * (server.MAX_FILENAME_LENGTH + 1)), owner_id=TEST_OWNER_ID
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertTrue(audio.exists())


class TestDelete(LibraryTestCase):
    def test_removes_the_file_stems_grid_and_manifest(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        self.seed_stems(md5)
        self.seed_grid(md5)

        result = server.delete_track(md5, owner_id=TEST_OWNER_ID)

        self.assertEqual(result["deleted"], md5)
        self.assertFalse(audio.exists())
        self.assertFalse((self.cache_dir / md5).exists())
        self.assertFalse((self.grids_dir / f"{md5}.json").exists())
        self.assertFalse(server._manifest_path(TEST_OWNER_ID, md5).exists())
        self.assertIsNone(self.entry(md5), "a deleted song must not still be listed")

    def test_is_unrecoverable_by_re_import(self):
        """Unlike an eviction, a delete also drops the manifest that would
        have reconnected a re-import to the old grid."""
        data = b"the one song that matters"
        first = _import("song.mp3", data)
        md5 = first["md5"]
        self.seed_grid(md5)

        server.delete_track(md5, owner_id=TEST_OWNER_ID)
        again = _import("song.mp3", data)

        self.assertEqual(again["md5"], md5, "re-import of identical bytes always hashes the same")
        entry = self.entry(md5)
        self.assertEqual(entry["state"], "needs_tap", "the old grid must not come back after a delete")
        self.assertFalse(entry["grid_present"])

    def test_is_idempotent(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        self.seed_stems(md5)
        self.seed_grid(md5)

        server.delete_track(md5, owner_id=TEST_OWNER_ID)
        server.delete_track(md5, owner_id=TEST_OWNER_ID)  # must not raise

        self.assertIsNone(self.entry(md5))

    def test_works_on_an_evicted_entry_with_no_source_file(self):
        """The row's whole point at that state is to still be deletable."""
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_OWNER_ID, audio, md5, 180.5)
        self.seed_grid(md5)
        audio.unlink()

        server.delete_track(md5, owner_id=TEST_OWNER_ID)
        self.assertIsNone(self.entry(md5))
        self.assertFalse((self.grids_dir / f"{md5}.json").exists())

    def test_rejects_a_path_traversal_md5_instead_of_wiping_the_whole_cache_tree(self):
        """The regression test for a real bug: delete_track built
        `CACHE_DIR / owner_id / md5` directly from the URL's md5 segment
        with no ".json" suffix to absorb a bare "..", so md5=".." resolved
        to CACHE_DIR/owner_id itself and shutil.rmtree'd it — deleting
        EVERY track's cached stems for this device (owner_id's own
        segment is what ".." strips) in one call, from nothing more than
        a single unencoded ".." — no multi-segment tricks needed. Proven
        here by seeding a real, unrelated, legitimately-cached track and
        confirming it survives an attempted md5=".." delete.
        """
        real_md5 = "a" * 32
        self.seed_stems(real_md5)
        self.assertTrue((self.cache_dir / real_md5).exists(), "test setup didn't seed the cache")

        with self.assertRaises(server.HTTPException) as ctx:
            server.delete_track("..", owner_id=TEST_OWNER_ID)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertTrue(
            (self.cache_dir / real_md5).exists(),
            "a rejected md5=\"..\" delete must not touch any other track's cache",
        )

    def test_rejects_other_malformed_md5_values(self):
        for value in ["", "not-an-md5", "a" * 31, "a" * 33, "A" * 32, "../../etc/passwd", "/etc/passwd"]:
            with self.subTest(value=value):
                with self.assertRaises(server.HTTPException) as ctx:
                    server.delete_track(value, owner_id=TEST_OWNER_ID)
                self.assertEqual(ctx.exception.status_code, 400)


class TestDeviceIsolation(LibraryTestCase):
    """The point of the whole refactor: two devices never see each other's
    tracks, even when they share a base directory tree and identical bytes.
    """

    OTHER_OWNER_ID = "22222222-2222-4222-8222-222222222222"

    def test_second_device_starts_with_an_empty_library(self):
        _import("song.mp3", b"the first device's song")
        self.assertEqual(server.list_library(owner_id=self.OTHER_OWNER_ID)["songs"], [])

    def test_identical_bytes_get_separate_stem_caches(self):
        """The concrete proof MD5 dedup is per-device, not global."""
        data = b"the exact same audio bytes"
        first = _import("song.mp3", data)
        md5 = first["md5"]
        self.seed_stems(md5)  # only under TEST_OWNER_ID's cache_dir

        other_cache_dir = server.CACHE_DIR / self.OTHER_OWNER_ID / md5
        self.assertFalse(
            other_cache_dir.exists(),
            "a second device must not see the first device's cached stems",
        )

    def test_a_track_id_cannot_be_read_across_devices(self):
        _import("song.mp3", b"only the first device should see this")
        with self.assertRaises(server.HTTPException) as ctx:
            server._track_file(self.OTHER_OWNER_ID, "song")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
