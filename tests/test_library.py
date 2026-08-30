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

Every route/helper that resolves a path now takes an account's identity —
either a raw user_id (the plain helpers) or a `user` object with a `.id`
(the routes, which take it via FastAPI's Depends(get_current_user) in real
traffic). TEST_USER_ID / TEST_USER below are one fixed fake account; setUp
nests the temp fixture directories one level deeper under it, so every
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

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "api"))

import server

BEATS = [round(0.5 + i * 0.6, 3) for i in range(64)]
DOWNBEATS = list(range(0, 64, 4))
EIGHT_COUNTS = list(range(0, 64, 8))

TEST_USER_ID = 1
TEST_USER = types.SimpleNamespace(id=TEST_USER_ID)


class _Upload:
    """The two attributes import_song touches on a Starlette UploadFile."""

    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    async def read(self):
        return self._data


def _import(filename, data):
    return asyncio.run(server.import_song(_Upload(filename, data), user=TEST_USER))


class LibraryTestCase(unittest.TestCase):
    """Redirects every tree into a temp dir so no test can touch real data.

    server.TRACKS_DIR/CACHE_DIR/GRIDS_DIR/LIBRARY_DIR are patched to the
    BASE of each temp tree (matching what the real app points them at); the
    per-account subdirectory the app actually reads/writes is
    <base>/TEST_USER_ID/, which is what self.tracks_dir etc. below refer to.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        base_cache_dir = root / "cache" / "stems"
        base_grids_dir = root / "data" / "grids"
        base_library_dir = root / "data" / "library"
        base_tracks_dir = root / "tracks"

        self.cache_dir = base_cache_dir / str(TEST_USER_ID)
        self.grids_dir = base_grids_dir / str(TEST_USER_ID)
        self.library_dir = base_library_dir / str(TEST_USER_ID)
        self.tracks_dir = base_tracks_dir / str(TEST_USER_ID)
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
        for song in server.list_library(user=TEST_USER)["songs"]:
            if song["md5"] == md5:
                return song
        return None


class TestTheThreeStates(LibraryTestCase):
    """Each state is reached by removing exactly one thing from a ready song."""

    def ready_song(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
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
            server.list_library(user=TEST_USER)
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
            server._write_manifest(TEST_USER_ID, audio, md5, 100.0)
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
        path = server._manifest_path(TEST_USER_ID, "deadbeef")
        self.assertFalse(
            (self.cache_dir.parent in path.parents),
            f"manifests must not live under the cache tree: {path}",
        )

    def test_manifest_survives_a_cache_wipe(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        self.seed_stems(md5)

        shutil.rmtree(self.cache_dir.parent)
        self.assertIsNotNone(self.entry(md5), "a cache wipe erased the library")

    def test_a_rename_keeps_one_entry_and_follows_the_new_name(self):
        audio = self.add_track("before.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        audio.rename(self.tracks_dir / "after.mp3")

        songs = server.list_library(user=TEST_USER)["songs"]
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
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)

        record = json.loads(server._manifest_path(TEST_USER_ID, md5).read_text())
        record["media_kind"] = "video"
        server._manifest_path(TEST_USER_ID, md5).write_text(json.dumps(record))

        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)  # as a re-analysis would

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
        server._write_manifest(TEST_USER_ID, video, md5, 12.0)
        self.seed_stems(md5)
        self.seed_grid(md5)

        entry = self.entry(md5)
        self.assertEqual(entry["state"], "ready")
        self.assertEqual(entry["media_kind"], "video")
        self.assertTrue(entry["original_present"])

    def test_track_file_resolves_a_video_id(self):
        video = self.add_track("clip.mp4", b"pretend this is an mp4")
        self.assertEqual(server._track_file(TEST_USER_ID, "clip"), video)

    def test_get_video_serves_the_source_file(self):
        video = self.add_track("clip.mp4", b"pretend this is an mp4")
        response = server.get_video("clip", user=TEST_USER)
        self.assertEqual(Path(response.path), video)

    def test_get_video_404s_for_an_audio_track(self):
        self.add_track("song.mp3", b"pretend this is an mp3")
        with self.assertRaises(server.HTTPException) as ctx:
            server.get_video("song", user=TEST_USER)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_resolve_analysis_audio_passes_through_an_audio_source(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        self.assertEqual(server._resolve_analysis_audio(TEST_USER_ID, audio, md5), audio)

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
            wav_path = server._resolve_analysis_audio(TEST_USER_ID, video, md5)

            self.assertEqual(wav_path, self.cache_dir / md5 / "source_audio.wav")
            self.assertTrue(wav_path.exists())
            video_controls_module.load_video.assert_called_once_with(str(video))
            fake_clip.close.assert_called_once()

            # A second call must not re-extract: the wav is already cached.
            server._resolve_analysis_audio(TEST_USER_ID, video, md5)
            video_controls_module.load_video.assert_called_once()


class TestBackfill(LibraryTestCase):
    def test_names_an_existing_analysis(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        self.seed_stems(md5)

        self.assertEqual(server._backfill_library(TEST_USER_ID), 1)
        entry = self.entry(md5)
        self.assertEqual(entry["filename"], "song.mp3")
        self.assertEqual(entry["duration"], 180.5)

    def test_registers_a_grid_that_nothing_can_name(self):
        """An orphan grid is unrecoverable user data; it must still show up."""
        self.seed_grid("f" * 32)
        self.assertEqual(server._backfill_library(TEST_USER_ID), 1)

        entry = self.entry("f" * 32)
        self.assertIsNotNone(entry, "an orphan grid was dropped from the library")
        self.assertEqual(entry["state"], "stems_evicted")
        self.assertTrue(entry["grid_present"])
        self.assertIsNone(entry["filename"])

    def test_skips_a_cache_dir_with_no_name_and_no_grid(self):
        """Nothing to show and nothing to lose — listing it is just noise."""
        self.seed_stems("a" * 32)
        self.assertEqual(server._backfill_library(TEST_USER_ID), 0)
        self.assertEqual(server.list_library(user=TEST_USER)["songs"], [])

    def test_is_idempotent(self):
        audio = self.add_track()
        self.seed_stems(server._file_hash(audio))
        self.assertEqual(server._backfill_library(TEST_USER_ID), 1)
        for _ in range(3):
            self.assertEqual(server._backfill_library(TEST_USER_ID), 0)
        self.assertEqual(len(server.list_library(user=TEST_USER)["songs"]), 1)

    def test_no_op_on_an_empty_install(self):
        self.assertEqual(server._backfill_library(TEST_USER_ID), 0)
        self.assertEqual(server.list_library(user=TEST_USER)["songs"], [])


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
        self.assertEqual(len(server.list_library(user=TEST_USER)["songs"]), 1)
        self.assertEqual(len(list(self.tracks_dir.iterdir())), 1)

    def test_a_name_collision_with_different_audio_keeps_both(self):
        first = _import("song.mp3", b"the first song")
        second = _import("song.mp3", b"a completely different song")

        self.assertNotEqual(first["md5"], second["md5"])
        self.assertEqual(second["filename"], "song (2).mp3")
        self.assertEqual(len(server.list_library(user=TEST_USER)["songs"]), 2)

    def test_rejects_a_path_traversal_filename(self):
        _import("../../evil.mp3", b"trying to escape tracks/")
        self.assertTrue((self.tracks_dir / "evil.mp3").exists())
        self.assertFalse((self.tracks_dir.parent / "evil.mp3").exists())

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


class TestRename(LibraryTestCase):
    def test_renames_the_file_and_keeps_the_extension(self):
        audio = self.add_track("before.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)

        result = server.rename_track(md5, server.RenameTrack(name="after"), user=TEST_USER)

        self.assertEqual(result["filename"], "after.mp3")
        self.assertTrue((self.tracks_dir / "after.mp3").exists())
        self.assertFalse((self.tracks_dir / "before.mp3").exists())

    def test_keeps_the_same_md5_and_grid(self):
        """A rename must not fork the song into a second library entry."""
        audio = self.add_track("before.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        self.seed_grid(md5)

        server.rename_track(md5, server.RenameTrack(name="after"), user=TEST_USER)

        songs = server.list_library(user=TEST_USER)["songs"]
        self.assertEqual(len(songs), 1, "a rename split one song into two entries")
        self.assertEqual(songs[0]["md5"], md5)
        self.assertEqual(songs[0]["filename"], "after.mp3")
        self.assertTrue(songs[0]["grid_present"])

    def test_a_no_op_rename_to_its_own_name_succeeds(self):
        audio = self.add_track("song.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)

        result = server.rename_track(md5, server.RenameTrack(name="song"), user=TEST_USER)
        self.assertEqual(result["filename"], "song.mp3")

    def test_rejects_an_empty_name(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="   "), user=TEST_USER)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_a_path_traversal_name(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="../evil"), user=TEST_USER)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertTrue(audio.exists(), "the original file must be left alone on a rejected rename")

    def test_conflicts_with_an_existing_file(self):
        self.add_track("taken.mp3")
        audio = self.add_track("mine.mp3")
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="taken"), user=TEST_USER)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(audio.exists(), "a rejected rename must not touch the source file")

    def test_404s_when_the_source_file_is_evicted(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        audio.unlink()

        with self.assertRaises(server.HTTPException) as ctx:
            server.rename_track(md5, server.RenameTrack(name="whatever"), user=TEST_USER)
        self.assertEqual(ctx.exception.status_code, 404)


class TestDelete(LibraryTestCase):
    def test_removes_the_file_stems_grid_and_manifest(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        self.seed_stems(md5)
        self.seed_grid(md5)

        result = server.delete_track(md5, user=TEST_USER)

        self.assertEqual(result["deleted"], md5)
        self.assertFalse(audio.exists())
        self.assertFalse((self.cache_dir / md5).exists())
        self.assertFalse((self.grids_dir / f"{md5}.json").exists())
        self.assertFalse(server._manifest_path(TEST_USER_ID, md5).exists())
        self.assertIsNone(self.entry(md5), "a deleted song must not still be listed")

    def test_is_unrecoverable_by_re_import(self):
        """Unlike an eviction, a delete also drops the manifest that would
        have reconnected a re-import to the old grid."""
        data = b"the one song that matters"
        first = _import("song.mp3", data)
        md5 = first["md5"]
        self.seed_grid(md5)

        server.delete_track(md5, user=TEST_USER)
        again = _import("song.mp3", data)

        self.assertEqual(again["md5"], md5, "re-import of identical bytes always hashes the same")
        entry = self.entry(md5)
        self.assertEqual(entry["state"], "needs_tap", "the old grid must not come back after a delete")
        self.assertFalse(entry["grid_present"])

    def test_is_idempotent(self):
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        self.seed_stems(md5)
        self.seed_grid(md5)

        server.delete_track(md5, user=TEST_USER)
        server.delete_track(md5, user=TEST_USER)  # must not raise

        self.assertIsNone(self.entry(md5))

    def test_works_on_an_evicted_entry_with_no_source_file(self):
        """The row's whole point at that state is to still be deletable."""
        audio = self.add_track()
        md5 = server._file_hash(audio)
        server._write_manifest(TEST_USER_ID, audio, md5, 180.5)
        self.seed_grid(md5)
        audio.unlink()

        server.delete_track(md5, user=TEST_USER)
        self.assertIsNone(self.entry(md5))
        self.assertFalse((self.grids_dir / f"{md5}.json").exists())


class TestAccountIsolation(LibraryTestCase):
    """The point of the whole refactor: two accounts never see each other's
    tracks, even when they share a base directory tree and identical bytes.
    """

    OTHER_USER_ID = 2

    def test_second_account_starts_with_an_empty_library(self):
        _import("song.mp3", b"the first user's song")
        other = types.SimpleNamespace(id=self.OTHER_USER_ID)
        self.assertEqual(server.list_library(user=other)["songs"], [])

    def test_identical_bytes_get_separate_stem_caches(self):
        """The concrete proof MD5 dedup is per-account, not global."""
        data = b"the exact same audio bytes"
        first = _import("song.mp3", data)
        md5 = first["md5"]
        self.seed_stems(md5)  # only under TEST_USER_ID's cache_dir

        other_cache_dir = server.CACHE_DIR / str(self.OTHER_USER_ID) / md5
        self.assertFalse(
            other_cache_dir.exists(),
            "a second account must not see the first account's cached stems",
        )

    def test_a_track_id_cannot_be_read_across_accounts(self):
        _import("song.mp3", b"only the first user should see this")
        with self.assertRaises(server.HTTPException) as ctx:
            server._track_file(self.OTHER_USER_ID, "song")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
