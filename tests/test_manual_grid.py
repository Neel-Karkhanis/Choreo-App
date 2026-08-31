"""Manual (tapped) grids are user data and must outlive the cache.

The invariant under test: `rm -rf cache/` is a SAFE operation. Everything under
cache/ is derived and regenerates from the audio; a tapped grid is a person
counting a song out by ear and cannot be recomputed by any means. These tests
exist because the two used to live in the same directory.

The endpoint functions are called directly rather than over HTTP — httpx (and
so starlette's TestClient) is not installed, and the routes are plain functions
anyway, so this exercises the real read/write path with no HTTP layer in the way.

Routes take an `owner_id` (as FastAPI's Depends(identity.get_owner_id) would
supply in real traffic — a UUIDv4 string identifying an anonymous device, not
an account); the plain helpers take the same raw owner_id. TEST_OWNER_ID below
is one fixed fake device — see test_library.py's module docstring for why
setUp nests the fixture directories one level deeper under it rather than
threading an owner id through every fixture helper.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "api"))

import server

BEATS = [round(0.5 + i * 0.6, 3) for i in range(64)]
DOWNBEATS = list(range(0, 64, 4))
EIGHT_COUNTS = list(range(0, 64, 8))

TEST_OWNER_ID = "11111111-1111-4111-8111-111111111111"


def _grid(beats=None):
    return server.ManualGrid(
        beats=beats or BEATS,
        downbeats=DOWNBEATS,
        eight_counts=EIGHT_COUNTS,
    )


class ManualGridTestCase(unittest.TestCase):
    """Redirects both trees into a temp dir so no test can touch real grids."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cache_root = root / "cache"
        base_cache_dir = self.cache_root / "stems"
        base_grids_dir = root / "data" / "grids"
        base_tracks_dir = root / "tracks"

        self.cache_dir = base_cache_dir / TEST_OWNER_ID
        self.grids_dir = base_grids_dir / TEST_OWNER_ID
        self.tracks_dir = base_tracks_dir / TEST_OWNER_ID
        self.tracks_dir.mkdir(parents=True)

        # A real file, so _file_hash computes a real MD5 off real bytes.
        self.audio = self.tracks_dir / "song.mp3"
        self.audio.write_bytes(b"not really an mp3, but it hashes just fine")
        self.track_id = "song"

        self._patches = {
            "CACHE_DIR": server.CACHE_DIR,
            "GRIDS_DIR": server.GRIDS_DIR,
            "TRACKS_DIR": server.TRACKS_DIR,
        }
        server.CACHE_DIR = base_cache_dir
        server.GRIDS_DIR = base_grids_dir
        server.TRACKS_DIR = base_tracks_dir

    def tearDown(self):
        for name, value in self._patches.items():
            setattr(server, name, value)
        self.tmp.cleanup()

    @property
    def md5(self):
        return server._file_hash(self.audio)

    def seed_cache(self):
        """Fill the cache tree the way a real analysis would."""
        d = self.cache_dir / self.md5
        d.mkdir(parents=True, exist_ok=True)
        (d / "analysis.json").write_text(json.dumps({"schema_version": 4, "beats": [1.0]}))
        for stem in server.STEM_NAMES:
            (d / f"{stem}.wav").write_bytes(b"RIFF....derived audio")
        return d


class TestGridsLiveOutsideTheCache(ManualGridTestCase):
    def test_grid_path_is_not_under_the_cache_tree(self):
        path = server._manual_grid_path(TEST_OWNER_ID, self.audio)
        self.assertFalse(
            self.cache_root in path.parents,
            f"tapped grid must not live under the cache tree: {path}",
        )

    def test_grid_path_is_data_grids_keyed_by_md5(self):
        path = server._manual_grid_path(TEST_OWNER_ID, self.audio)
        self.assertEqual(path, self.grids_dir / f"{self.md5}.json")

    def test_key_is_the_audio_md5_not_the_filename(self):
        """A rename must not orphan a grid, and must not adopt another's."""
        before = server._manual_grid_path(TEST_OWNER_ID, self.audio)
        renamed = self.tracks_dir / "renamed.mp3"
        self.audio.rename(renamed)
        self.assertEqual(server._manual_grid_path(TEST_OWNER_ID, renamed), before)

        different = self.tracks_dir / "other.mp3"
        different.write_bytes(b"entirely different audio bytes")
        self.assertNotEqual(server._manual_grid_path(TEST_OWNER_ID, different), before)


class TestCacheWipeIsSafe(ManualGridTestCase):
    """The whole point of this change."""

    def test_manual_grid_survives_rm_rf_cache(self):
        self.seed_cache()
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        self.assertEqual(
            server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]["beats"], BEATS
        )

        # rm -rf cache/ — the entire tree, not just the stems subdir.
        shutil.rmtree(self.cache_root)
        self.assertFalse(self.cache_root.exists())

        # "Reload": the grid is read back from disk, and drives the same beats.
        after = server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertIsNotNone(after, "tapped grid was destroyed by a cache wipe")
        self.assertEqual(after["beats"], BEATS)
        self.assertEqual(after["downbeats"], DOWNBEATS)
        self.assertEqual(after["eight_counts"], EIGHT_COUNTS)

    def test_cache_wipe_removes_the_derived_data_it_should(self):
        """Guards the test above from passing vacuously against an empty cache."""
        cached = self.seed_cache()
        self.assertTrue((cached / "analysis.json").exists())
        self.assertTrue((cached / "drums.wav").exists())
        shutil.rmtree(self.cache_root)
        self.assertFalse((cached / "analysis.json").exists())
        self.assertFalse((cached / "drums.wav").exists())

    def test_writing_a_grid_does_not_need_the_cache_to_exist(self):
        """Post-wipe, before re-analysis, a tapped grid must still be writable."""
        self.assertFalse(self.cache_dir.exists())
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        self.assertEqual(
            server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]["beats"], BEATS
        )


class TestMigration(ManualGridTestCase):
    def legacy_write(self, beats):
        d = self.cache_dir / self.md5
        d.mkdir(parents=True, exist_ok=True)
        path = d / "manual_grid.json"
        path.write_text(
            json.dumps({"beats": beats, "downbeats": DOWNBEATS, "eight_counts": EIGHT_COUNTS})
        )
        return path

    def test_moves_legacy_grid_out_of_the_cache(self):
        legacy = self.legacy_write(BEATS)
        self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 1)
        self.assertFalse(legacy.exists(), "legacy copy left behind in the cache")
        self.assertEqual(
            server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]["beats"], BEATS
        )

    def test_migrated_grid_then_survives_a_cache_wipe(self):
        """The migration is only worth anything if it lands somewhere safe."""
        self.legacy_write(BEATS)
        server._migrate_manual_grids(TEST_OWNER_ID)
        shutil.rmtree(self.cache_root)
        self.assertEqual(
            server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]["beats"], BEATS
        )

    def test_is_idempotent(self):
        self.legacy_write(BEATS)
        self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 1)
        for _ in range(3):
            self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 0)
        self.assertEqual(
            server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]["beats"], BEATS
        )

    def test_no_op_on_a_clean_tree(self):
        self.seed_cache()
        self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 0)

    def test_no_op_when_there_is_no_cache_at_all(self):
        self.assertFalse(self.cache_dir.exists())
        self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 0)

    def test_already_migrated_grid_wins_over_a_stale_cache_copy(self):
        """data/ is the only tree the app writes to, so a cache copy is stale."""
        current = [round(b + 10, 3) for b in BEATS]
        server.put_manual_grid(self.track_id, _grid(current), owner_id=TEST_OWNER_ID)
        legacy = self.legacy_write(BEATS)

        self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 0)
        self.assertFalse(legacy.exists(), "stale cache copy should be dropped")
        self.assertEqual(
            server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]["beats"],
            current,
            "migration clobbered the live grid with a stale cache copy",
        )

    def test_migrates_every_track(self):
        second = self.tracks_dir / "second.mp3"
        second.write_bytes(b"a second track with different bytes")
        for audio in (self.audio, second):
            d = self.cache_dir / server._file_hash(audio)
            d.mkdir(parents=True, exist_ok=True)
            (d / "manual_grid.json").write_text(
                json.dumps({"beats": BEATS, "downbeats": DOWNBEATS, "eight_counts": EIGHT_COUNTS})
            )
        self.assertEqual(server._migrate_manual_grids(TEST_OWNER_ID), 2)
        self.assertEqual(len(list(self.grids_dir.glob("*.json"))), 2)
        self.assertEqual(len(list(self.cache_dir.glob("*/manual_grid.json"))), 0)


class TestMigrationDoesNotLogTheDeviceId(ManualGridTestCase):
    """_migrate_manual_grids prints a summary line whenever it moves
    something — the raw device id must never be part of it (see
    identity.owner_key's own docstring on why). This is the regression
    test for that: it captures the actual stdout the function writes and
    asserts on it directly, rather than trusting the source doesn't
    regress silently. A standalone case, not a TestMigration subclass —
    subclassing a TestCase re-runs every inherited test_* method too,
    which is not what duplicating one small fixture helper is worth here.
    """

    def legacy_write(self, beats):
        d = self.cache_dir / self.md5
        d.mkdir(parents=True, exist_ok=True)
        path = d / "manual_grid.json"
        path.write_text(
            json.dumps({"beats": beats, "downbeats": DOWNBEATS, "eight_counts": EIGHT_COUNTS})
        )
        return path

    def test_log_line_contains_the_hashed_key_not_the_raw_owner_id(self):
        import contextlib
        import io

        import identity

        self.legacy_write(BEATS)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            server._migrate_manual_grids(TEST_OWNER_ID)
        output = captured.getvalue()

        self.assertNotIn(TEST_OWNER_ID, output)
        self.assertIn(identity.owner_key(TEST_OWNER_ID), output)


class TestGridSurvivesARestart(ManualGridTestCase):
    """With auto detection gone, the tapped grid is the ONLY grid.

    A persistence failure is no longer a degraded state — it makes the song
    unusable and re-prompts a human to count it out again. Tap once, restart,
    open fully gridded: the frontend shows its tap prompt exactly when this
    read returns null, so a grid that comes back here is a song that never
    prompts again.
    """

    def test_tapped_grid_round_trips_across_a_restart(self):
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)

        # A restart drops all process state. The memoized hash cache is the
        # only in-memory state on this path; clearing it forces the read below
        # to run cold, entirely from disk — exactly like a fresh boot.
        server._hash_cache.clear()

        stored = server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertIsNotNone(stored, "restart lost the tapped grid")
        self.assertEqual(stored["beats"], BEATS)
        self.assertEqual(stored["downbeats"], DOWNBEATS)
        self.assertEqual(stored["eight_counts"], EIGHT_COUNTS)


class TestRevertToAuto(ManualGridTestCase):
    def test_delete_removes_the_grid(self):
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        server.delete_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)
        self.assertIsNone(server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"])

    def test_delete_is_idempotent(self):
        server.delete_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)
        self.assertIsNone(server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"])

    def test_delete_leaves_the_cache_alone(self):
        cached = self.seed_cache()
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        server.delete_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)
        self.assertTrue((cached / "analysis.json").exists())
        self.assertTrue((cached / "drums.wav").exists())


class TestGridActiveToggle(ManualGridTestCase):
    """The reversible path: switching to auto must not require a retap."""

    def test_put_saves_as_active(self):
        result = server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        self.assertTrue(result["manual_grid"]["active"])

    def test_deactivating_keeps_the_taps_on_disk(self):
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        server.set_manual_grid_active(
            self.track_id, server.ManualGridActive(active=False), owner_id=TEST_OWNER_ID
        )

        stored = server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertFalse(stored["active"])
        self.assertEqual(stored["beats"], BEATS)
        self.assertEqual(stored["downbeats"], DOWNBEATS)
        self.assertEqual(stored["eight_counts"], EIGHT_COUNTS)

    def test_reactivating_restores_the_same_grid_without_retapping(self):
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        server.set_manual_grid_active(
            self.track_id, server.ManualGridActive(active=False), owner_id=TEST_OWNER_ID
        )
        server.set_manual_grid_active(
            self.track_id, server.ManualGridActive(active=True), owner_id=TEST_OWNER_ID
        )

        stored = server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertTrue(stored["active"])
        self.assertEqual(stored["beats"], BEATS)

    def test_re_tapping_replaces_the_saved_grid_and_reactivates_it(self):
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)
        server.set_manual_grid_active(
            self.track_id, server.ManualGridActive(active=False), owner_id=TEST_OWNER_ID
        )

        newer = [round(b + 5, 3) for b in BEATS]
        server.put_manual_grid(self.track_id, _grid(newer), owner_id=TEST_OWNER_ID)

        stored = server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertTrue(stored["active"])
        self.assertEqual(stored["beats"], newer)

    def test_setting_active_with_no_saved_grid_is_a_404(self):
        with self.assertRaises(server.HTTPException) as ctx:
            server.set_manual_grid_active(
                self.track_id, server.ManualGridActive(active=True), owner_id=TEST_OWNER_ID
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_grids_saved_before_active_existed_default_to_active(self):
        path = server._manual_grid_path(TEST_OWNER_ID, self.audio)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"beats": BEATS, "downbeats": DOWNBEATS, "eight_counts": EIGHT_COUNTS})
        )
        stored = server.get_manual_grid(self.track_id, owner_id=TEST_OWNER_ID)["manual_grid"]
        self.assertTrue(stored["active"])


class TestGridDeviceIsolation(ManualGridTestCase):
    """A tapped grid is exactly the irreplaceable data the whole per-device
    filesystem split exists to protect — worth its own direct check here,
    not just at the library level (test_library.py's TestDeviceIsolation).
    """

    OTHER_OWNER_ID = "22222222-2222-4222-8222-222222222222"

    def test_a_grid_is_invisible_to_a_different_device(self):
        server.put_manual_grid(self.track_id, _grid(), owner_id=TEST_OWNER_ID)

        other_dir = server.TRACKS_DIR / self.OTHER_OWNER_ID
        other_dir.mkdir(parents=True, exist_ok=True)
        (other_dir / "song.mp3").write_bytes(self.audio.read_bytes())

        self.assertIsNone(
            server.get_manual_grid(self.track_id, owner_id=self.OTHER_OWNER_ID)["manual_grid"]
        )


if __name__ == "__main__":
    unittest.main()
