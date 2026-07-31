import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import bookmark


class TestBookmarkPath(unittest.TestCase):
    def test_returns_sidecar_next_to_audio(self):
        result = bookmark._bookmark_path("/tmp/song.mp3")
        self.assertEqual(result, Path("/tmp/song.bookmarks.json"))

    def test_strips_extension(self):
        result = bookmark._bookmark_path("song.wav")
        self.assertEqual(result.name, "song.bookmarks.json")

    def test_non_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark._bookmark_path(123)

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark._bookmark_path("")


class TestLoadBookmarks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = str(Path(self.tmp.name) / "song.mp3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_empty_dict_when_no_file(self):
        self.assertEqual(bookmark.load_bookmarks(self.audio_path), {})

    def test_returns_existing_bookmarks(self):
        sidecar = Path(self.audio_path).with_name("song.bookmarks.json")
        data = {"abc": {"id": "abc", "timestamp": 1.0, "label": "x", "snap_mode": "none"}}
        sidecar.write_text(json.dumps(data))

        self.assertEqual(bookmark.load_bookmarks(self.audio_path), data)

    def test_non_string_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.load_bookmarks(None)

    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.load_bookmarks("")


class TestSaveBookmarks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = str(Path(self.tmp.name) / "song.mp3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_json_to_sidecar(self):
        data = {"abc": {"id": "abc", "timestamp": 2.5, "label": "drop", "snap_mode": "beat"}}
        bookmark.save_bookmarks(data, self.audio_path)

        sidecar = Path(self.audio_path).with_name("song.bookmarks.json")
        self.assertTrue(sidecar.exists())
        self.assertEqual(json.loads(sidecar.read_text()), data)

    def test_overwrites_existing_file(self):
        sidecar = Path(self.audio_path).with_name("song.bookmarks.json")
        sidecar.write_text(json.dumps({"old": {}}))

        bookmark.save_bookmarks({"new": {}}, self.audio_path)
        self.assertEqual(json.loads(sidecar.read_text()), {"new": {}})

    def test_returns_none(self):
        self.assertIsNone(bookmark.save_bookmarks({}, self.audio_path))

    def test_non_dict_bookmarks_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.save_bookmarks([], self.audio_path)

    def test_non_string_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.save_bookmarks({}, 123)

    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.save_bookmarks({}, "")


class TestSnapTimestamp(unittest.TestCase):
    def test_before_first_anchor_returns_t(self):
        self.assertEqual(bookmark.snap_timestamp(0.5, [1.0, 2.0, 3.0]), 0.5)

    def test_between_anchors_snaps_back(self):
        self.assertEqual(bookmark.snap_timestamp(2.4, [1.0, 2.0, 3.0]), 2.0)

    def test_exact_anchor_snaps_to_previous(self):
        # t == anchors[i] is not less-than, so the loop walks past it.
        self.assertEqual(bookmark.snap_timestamp(2.0, [1.0, 2.0, 3.0]), 2.0)

    def test_past_last_anchor_returns_last(self):
        self.assertEqual(bookmark.snap_timestamp(99.0, [1.0, 2.0, 3.0]), 3.0)

    def test_non_numeric_t_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.snap_timestamp("abc", [1.0, 2.0])

    def test_bool_t_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.snap_timestamp(True, [1.0, 2.0])

    def test_empty_sliced_beats_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.snap_timestamp(1.0, [])


class TestGetCountAnchors(unittest.TestCase):
    def test_returns_every_fourth_beat(self):
        beats = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.assertEqual(bookmark.get_count_anchors(beats, 4), [0.0, 4.0, 8.0])

    def test_returns_every_eighth_beat(self):
        beats = list(range(16))
        self.assertEqual(bookmark.get_count_anchors(beats, 8), [0, 8])

    def test_count_one_returns_all_beats(self):
        beats = [1.0, 2.0, 3.0]
        self.assertEqual(bookmark.get_count_anchors(beats, 1), beats)

    def test_non_int_count_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.get_count_anchors([1.0, 2.0], 2.5)

    def test_bool_count_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.get_count_anchors([1.0, 2.0], True)

    def test_zero_count_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.get_count_anchors([1.0, 2.0], 0)

    def test_negative_count_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.get_count_anchors([1.0, 2.0], -1)


class TestAddBookmark(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = str(Path(self.tmp.name) / "song.mp3")
        self.beats = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.bookmarks = {}

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_uuid_string(self):
        bid = bookmark.add_bookmark(1.5, self.beats, "none", self.audio_path, self.bookmarks)
        self.assertIsInstance(bid, str)
        self.assertEqual(len(bid), 36)  # standard UUID length

    def test_inserts_into_dict_with_record(self):
        bid = bookmark.add_bookmark(2.7, self.beats, "none", self.audio_path, self.bookmarks, label="hit")
        self.assertIn(bid, self.bookmarks)
        record = self.bookmarks[bid]
        self.assertEqual(record["id"], bid)
        self.assertEqual(record["timestamp"], 2.7)
        self.assertEqual(record["label"], "hit")
        self.assertEqual(record["snap_mode"], "none")

    def test_snap_none_keeps_exact_timestamp(self):
        bid = bookmark.add_bookmark(2.7, self.beats, "none", self.audio_path, self.bookmarks)
        self.assertEqual(self.bookmarks[bid]["timestamp"], 2.7)

    def test_snap_beat_snaps_to_nearest_prior_beat(self):
        bid = bookmark.add_bookmark(2.7, self.beats, "beat", self.audio_path, self.bookmarks)
        self.assertEqual(self.bookmarks[bid]["timestamp"], 2.0)

    def test_snap_4_count_snaps_to_nearest_prior_4_count(self):
        bid = bookmark.add_bookmark(5.5, self.beats, "4-count", self.audio_path, self.bookmarks)
        self.assertEqual(self.bookmarks[bid]["timestamp"], 4.0)

    def test_snap_8_count_snaps_to_nearest_prior_8_count(self):
        bid = bookmark.add_bookmark(7.9, self.beats, "8-count", self.audio_path, self.bookmarks)
        self.assertEqual(self.bookmarks[bid]["timestamp"], 0.0)

    def test_default_label_is_empty(self):
        bid = bookmark.add_bookmark(1.0, self.beats, "none", self.audio_path, self.bookmarks)
        self.assertEqual(self.bookmarks[bid]["label"], "")

    def test_persists_to_sidecar(self):
        bid = bookmark.add_bookmark(1.0, self.beats, "none", self.audio_path, self.bookmarks)
        sidecar = Path(self.audio_path).with_name("song.bookmarks.json")
        on_disk = json.loads(sidecar.read_text())
        self.assertIn(bid, on_disk)

    def test_invalid_snap_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.add_bookmark(1.0, self.beats, "16-count", self.audio_path, self.bookmarks)

    def test_non_numeric_t_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.add_bookmark("nope", self.beats, "none", self.audio_path, self.bookmarks)

    def test_non_string_audio_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.add_bookmark(1.0, self.beats, "none", 123, self.bookmarks)

    def test_empty_audio_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.add_bookmark(1.0, self.beats, "none", "", self.bookmarks)

    def test_non_dict_bookmarks_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.add_bookmark(1.0, self.beats, "none", self.audio_path, [])

    def test_non_string_label_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.add_bookmark(1.0, self.beats, "none", self.audio_path, self.bookmarks, label=123)


class TestDeleteBookmark(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = str(Path(self.tmp.name) / "song.mp3")
        self.bookmarks = {
            "id-1": {"id": "id-1", "timestamp": 1.0, "label": "a", "snap_mode": "none"},
            "id-2": {"id": "id-2", "timestamp": 2.0, "label": "b", "snap_mode": "none"},
        }
        bookmark.save_bookmarks(self.bookmarks, self.audio_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_removes_from_dict(self):
        bookmark.delete_bookmark("id-1", self.bookmarks, self.audio_path)
        self.assertNotIn("id-1", self.bookmarks)
        self.assertIn("id-2", self.bookmarks)

    def test_persists_deletion_to_sidecar(self):
        bookmark.delete_bookmark("id-1", self.bookmarks, self.audio_path)
        on_disk = json.loads(Path(self.audio_path).with_name("song.bookmarks.json").read_text())
        self.assertNotIn("id-1", on_disk)
        self.assertIn("id-2", on_disk)

    def test_missing_id_is_silent(self):
        bookmark.delete_bookmark("missing", self.bookmarks, self.audio_path)
        self.assertEqual(set(self.bookmarks.keys()), {"id-1", "id-2"})

    def test_non_string_id_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.delete_bookmark(123, self.bookmarks, self.audio_path)

    def test_non_dict_bookmarks_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.delete_bookmark("id-1", [], self.audio_path)

    def test_non_string_audio_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.delete_bookmark("id-1", self.bookmarks, 123)

    def test_empty_audio_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.delete_bookmark("id-1", self.bookmarks, "")


class TestUpdateBookmarkLabel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = str(Path(self.tmp.name) / "song.mp3")
        self.bookmarks = {
            "id-1": {"id": "id-1", "timestamp": 1.0, "label": "old", "snap_mode": "none"},
        }
        bookmark.save_bookmarks(self.bookmarks, self.audio_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_updates_label_in_dict(self):
        bookmark.update_bookmark_label("id-1", "new", self.bookmarks, self.audio_path)
        self.assertEqual(self.bookmarks["id-1"]["label"], "new")

    def test_persists_update_to_sidecar(self):
        bookmark.update_bookmark_label("id-1", "new", self.bookmarks, self.audio_path)
        on_disk = json.loads(Path(self.audio_path).with_name("song.bookmarks.json").read_text())
        self.assertEqual(on_disk["id-1"]["label"], "new")

    def test_same_label_does_not_rewrite_file(self):
        sidecar = Path(self.audio_path).with_name("song.bookmarks.json")
        mtime_before = sidecar.stat().st_mtime_ns
        bookmark.update_bookmark_label("id-1", "old", self.bookmarks, self.audio_path)
        self.assertEqual(sidecar.stat().st_mtime_ns, mtime_before)

    def test_missing_id_raises_key_error(self):
        with self.assertRaises(KeyError):
            bookmark.update_bookmark_label("nope", "x", self.bookmarks, self.audio_path)

    def test_non_string_id_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.update_bookmark_label(123, "x", self.bookmarks, self.audio_path)

    def test_non_string_label_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.update_bookmark_label("id-1", 123, self.bookmarks, self.audio_path)

    def test_non_dict_bookmarks_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.update_bookmark_label("id-1", "x", [], self.audio_path)

    def test_non_string_audio_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.update_bookmark_label("id-1", "x", self.bookmarks, 123)

    def test_empty_audio_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            bookmark.update_bookmark_label("id-1", "x", self.bookmarks, "")


class TestListBookmarks(unittest.TestCase):
    def test_sorts_by_timestamp_ascending(self):
        bookmarks = {
            "b": {"id": "b", "timestamp": 5.0, "label": "", "snap_mode": "none"},
            "a": {"id": "a", "timestamp": 1.0, "label": "", "snap_mode": "none"},
            "c": {"id": "c", "timestamp": 3.0, "label": "", "snap_mode": "none"},
        }
        result = bookmark.list_bookmarks(bookmarks)
        timestamps = [record["timestamp"] for _, record in result]
        self.assertEqual(timestamps, [1.0, 3.0, 5.0])

    def test_returns_list_of_tuples(self):
        bookmarks = {"a": {"id": "a", "timestamp": 1.0, "label": "", "snap_mode": "none"}}
        result = bookmark.list_bookmarks(bookmarks)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(result[0][0], "a")

    def test_empty_dict_returns_empty_list(self):
        self.assertEqual(bookmark.list_bookmarks({}), [])

    def test_non_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            bookmark.list_bookmarks([])


if __name__ == "__main__":
    unittest.main()
