import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from beat_detection import detect_beats, eight_count_grouping

TEST_AUDIO_PATH = "test_files/TBH.mp3"


class TestDetectBeats(unittest.TestCase):
    def test_returns_tuple_of_arrays(self):
        beats, downbeats = detect_beats(TEST_AUDIO_PATH)
        self.assertIsInstance(beats, np.ndarray)
        self.assertIsInstance(downbeats, np.ndarray)

    def test_returns_sorted_beats(self):
        beats, _ = detect_beats(TEST_AUDIO_PATH)
        self.assertTrue(np.all(beats[1:] >= beats[:-1]))

    def test_returns_sorted_downbeats(self):
        _, downbeats = detect_beats(TEST_AUDIO_PATH)
        self.assertTrue(np.all(downbeats[1:] >= downbeats[:-1]))

    def test_returns_positive_timestamps(self):
        beats, downbeats = detect_beats(TEST_AUDIO_PATH)
        self.assertTrue(np.all(beats >= 0))
        self.assertTrue(np.all(downbeats >= 0))

    def test_returns_nonempty_on_real_audio(self):
        beats, downbeats = detect_beats(TEST_AUDIO_PATH)
        self.assertGreater(len(beats), 0)
        self.assertGreater(len(downbeats), 0)

    def test_downbeats_subset_of_beats(self):
        beats, downbeats = detect_beats(TEST_AUDIO_PATH)
        for db in downbeats:
            self.assertIn(db, beats)

    def test_non_string_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_beats(123)

    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_beats("")

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            detect_beats("does/not/exist.mp3")


class TestEightCountGrouping(unittest.TestCase):
    def test_groups_into_eight_counts(self):
        beats = np.arange(1.0, 17.0)  # 16 beats
        downbeats = np.array([1.0])
        groups = eight_count_grouping(beats, downbeats)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 8)
        self.assertEqual(len(groups[1]), 8)

    def test_partial_last_group(self):
        beats = np.arange(1.0, 11.0)  # 10 beats
        downbeats = np.array([1.0])
        groups = eight_count_grouping(beats, downbeats)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 8)
        self.assertEqual(len(groups[1]), 2)

    def test_no_subdivision_marks_all_as_beats(self):
        beats = np.array([1.0, 2.0, 3.0])
        downbeats = np.array([1.0])
        groups = eight_count_grouping(beats, downbeats)
        for group in groups:
            for timestamp, is_beat in group:
                self.assertTrue(is_beat)

    def test_aligns_to_first_downbeat(self):
        beats = np.array([0.5, 1.0, 2.0, 3.0])
        downbeats = np.array([1.0])
        groups = eight_count_grouping(beats, downbeats)
        all_tuples = [t for g in groups for t in g]
        # 0.5 is dropped because it precedes the first downbeat at 1.0
        self.assertEqual(all_tuples[0][0], 1.0)
        self.assertEqual(len(all_tuples), 3)

    def test_subdivision_creates_intermediate_points(self):
        beats = np.array([0.0, 1.0, 2.0])
        downbeats = np.array([0.0])
        groups = eight_count_grouping(beats, downbeats, subdivisions=2)
        # 3 beats with subdivisions=2: beat0, sub, beat1, sub, beat2 = 5 entries
        all_tuples = [t for g in groups for t in g]
        self.assertEqual(len(all_tuples), 5)
        self.assertAlmostEqual(all_tuples[1][0], 0.5)
        self.assertFalse(all_tuples[1][1])
        self.assertAlmostEqual(all_tuples[3][0], 1.5)
        self.assertFalse(all_tuples[3][1])

    def test_subdivision_groups_respect_eight_count(self):
        beats = np.arange(0.0, 9.0)  # 9 beats
        downbeats = np.array([0.0])
        groups = eight_count_grouping(beats, downbeats, subdivisions=2)
        # 8 intervals * 2 entries + final beat = 17 entries; first group has 16
        self.assertEqual(len(groups[0]), 16)

    def test_subdivisions_four_creates_three_intermediate_points(self):
        beats = np.array([0.0, 1.0])
        downbeats = np.array([0.0])
        groups = eight_count_grouping(beats, downbeats, subdivisions=4)
        all_tuples = [t for g in groups for t in g]
        # beat0, sub, sub, sub, beat1 = 5 entries
        self.assertEqual(len(all_tuples), 5)
        self.assertAlmostEqual(all_tuples[1][0], 0.25)
        self.assertAlmostEqual(all_tuples[2][0], 0.50)
        self.assertAlmostEqual(all_tuples[3][0], 0.75)
        self.assertFalse(all_tuples[1][1])
        self.assertFalse(all_tuples[2][1])
        self.assertFalse(all_tuples[3][1])

    def test_single_beat(self):
        beats = np.array([5.0])
        downbeats = np.array([5.0])
        groups = eight_count_grouping(beats, downbeats)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0], [(5.0, True)])

    def test_beats_not_ndarray_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping([1.0, 2.0], np.array([1.0]))

    def test_downbeats_not_ndarray_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(np.array([1.0, 2.0]), [1.0])

    def test_non_numeric_beats_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(np.array(["a", "b"]), np.array([1.0]))

    def test_non_numeric_downbeats_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array(["a"]))

    def test_empty_beats_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([]), np.array([1.0]))

    def test_empty_downbeats_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array([]))

    def test_first_downbeat_missing_from_beats_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([1.0, 2.0, 3.0]), np.array([99.0]))

    def test_non_integer_subdivisions_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array([1.0]), subdivisions=1.5)

    def test_bool_subdivisions_raises_type_error(self):
        with self.assertRaises(TypeError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array([1.0]), subdivisions=True)

    def test_zero_subdivisions_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array([1.0]), subdivisions=0)

    def test_negative_subdivisions_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array([1.0]), subdivisions=-1)

    def test_subdivisions_above_four_raises_value_error(self):
        with self.assertRaises(ValueError):
            eight_count_grouping(np.array([1.0, 2.0]), np.array([1.0]), subdivisions=5)


if __name__ == "__main__":
    unittest.main()
