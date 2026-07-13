import unittest
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from onset_detection import detect_onsets, ONSET_PARAMS

TEST_AUDIO_PATH = "test_files/TBH.mp3"


class TestDetectOnsetsDrums(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = detect_onsets(TEST_AUDIO_PATH, "drums")
        cls.onsets = cls.result["t"]
        cls.strengths = cls.result["strength"]

    def test_returns_ndarray(self):
        self.assertIsInstance(self.onsets, np.ndarray)

    def test_returns_one_dimensional(self):
        self.assertEqual(self.onsets.ndim, 1)

    def test_returns_nonempty_on_real_audio(self):
        self.assertGreater(len(self.onsets), 0)

    def test_returns_sorted_onsets(self):
        self.assertTrue(np.all(self.onsets[1:] >= self.onsets[:-1]))

    def test_returns_nonnegative_timestamps(self):
        self.assertTrue(np.all(self.onsets >= 0))

    def test_returns_seconds_not_frames(self):
        # Timestamps are in seconds. For a typical pop song the last onset
        # should be well under one hour but at least one second in.
        self.assertGreater(self.onsets[-1], 1.0)
        self.assertLess(self.onsets[-1], 3600.0)

    def test_strengths_parallel_to_timestamps(self):
        self.assertIsInstance(self.strengths, np.ndarray)
        self.assertEqual(self.strengths.shape, self.onsets.shape)

    def test_strengths_finite_and_nonnegative(self):
        self.assertTrue(np.all(np.isfinite(self.strengths)))
        self.assertTrue(np.all(self.strengths >= 0))
        # Detected peaks sit above the envelope's moving average by delta,
        # so at least some strengths must be strictly positive.
        self.assertGreater(self.strengths.max(), 0)


class TestDetectOnsetsBass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = detect_onsets(TEST_AUDIO_PATH, "bass")
        cls.onsets = cls.result["t"]
        cls.strengths = cls.result["strength"]

    def test_returns_ndarray(self):
        self.assertIsInstance(self.onsets, np.ndarray)

    def test_returns_one_dimensional(self):
        self.assertEqual(self.onsets.ndim, 1)

    def test_returns_nonempty_on_real_audio(self):
        self.assertGreater(len(self.onsets), 0)

    def test_returns_sorted_onsets(self):
        self.assertTrue(np.all(self.onsets[1:] >= self.onsets[:-1]))

    def test_returns_nonnegative_timestamps(self):
        self.assertTrue(np.all(self.onsets >= 0))

    def test_strengths_parallel_to_timestamps(self):
        self.assertIsInstance(self.strengths, np.ndarray)
        self.assertEqual(self.strengths.shape, self.onsets.shape)

    def test_strengths_finite_and_nonnegative(self):
        self.assertTrue(np.all(np.isfinite(self.strengths)))
        self.assertTrue(np.all(self.strengths >= 0))
        self.assertGreater(self.strengths.max(), 0)


class TestDetectOnsetsValidation(unittest.TestCase):
    def test_non_string_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_onsets(123, "drums")

    def test_none_path_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_onsets(None, "drums")

    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_onsets("", "drums")

    def test_non_string_stem_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_onsets(TEST_AUDIO_PATH, 123)

    def test_none_stem_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            detect_onsets(TEST_AUDIO_PATH, None)

    def test_unknown_stem_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_onsets(TEST_AUDIO_PATH, "vocals")

    def test_empty_stem_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_onsets(TEST_AUDIO_PATH, "")

    def test_wrong_case_stem_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_onsets(TEST_AUDIO_PATH, "Drums")

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            detect_onsets("does/not/exist.wav", "drums")

    def test_validation_runs_before_load(self):
        # An empty path should fail before librosa is asked to load it,
        # so the error is ValueError, not whatever librosa raises.
        with self.assertRaises(ValueError):
            detect_onsets("", "drums")


class TestOnsetParams(unittest.TestCase):
    def test_drums_present(self):
        self.assertIn("drums", ONSET_PARAMS)

    def test_bass_present(self):
        self.assertIn("bass", ONSET_PARAMS)

    def test_drums_has_expected_keys(self):
        expected = {
            "hop_length", "backtrack", "delta",
            "pre_max", "post_max", "pre_avg", "post_avg", "wait",
        }
        self.assertEqual(set(ONSET_PARAMS["drums"].keys()), expected)

    def test_bass_has_expected_keys(self):
        expected = {
            "hop_length", "backtrack", "delta",
            "pre_max", "post_max", "pre_avg", "post_avg", "wait",
        }
        self.assertEqual(set(ONSET_PARAMS["bass"].keys()), expected)

    def test_bass_delta_higher_than_drums(self):
        # Bass uses a higher threshold because the custom mel-envelope
        # produces stronger peaks; if this inverts, onset behavior changes.
        self.assertGreater(
            ONSET_PARAMS["bass"]["delta"],
            ONSET_PARAMS["drums"]["delta"],
        )


if __name__ == "__main__":
    unittest.main()
