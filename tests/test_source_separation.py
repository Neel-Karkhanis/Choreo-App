import unittest
import os
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import source_separation
from source_separation import (
    separate, get_stem, get_vocals, get_drums,
    get_bass, get_other, get_instrumental,
    _hash_file, _is_cached, EXPECTED_STEMS,
)

TEST_AUDIO = "test_files/TBH.mp3"

# ISOLATION: every cache write in this suite lands in a throwaway temp dir,
# never in the real cache/stems/ tree. That tree is LIVE APP DATA — the API
# server serves stems out of it, keyed by content hash, so a fixture whose
# bytes match a real track would otherwise collide with (and this suite's
# unlink/rmtree calls would destroy) data the running app depends on.
_cache_tmp = None
_original_cache_root = None


def setUpModule():
    global _cache_tmp, _original_cache_root
    _cache_tmp = tempfile.TemporaryDirectory()
    _original_cache_root = source_separation.CACHE_ROOT
    source_separation.CACHE_ROOT = Path(_cache_tmp.name)


def tearDownModule():
    source_separation.CACHE_ROOT = _original_cache_root
    _cache_tmp.cleanup()


class TestSeparate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stems = separate(TEST_AUDIO)
        cls.file_hash = _hash_file(TEST_AUDIO)
        cls.cache_dir = source_separation.CACHE_ROOT / cls.file_hash

    def test_returns_dict(self):
        self.assertIsInstance(self.stems, dict)

    def test_contains_all_stems(self):
        for stem in ["vocals", "drums", "bass", "other"]:
            self.assertIn(stem, self.stems)

    def test_stem_files_exist(self):
        for path in self.stems.values():
            self.assertTrue(os.path.exists(path))

    def test_stem_files_are_wav(self):
        for path in self.stems.values():
            self.assertTrue(path.endswith(".wav"))

    def test_stems_written_under_cache_dir(self):
        for path in self.stems.values():
            self.assertEqual(Path(path).parent, self.cache_dir)

    def test_intermediate_demucs_dir_cleaned_up(self):
        self.assertFalse((self.cache_dir / "htdemucs").exists())

    def test_type_error_on_non_string(self):
        with self.assertRaises(TypeError):
            separate(123)

    def test_value_error_on_empty_string(self):
        with self.assertRaises(ValueError):
            separate("")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            separate("nonexistent.mp3")


class TestSeparateCaching(unittest.TestCase):
    """Verifies separate() reuses cached stems instead of re-running Demucs."""

    @classmethod
    def setUpClass(cls):
        # Prime the cache so the cache-hit path is exercised below.
        separate(TEST_AUDIO)
        cls.file_hash = _hash_file(TEST_AUDIO)
        cls.cache_dir = source_separation.CACHE_ROOT / cls.file_hash

    def test_cache_hit_skips_demucs(self):
        with patch("source_separation.demucs.separate.main") as mock_main:
            stems = separate(TEST_AUDIO)
            mock_main.assert_not_called()

        for stem in ["vocals", "drums", "bass", "other"]:
            self.assertEqual(
                Path(stems[stem]),
                self.cache_dir / f"{stem}.wav",
            )

    def test_cache_miss_runs_demucs(self):
        missing_hash = "0" * 32
        missing_dir = source_separation.CACHE_ROOT / missing_hash
        if missing_dir.exists():
            shutil.rmtree(missing_dir)

        with patch("source_separation._hash_file", return_value=missing_hash), \
             patch("source_separation.demucs.separate.main") as mock_main:
            # Demucs is mocked, so separate() will fail when it looks for
            # the stems Demucs would have produced. We only care that it
            # tried to invoke Demucs at all.
            with self.assertRaises(FileNotFoundError):
                separate(TEST_AUDIO)
            mock_main.assert_called_once()

        if missing_dir.exists():
            shutil.rmtree(missing_dir)


class TestHashFile(unittest.TestCase):
    def test_returns_hex_string(self):
        digest = _hash_file(TEST_AUDIO)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 32)
        int(digest, 16)  # raises if not hex

    def test_deterministic(self):
        self.assertEqual(_hash_file(TEST_AUDIO), _hash_file(TEST_AUDIO))

    def test_differs_for_different_content(self):
        a = source_separation.CACHE_ROOT / "_hash_a.tmp"
        b = source_separation.CACHE_ROOT / "_hash_b.tmp"
        a.parent.mkdir(parents=True, exist_ok=True)
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        try:
            self.assertNotEqual(_hash_file(str(a)), _hash_file(str(b)))
        finally:
            a.unlink()
            b.unlink()


class TestIsCached(unittest.TestCase):
    def setUp(self):
        self.fake_hash = "deadbeef" * 4
        self.cache_dir = source_separation.CACHE_ROOT / self.fake_hash
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def tearDown(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def test_returns_false_when_dir_missing(self):
        self.assertFalse(_is_cached(self.fake_hash))

    def test_returns_false_when_partial(self):
        self.cache_dir.mkdir(parents=True)
        # Only write some of the expected stems.
        for stem in EXPECTED_STEMS[:-1]:
            (self.cache_dir / stem).write_bytes(b"")
        self.assertFalse(_is_cached(self.fake_hash))

    def test_returns_true_when_complete(self):
        self.cache_dir.mkdir(parents=True)
        for stem in EXPECTED_STEMS:
            (self.cache_dir / stem).write_bytes(b"")
        self.assertTrue(_is_cached(self.fake_hash))


class TestGetStem(unittest.TestCase):
    def setUp(self):
        self.stems = {
            "vocals": "cache/stems/abc/vocals.wav",
            "drums": "cache/stems/abc/drums.wav",
            "bass": "cache/stems/abc/bass.wav",
            "other": "cache/stems/abc/other.wav",
        }

    def test_returns_correct_path(self):
        self.assertEqual(get_stem(self.stems, "drums"), self.stems["drums"])

    def test_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_stem(self.stems, "snare")


class TestConvenienceFunctions(unittest.TestCase):
    def setUp(self):
        self.stems = {
            "vocals": "cache/stems/abc/vocals.wav",
            "drums": "cache/stems/abc/drums.wav",
            "bass": "cache/stems/abc/bass.wav",
            "other": "cache/stems/abc/other.wav",
        }

    def test_get_vocals(self):
        self.assertEqual(get_vocals(self.stems), self.stems["vocals"])

    def test_get_drums(self):
        self.assertEqual(get_drums(self.stems), self.stems["drums"])

    def test_get_bass(self):
        self.assertEqual(get_bass(self.stems), self.stems["bass"])

    def test_get_other(self):
        self.assertEqual(get_other(self.stems), self.stems["other"])


class TestGetInstrumental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stems = separate(TEST_AUDIO)
        cls.file_hash = _hash_file(TEST_AUDIO)
        cls.cache_dir = source_separation.CACHE_ROOT / cls.file_hash
        cls.expected_path = cls.cache_dir / "instrumental.wav"

    def setUp(self):
        if self.expected_path.exists():
            self.expected_path.unlink()

    def test_creates_file_in_cache_dir(self):
        path = get_instrumental(self.stems)
        self.assertEqual(Path(path), self.expected_path)
        self.assertTrue(os.path.exists(path))

    def test_file_is_nonempty(self):
        path = get_instrumental(self.stems)
        self.assertGreater(os.path.getsize(path), 0)

    def test_reuses_existing_instrumental(self):
        first = get_instrumental(self.stems)
        mtime_before = os.path.getmtime(first)

        with patch("source_separation.AudioSegment.from_wav") as mock_from_wav:
            second = get_instrumental(self.stems)
            mock_from_wav.assert_not_called()

        self.assertEqual(first, second)
        self.assertEqual(os.path.getmtime(second), mtime_before)


if __name__ == "__main__":
    unittest.main()
