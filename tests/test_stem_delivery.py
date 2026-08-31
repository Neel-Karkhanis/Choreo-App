"""Stems are delivered as Opus, not the WAV Demucs actually produces.

100-180MB per song across the four WAV stems (see server.py's
STEM_OPUS_BITRATE comment) is far too large to hand a client or cache
on-device, so get_stem lazily encodes a compressed delivery copy the first
time a stem is requested and reuses it after that — these tests are about
that lazy-cache-once contract, not about audio quality.

Exercises the real `ffmpeg` binary (present in dev and in the Docker image
— see the Dockerfile's own comment on why it's installed in both stages),
against a tiny synthetic WAV built with the stdlib `wave` module rather
than a fixture file, so the test carries no binary asset.
"""

import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "api"))

import server


def _write_silent_wav(path: Path, seconds: float = 0.1) -> None:
    frame_rate = 8000
    n_frames = int(frame_rate * seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class TestEnsureStemOpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav_path = self.dir / "drums.wav"
        self.opus_path = self.dir / "drums.opus"
        _write_silent_wav(self.wav_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_encodes_wav_to_opus(self):
        server._ensure_stem_opus(self.wav_path, self.opus_path)
        self.assertTrue(self.opus_path.exists())
        self.assertGreater(self.opus_path.stat().st_size, 0)

    def test_skips_encoding_when_already_cached(self):
        """Once per song, not once per request: a pre-existing opus file is
        left untouched rather than re-encoded.
        """
        sentinel = b"not actually opus, but that's the point"
        self.opus_path.write_bytes(sentinel)

        server._ensure_stem_opus(self.wav_path, self.opus_path)

        self.assertEqual(self.opus_path.read_bytes(), sentinel)

    def test_raises_and_cleans_up_on_encode_failure(self):
        garbage_wav = self.dir / "not_really_audio.wav"
        garbage_wav.write_bytes(b"this is not a wav file at all")

        with self.assertRaises(server.HTTPException) as ctx:
            server._ensure_stem_opus(garbage_wav, self.opus_path)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertFalse(
            self.opus_path.exists(), "a failed encode must not leave a partial file behind"
        )


if __name__ == "__main__":
    unittest.main()
