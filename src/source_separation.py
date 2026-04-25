import os
import demucs.separate
from pydub import AudioSegment
import hashlib
from pathlib import Path
import shutil

EXPECTED_STEMS = ("drums.wav", "bass.wav", "vocals.wav", "other.wav")

def _hash_file(path):
    """Compute the MD5 hash of a file's contents.

    Used as a stable cache key so the same audio file maps to the same
    cache directory across runs, regardless of its filename.

    Args:
        path: Path to the file to hash.

    Returns:
        The hex-encoded MD5 digest of the file's contents.
    """
    hash_object = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_object.update(chunk)

    return hash_object.hexdigest()

def _is_cached(file_hash):
    """Check whether a complete set of cached stems exists for a file hash.

    A cache entry is only considered valid if every stem in EXPECTED_STEMS
    is present, so a partial/interrupted run does not falsely signal a hit.

    Args:
        file_hash: The MD5 hash returned by _hash_file().

    Returns:
        True if cache/stems/<file_hash>/ contains all expected stems, else False.
    """
    cache_dir = Path("cache/stems") / file_hash
    if not cache_dir.exists():
        return False

    for stem in EXPECTED_STEMS:
        if not (cache_dir / stem).exists():
            return False

    return True

def separate(audio_path):
    """Run Demucs 4-stem separation on an audio file, with caching.

    The audio file is hashed and stems are written to cache/stems/<hash>/.
    If a complete cached set already exists for this file, Demucs is skipped
    and the cached paths are returned directly.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A dictionary mapping stem names ("vocals", "drums", "bass", "other")
        to their .wav file paths inside the per-file cache directory.

    Raises:
        TypeError: If audio_path is not a string.
        ValueError: If audio_path is empty.
        FileNotFoundError: If audio_path does not exist, or if Demucs
            failed to produce one of the expected stems.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    file_hash = _hash_file(audio_path)
    cache_dir = Path("cache/stems") / file_hash 

    if _is_cached(file_hash):
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            stems[stem] = str(cache_dir / f"{stem}.wav")

        return stems
    
    cache_dir.mkdir(parents=True, exist_ok=True)

    demucs.separate.main([
        "-n", "htdemucs",
        "-o", str(cache_dir),
        audio_path
    ])

    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(cache_dir, "htdemucs", track_name)

    stems = {}
    for stem in ["vocals", "drums", "bass", "other"]:
        path = os.path.join(stem_dir, f"{stem}.wav")
        if os.path.exists(path):
            new_path = cache_dir / f"{stem}.wav"
            shutil.move(path, new_path)
            stems[stem] = str(new_path)
        else:
            raise FileNotFoundError(f"Expected stem '{stem}' not found at {path}")

    (cache_dir / "htdemucs" / track_name).rmdir()
    (cache_dir / "htdemucs").rmdir()

    return stems

def get_stem(stems, name):
    """Get the path to a specific stem.

    Args:
        stems: Dictionary returned by separate().
        name: Stem name (vocals, drums, bass, other).

    Returns:
        The file path to the requested stem.

    Raises:
        KeyError: If the stem name is not found.
    """
    if name not in stems:
        raise KeyError(f"Stem '{name}' not found. Available: {list(stems.keys())}")
    return stems[name]

def get_vocals(stems):
    return get_stem(stems, "vocals")

def get_drums(stems):
    return get_stem(stems, "drums")

def get_bass(stems):
    return get_stem(stems, "bass")

def get_other(stems):
    return get_stem(stems, "other")

def get_instrumental(stems):
    """Combine drums, bass, and other into an instrumental track.

    The output is written to instrumental.wav alongside the cached stems.
    If that file already exists, the existing path is returned without
    re-running the overlay.

    Args:
        stems: Dictionary returned by separate().

    Returns:
        The path to instrumental.wav inside the per-file cache directory.
    """
    cache_dir = Path(stems["drums"]).parent
    output_path = cache_dir / "instrumental.wav"

    if output_path.exists():
        return str(output_path)

    drums = AudioSegment.from_wav(get_drums(stems))
    bass = AudioSegment.from_wav(get_bass(stems))
    other = AudioSegment.from_wav(get_other(stems))

    instrumental = drums.overlay(bass).overlay(other)
    instrumental.export(str(output_path), format="wav")

    return str(output_path)
