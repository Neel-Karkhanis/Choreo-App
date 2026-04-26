import numpy as np
import librosa


def detect_beats(audio, sr):
    """Detect beat timestamps from audio data.

    Args:
        audio: NumPy array of audio samples.
        sr: Sample rate of the audio.

    Returns:
        A NumPy array of beat timestamps in seconds.

    Raises:
        TypeError: If audio is not array-like with numeric values or sr is not numeric.
        ValueError: If audio is empty or sr is not positive.
    """
    audio = np.asarray(audio)
    if not np.issubdtype(audio.dtype, np.number):
        raise TypeError("audio must contain numeric values")
    if audio.size == 0:
        raise ValueError("audio must not be empty")
    if not isinstance(sr, (int, float)):
        raise TypeError("sr must be numeric")
    if sr <= 0:
        raise ValueError("sr must be positive")

    _, beats = librosa.beat.beat_track(y=audio, sr=sr, start_bpm=96)
    return librosa.frames_to_time(beats, sr=sr)


def merge_beats(drum_beats, mix_beats, gap_threshold=1.0):
    """Merge two beat arrays, keeping mix beats that are far enough from drum beats.

    Args:
        drum_beats: NumPy array of beat timestamps from the drum stem, in seconds.
        mix_beats: NumPy array of beat timestamps from the full mix, in seconds.
        gap_threshold: Minimum distance in seconds a mix beat must be from all
            drum beats to be included in the result.

    Returns:
        A sorted NumPy array of merged beat timestamps in seconds.

    Raises:
        TypeError: If drum_beats or mix_beats are not array-like, or
            gap_threshold is not numeric.
        ValueError: If gap_threshold is not positive.
    """
    drum_beats = np.asarray(drum_beats)
    mix_beats = np.asarray(mix_beats)

    if not np.issubdtype(drum_beats.dtype, np.number):
        raise TypeError("drum_beats must contain numeric values")
    if not np.issubdtype(mix_beats.dtype, np.number):
        raise TypeError("mix_beats must contain numeric values")
    if not isinstance(gap_threshold, (int, float)):
        raise TypeError("gap_threshold must be numeric")
    if gap_threshold <= 0:
        raise ValueError("gap_threshold must be positive")

    if len(drum_beats) == 0:
        return mix_beats

    merged = list(drum_beats)
    for t in mix_beats:
        if min(abs(drum_beats - t)) >= gap_threshold:
            merged.append(t)

    return np.sort(merged)

def eight_count_grouping(merged_beats, subdivisions=1):
    """Group beats into eight-counts, optionally subdividing between beats.

    Args:
        merged_beats: Array-like of beat timestamps in seconds.
        subdivisions: Number of subdivisions per beat interval. Must be a
            positive integer. Defaults to 1 (no subdivision).

    Returns:
        A list of eight-count groups. Each group is a list of
        (timestamp, is_beat) tuples, where is_beat is True for original
        beats and False for interpolated subdivision points.

    Raises:
        TypeError: If merged_beats is not array-like with numeric values,
            or subdivisions is not an integer.
        ValueError: If merged_beats is empty or subdivisions is not positive.
    """
    try:
        merged_beats = np.asarray(merged_beats, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("merged_beats must contain numeric values")
    if merged_beats.size == 0:
        raise ValueError("merged_beats must not be empty")
    if not isinstance(subdivisions, int):
        raise TypeError("subdivisions must be an integer")
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    subdivided_list = []
    if subdivisions != 1:
        for i, beat in enumerate(merged_beats):
            if i >= len(merged_beats) - 1:
                break
            
            subdivided_list.append((beat, True))
            time = merged_beats[i+1] - beat
            time /= subdivisions
            for j in range(subdivisions - 1):
                subdivided_list.append((beat + (j+1)*time, False))
            time = 0

        subdivided_list.append((merged_beats[-1], True))

    else:
        for beat in merged_beats:
            subdivided_list.append((beat, True))

    grouped_counts = []
    eight_count = []

    for i, beat in enumerate(subdivided_list):
        if i != 0 and i % (8 * subdivisions) == 0:
            grouped_counts.append(eight_count)
            eight_count = []

        eight_count.append(beat)

    grouped_counts.append(eight_count)

    return grouped_counts
