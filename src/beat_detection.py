import os
import numpy as np
from beat_this.inference import File2Beats

file2beats = File2Beats(checkpoint_path="final0", device="cpu")

def detect_beats(audio_path):
    """Detect beat and downbeat timestamps in an audio file.

    Runs the beat_this File2Beats model on the file at audio_path and
    returns two arrays of timestamps in seconds: every detected beat,
    and the subset of those beats that are downbeats (the "1" of each bar).

    Args:
        audio_path: Path to the audio file.

    Returns:
        A tuple (beats, downbeats) of numpy arrays of timestamps in seconds.

    Raises:
        TypeError: If audio_path is not a string.
        ValueError: If audio_path is empty.
        FileNotFoundError: If audio_path does not exist.
    """
    if not isinstance(audio_path, str):
        raise TypeError("audio_path must be a string")
    if not audio_path:
        raise ValueError("audio_path must not be empty")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    beats, downbeats = file2beats(audio_path)
    return (beats, downbeats)

def eight_count_grouping(beats, downbeats, subdivisions=1):
    """Align beats to the first downbeat and group them into eight-counts.

    The beats array is trimmed to start at the first downbeat, then optionally
    subdivided so each beat interval is split into `subdivisions` equal parts.
    Each entry in the returned groups is a (timestamp, is_beat) tuple, where
    is_beat is True for original beats and False for inserted subdivision points.
    Entries are then chunked into groups of size 8 * subdivisions; the final
    group may be partial.

    Args:
        beats: 1-D numpy array of beat timestamps in seconds.
        downbeats: 1-D numpy array of downbeat timestamps in seconds. The
            first downbeat must appear in `beats`.
        subdivisions: Positive integer <= 4. The number of equal parts each
            beat interval is split into. 1 means no subdivision.

    Returns:
        A list of groups, where each group is a list of (timestamp, is_beat)
        tuples of length up to 8 * subdivisions.

    Raises:
        TypeError: If beats or downbeats is not a numpy array, or if
            subdivisions is not an int.
        ValueError: If beats or downbeats is empty, if subdivisions is not
            in the range 1..4, or if the first downbeat is not in beats.
    """
    if not isinstance(beats, np.ndarray):
        raise TypeError("beats must be a numpy array")
    if not isinstance(downbeats, np.ndarray):
        raise TypeError("downbeats must be a numpy array")
    if not np.issubdtype(beats.dtype, np.number):
        raise TypeError("beats must contain numeric values")
    if not np.issubdtype(downbeats.dtype, np.number):
        raise TypeError("downbeats must contain numeric values")
    if beats.size == 0:
        raise ValueError("beats must not be empty")
    if downbeats.size == 0:
        raise ValueError("downbeats must not be empty")
    if not isinstance(subdivisions, int) or isinstance(subdivisions, bool):
        raise TypeError("subdivisions must be an int")
    if subdivisions < 1 or subdivisions > 4:
        raise ValueError("subdivisions must be a positive integer <= 4")

    matches = np.where(beats == downbeats[0])[0]
    if len(matches) == 0:
        raise ValueError("First downbeat isn't in beats")

    start_index = matches[0]
    beats = beats[start_index:]

    subdivided_list = []
    if subdivisions != 1:
        for i, beat in enumerate(beats):
            if i >= len(beats) - 1:
                break

            subdivided_list.append((beat, True))
            time = beats[i+1] - beat
            time /= subdivisions
            for j in range(subdivisions - 1):
                subdivided_list.append((beat + (j+1)*time, False))
            time = 0

        subdivided_list.append((beats[-1], True))

    else:
        for beat in beats:
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
