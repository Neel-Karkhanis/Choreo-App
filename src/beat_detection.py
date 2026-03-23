import os

from scipy.io import wavfile
import numpy as np


def load_audio(filepath):
    """Load a WAV file from disk.

    Args:
        filepath: Path to the WAV file to read.

    Returns:
        A tuple of `(sample_rate, data)` returned by `scipy.io.wavfile.read`.

    Raises:
        ValueError: If `filepath` is empty or does not point to a `.wav` file.
        FileNotFoundError: If `filepath` does not exist.
    """
    if not isinstance(filepath, (str, os.PathLike)):
        raise TypeError("filepath must be a path-like value")
    if not filepath:
        raise ValueError("filepath must not be empty")
    if not str(filepath).lower().endswith(".wav"):
        raise ValueError("filepath must point to a .wav file")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at: {filepath}")
    sample_rate, data = wavfile.read(filepath)
    return sample_rate, data


def get_frame(data, frame_size=1024):
    """Split audio data into equally sized frames.

    Args:
        data: Indexable audio sample data.
        frame_size: Number of samples per frame.

    Returns:
        A list containing slices of `data` with at most `frame_size` samples each.

    Raises:
        TypeError: If `data` is not sliceable or `frame_size` is not an integer.
        ValueError: If `frame_size` is not positive.
    """
    if not hasattr(data, "__len__") or not hasattr(data, "__getitem__"):
        raise TypeError("data must be a sliceable sequence")
    if not isinstance(frame_size, int):
        raise TypeError("frame_size must be an integer")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    return [data[i:i + frame_size] for i in range(0, len(data), frame_size)]


def RMS(frame):
    """Compute the root-mean-square energy of one frame.

    Args:
        frame: A non-empty numeric frame of audio samples.

    Returns:
        The RMS value as a float-compatible NumPy scalar.

    Raises:
        ValueError: If `frame` is empty.
        TypeError: If `frame` does not contain numeric samples.
    """
    frame_array = np.asarray(frame)
    if frame_array.size == 0:
        raise ValueError("frame must not be empty")
    if not np.issubdtype(frame_array.dtype, np.number):
        raise TypeError("frame must contain numeric samples")
    return np.sqrt(np.mean(np.square(frame_array)))


def get_RMS_energy(list_of_frames):
    """Compute RMS energy for every frame in a sequence.

    Args:
        list_of_frames: A sequence of non-empty numeric audio frames.

    Returns:
        A list of RMS values, one per frame.

    Raises:
        ValueError: If `list_of_frames` is `None`.
    """
    if list_of_frames is None:
        raise ValueError("list_of_frames must not be None")
    return [RMS(frame) for frame in list_of_frames]


def get_onset_strength(list_of_RMS_values):
    """Measure positive frame-to-frame increases in RMS energy.

    Args:
        list_of_RMS_values: A sequence of numeric RMS values.

    Returns:
        A list whose entries are the positive increases between adjacent RMS values.

    Raises:
        ValueError: If `list_of_RMS_values` is `None`.
        TypeError: If any RMS value is non-numeric.
    """
    if list_of_RMS_values is None:
        raise ValueError("list_of_RMS_values must not be None")
    list_of_onset_strength = []
    for i in range(len(list_of_RMS_values) - 1):
        prev_value = list_of_RMS_values[i]
        value = list_of_RMS_values[i + 1]
        if not np.issubdtype(type(prev_value), np.number):
            raise TypeError("RMS values must be numeric")
        if not np.issubdtype(type(value), np.number):
            raise TypeError("RMS values must be numeric")
        onset = value - prev_value

        if onset < 0:
            onset = 0

        list_of_onset_strength.append(onset)
        
    return list_of_onset_strength


def get_beats(list_of_onset_strength, threshold=None):
    """Find frame indices whose onset strength exceeds a threshold.

    Args:
        list_of_onset_strength: A sequence of numeric onset-strength values.
        threshold: Optional numeric threshold. If omitted, the mean onset strength is used.

    Returns:
        A list of frame indices where onset strength is greater than the threshold.

    Raises:
        ValueError: If `list_of_onset_strength` is `None`.
        TypeError: If onset strengths or `threshold` are non-numeric.
    """
    if list_of_onset_strength is None:
        raise ValueError("list_of_onset_strength must not be None")
    if len(list_of_onset_strength) == 0:
        return []

    for onset in list_of_onset_strength:
        if not np.issubdtype(type(onset), np.number):
            raise TypeError("onset strengths must be numeric")

    if threshold is None:
        threshold = np.mean(list_of_onset_strength)
    else:
        if not np.issubdtype(type(threshold), np.number):
            raise TypeError("threshold must be numeric")

    return [i for i, onset in enumerate(list_of_onset_strength) if onset > threshold]


def get_timestamp(beat_indices, frame_size, sample_rate):
    """Convert beat frame indices into timestamps in seconds.

    Args:
        beat_indices: A sequence of non-negative frame indices.
        frame_size: Number of samples represented by each frame.
        sample_rate: Audio sample rate in samples per second.

    Returns:
        A list of timestamps in seconds for each beat index.

    Raises:
        ValueError: If `beat_indices` is `None`, indices are negative, or numeric values are non-positive.
        TypeError: If indices are not integers or numeric parameters have the wrong type.
    """
    if beat_indices is None:
        raise ValueError("beat_indices must not be None")
    if not isinstance(frame_size, int):
        raise TypeError("frame_size must be an integer")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    if not np.issubdtype(type(sample_rate), np.number):
        raise TypeError("sample_rate must be numeric")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    for indice in beat_indices:
        if not isinstance(indice, (int, np.integer)):
            raise TypeError("beat indices must be integers")
        if indice < 0:
            raise ValueError("beat indices must be non-negative")

    return [indice * frame_size / sample_rate for indice in beat_indices]
