import os
import librosa
import numpy as np

ONSET_PARAMS = {
    'drums': {
        'hop_length': 512,
        'backtrack': False,
        'delta': 0.05,
        'pre_max': 1,
        'post_max': 1,
        'pre_avg': 1,
        'post_avg': 1,
        'wait': 10,
    },
    'bass': {
        'hop_length': 512,
        'backtrack': False,
        'delta': 0.08,
        'pre_max': 3,
        'post_max': 3,
        'pre_avg': 3,
        'post_avg': 3,
        'wait': 10,
    },
}


def detect_onsets(stem_path, stem_type):
    """Detect onset timestamps and strengths for a single separated stem.

    Loads the audio at ``stem_path`` and runs ``librosa.onset.onset_detect``
    with parameters tuned for ``stem_type`` (see ``ONSET_PARAMS``). For the
    ``"bass"`` stem a custom mel-spectrogram-based onset envelope aggregated
    by median is used, which yields cleaner onsets on low-frequency content;
    other stems use librosa's default envelope. The envelope is computed
    explicitly (identical to what ``onset_detect`` would build internally)
    so each onset's strength can be sampled at its detected frame.

    Args:
        stem_path: Path to the stem audio file (e.g. ``drums.wav``).
        stem_type: Which stem ``stem_path`` is. Must be a key in
            ``ONSET_PARAMS`` (currently ``"drums"`` or ``"bass"``).

    Returns:
        A dict of two parallel 1-D numpy arrays:
            - ``"t"``: onset timestamps in seconds, sorted ascending.
            - ``"strength"``: the onset-strength envelope sampled at each
              onset's frame — raw (un-normalized) values from the same
              envelope peak-picking ran on.

    Raises:
        TypeError: If stem_path or stem_type is not a string.
        ValueError: If stem_path is empty, or if stem_type is not a key
            in ``ONSET_PARAMS``.
        FileNotFoundError: If stem_path does not exist.
    """
    if not isinstance(stem_path, str):
        raise TypeError("stem_path must be a string")
    if not stem_path:
        raise ValueError("stem_path must not be empty")
    if not isinstance(stem_type, str):
        raise TypeError("stem_type must be a string")
    if stem_type not in ONSET_PARAMS:
        raise ValueError(
            f"Unknown stem_type: {stem_type!r}. "
            f"Expected one of {sorted(ONSET_PARAMS.keys())}"
        )
    if not os.path.exists(stem_path):
        raise FileNotFoundError(f"File not found: {stem_path}")

    data, sr = librosa.load(stem_path, sr=None)
    params = ONSET_PARAMS[stem_type]
    hop_length = params['hop_length']

    if stem_type == 'bass':
        # use a custom onset envelope based on log-energy difference
        onset_env = librosa.onset.onset_strength(
            y=data, sr=sr, hop_length=hop_length,
            feature=librosa.feature.melspectrogram,
            aggregate=np.median
        )
    else:
        onset_env = librosa.onset.onset_strength(y=data, sr=sr, hop_length=hop_length)

    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames', **params
    )
    return {
        "t": librosa.frames_to_time(frames, sr=sr, hop_length=hop_length),
        "strength": onset_env[frames],
    }
