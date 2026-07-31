import source_separation
import beat_detection
import onset_detection


def analyze_audio(file_path, subdivisions=1):
    """Run the full audio analysis pipeline and return a serializable result.

    Separates the audio into stems via Demucs, detects beats and downbeats,
    groups beats into eight-counts (optionally subdivided), and runs onset
    detection on the drums and bass stems. All numpy arrays are converted to
    plain Python lists so the returned dict is directly JSON-serializable.

    Args:
        file_path: Path to the audio file to analyze.
        subdivisions: Positive integer <= 4 passed through to
            ``beat_detection.eight_count_grouping``. 1 means no subdivision.

    Returns:
        A dict with the following keys:
            - ``audio_path``: The input file path.
            - ``stems``: Dict mapping stem names to .wav file paths.
            - ``instrumental``: Path to the combined drums+bass+other track.
            - ``beats``: List of beat timestamps in seconds.
            - ``downbeats``: List of downbeat timestamps in seconds.
            - ``grouped_counts``: List of eight-count groups, each a list of
              ``(timestamp, is_beat)`` tuples.
            - ``subdivisions``: The subdivisions value used.
            - ``onsets``: Dict with ``"drums"`` and ``"bass"`` keys, each
              a dict of parallel lists ``t`` (onset timestamps in seconds)
              and ``strength`` (onset-envelope value at that onset)."""

    # audio separation
    stems = source_separation.separate(file_path)
    instrumental = source_separation.get_instrumental(stems)

    # beat detection
    beats, downbeats = beat_detection.detect_beats(file_path)
    grouped_counts = beat_detection.eight_count_grouping(beats, downbeats, subdivisions=subdivisions)

    # onset detection 
    drums_path = source_separation.get_drums(stems)
    bass_path = source_separation.get_bass(stems)
    drum_onset = onset_detection.detect_onsets(drums_path, 'drums')
    bass_onset = onset_detection.detect_onsets(bass_path, 'bass')

    api = {
        "audio_path": file_path,
        "stems": stems,
        "instrumental": instrumental,
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "grouped_counts": grouped_counts,
        "subdivisions": subdivisions,
        "onsets": {
            "drums": {"t": drum_onset["t"].tolist(), "strength": drum_onset["strength"].tolist()},
            "bass": {"t": bass_onset["t"].tolist(), "strength": bass_onset["strength"].tolist()},
        },
    }

    return api