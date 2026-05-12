import source_separation
import beat_detection
import onset_detection
import bookmark


def analyze_audio(file_path, subdivisions=1):
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

    # bookmark dictionary
    bookmarks = bookmark.load_bookmarks(file_path)

    api = {
        "audio_path": file_path,
        "stems": stems,
        "instrumental": instrumental,
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "grouped_counts": grouped_counts,
        "subdivisions": subdivisions,
        "onsets": {
            "drums": drum_onset.tolist(),
            "bass": bass_onset.tolist(),
        },
        "existing_bookmarks": bookmarks,
    }

    return api