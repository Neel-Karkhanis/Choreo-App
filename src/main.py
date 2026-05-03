import source_separation
import audio_utils
import beat_detection

def main():
    #loading audio
    file_path = audio_utils.get_file_path()
    audio, sr = audio_utils.load_audio(file_path)

    #audio separation
    stems = source_separation.separate(file_path)
    # intrumental = source_separation.get_instrumental(stems) (not going to be used yet)

    #beat detection
    beats, downbeats = beat_detection.detect_beats(file_path)
    grouped_counts = beat_detection.eight_count_grouping(beats, downbeats, subdivisions=2)


if __name__ == "__main__":
    main()