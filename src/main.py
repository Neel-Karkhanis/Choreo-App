import json
from audio_analyzer import analyze_audio
from audio_utils import get_file_path

def main():
    file_path = get_file_path()
    result = analyze_audio(file_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()