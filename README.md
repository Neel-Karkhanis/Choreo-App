# Choreo App

A tool for learning/creating choreography from video or songs. Instead of endlessly rewinding, this app gives you precise controls to slow down, loop, and analyze movement in any video file.

## Features (in progress)
- Playback speed control
- Loop over custom start/stop points
- Frame-by-frame stepping
- Beat detection (kicks and snares visualized separately)
- Encrypted storage and secure shareable links

## Tech Stack
- Python + FastAPI (backend)
- Flutter (mobile/desktop frontend, coming later)
- Librosa, MediaPipe, PyTorch (ML features)

## Setup
```bash
# clone the repo
git clone https://github.com/Neel-Karkhanis/choreo-app.git
cd choreo-app

# create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Project Structure
```
choreo-app/
├── video_controls.py   # speed, looping, frame stepping
├── README.md
└── requirements.txt
```

## Roadmap
- [x] Video controls (speed, loop, frame stepping)
- [ ] Beat detection from scratch
- [ ] Streamlit UI
- [ ] Pose estimation
- [ ] FastAPI backend
- [ ] Flutter frontend

