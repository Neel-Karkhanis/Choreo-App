# Backend + worker image. Both docker-compose services (`backend`, `worker`)
# build from this one Dockerfile and share the same image — only their
# `command:` differs (uvicorn vs. `rq worker`) — matching the app's existing
# one-server-file philosophy rather than maintaining two images that could
# drift apart.
#
# Two stages: `builder` installs build-only tooling (git, a compiler, for
# beat_this's git-based pip install and any wheel compilation) and produces
# a venv; `runtime` copies just that venv into a leaner final image. ffmpeg
# is installed in BOTH stages deliberately — not just at build time — because
# torchaudio/torchcodec link against its shared libraries (libavcodec etc.)
# at IMPORT time, not just when moviepy shells out to it (see server.py's
# Windows DLL-shim comment for the same requirement on that platform).

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only wheels for torch/torchaudio/torchcodec, installed from PyTorch's
# own CPU index BEFORE the rest of requirements.txt — so when pip processes
# requirements.txt's own (unpinned-to-an-index) entries for these three
# packages, they're already satisfied and it never reaches out to PyPI's
# default index, which would resolve much larger CUDA-enabled wheels this
# GPU-less deployment has no use for.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0 torchaudio==2.11.0 torchcodec==0.11.0

# Needs `git` on PATH (already installed above) for the `beat_this @
# git+https://...` entry in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src/ src/
COPY scripts/ scripts/

# Covers every bare top-level import in this codebase (server.py importing
# db/auth/tos/jobs/analysis from src/api/, and all of those + server.py
# importing beat_detection/onset_detection/source_separation/video_controls/
# progress_store from src/) without relying on server.py's own
# sys.path.insert/os.chdir shim, which still runs too (see its comment) but
# becomes a harmless no-op here since WORKDIR already puts REPO_ROOT at /app.
ENV PYTHONPATH=/app/src:/app/src/api

EXPOSE 8000

CMD ["uvicorn", "server:app", "--app-dir", "src/api", "--host", "0.0.0.0", "--port", "8000"]
