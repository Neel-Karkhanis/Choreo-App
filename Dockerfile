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

# --proxy-headers --forwarded-allow-ips=<CIDR> is what makes
# request.client.host (Phase 5's per-IP rate limit — see server.py's
# CHOREO_IMPORT_RATE_LIMIT comment) the REAL caller's address instead of
# Caddy's own container IP. Without it, uvicorn ignores the
# X-Forwarded-For header Caddy's reverse_proxy sets by default, and every
# request would appear to come from the same address — silently turning
# the rate limit into either a no-op or a lockout shared by every visitor,
# neither of which anyone would notice from a quick smoke test.
#
# The CIDR (not '*') is docker-compose.yml's pinned default-network
# subnet — see that file's own comment on why a wildcard here would stop
# being safe the moment `backend` ever gained a `ports:` mapping, and why
# pinning the subnet instead keeps that mistake from becoming a spoofing
# vector even if it happens. This value and that file's subnet must
# always match; local (non-Docker) `venv`/`uvicorn` dev never sets
# X-Forwarded-For at all, so this is a no-op there regardless.
CMD ["uvicorn", "server:app", "--app-dir", "src/api", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=172.30.0.0/24"]
