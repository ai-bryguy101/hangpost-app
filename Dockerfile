# Hangpost Matching Engine — production image
#
# Build:
#   docker build -t hangpost-matching .
#
# Run (rules-only mode, no model download):
#   docker run --rm -p 8000:8000 hangpost-matching
#
# Run (Phase 2 — embeddings; first start downloads ~90MB of model weights):
#   docker run --rm -p 8000:8000 -e HANGPOST_MODE=embeddings hangpost-matching
#
# Run (Phase 3 — learned ranker; mount a directory containing the joblib):
#   docker run --rm -p 8000:8000 \
#     -e HANGPOST_MODE=learned \
#     -e HANGPOST_LEARNED_MODEL_PATH=/models/learned_ranker.joblib \
#     -v "$(pwd)/models:/models" \
#     hangpost-matching

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HANGPOST_MODE=rules

WORKDIR /app

# Install build deps for any wheel-less packages, then drop them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal install surface first to maximise layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[ml,serve]"

# App code (kept as a separate layer so source edits don't bust the dep cache).
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

# Non-root user for runtime.
RUN useradd --create-home --uid 1000 hangpost \
    && chown -R hangpost:hangpost /app
USER hangpost

CMD ["uvicorn", "hangpost_matching.server:app", "--host", "0.0.0.0", "--port", "8000"]
