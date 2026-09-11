# Cloud Run (us-central1). Public metadata finder only.
# No Gemini. No Vertex. No Firestore. Rank baked graph only.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY out/link_graph.json out/goal_find_A.json ./out/
COPY fixtures/catalog_notes.json ./fixtures/catalog_notes.json

RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["sh", "-c", "exec python -m cancer_output_atlas serve --out /app/out --host 0.0.0.0"]

