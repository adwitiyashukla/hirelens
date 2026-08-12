FROM node:22-slim AS web

WORKDIR /web

COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[api]"

COPY --from=web /web/dist ./web/dist

RUN useradd --create-home --shell /bin/bash hirelens \
    && mkdir -p /app/.hirelens_cache \
    && chown -R hirelens:hirelens /app
USER hirelens

EXPOSE 8000

CMD ["uvicorn", "hirelens.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
