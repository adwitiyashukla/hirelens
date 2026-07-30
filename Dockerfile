# Two stages. The first builds the dashboard, the second runs the API and serves
# the result. One image and one process, because the free hosting tiers this is
# meant to deploy to give you exactly one container.

FROM node:22-slim AS web

WORKDIR /web

# Manifest first: a change to a component should not reinstall node_modules.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web/ ./
RUN npm run build


# Slim rather than alpine: PyMuPDF ships manylinux wheels that alpine's musl libc
# cannot use, so alpine would trigger a source build of the whole PDF stack.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata first, so a code change does not invalidate the pip layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[api]"

# The compiled dashboard. `hirelens.api.app.find_static_dir` looks here.
COPY --from=web /web/dist ./web/dist

# Run unprivileged. Nothing here needs root.
RUN useradd --create-home --shell /bin/bash hirelens \
    && mkdir -p /app/.hirelens_cache \
    && chown -R hirelens:hirelens /app
USER hirelens

EXPOSE 8000

# One worker on purpose: the background screening runner keeps run state in
# process memory, so multiple workers would each hold a different view of a run.
# Scaling out means moving to a real job queue, which is the documented next step.
CMD ["uvicorn", "hirelens.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
