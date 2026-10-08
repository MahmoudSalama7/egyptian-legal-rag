# ============================================================================
# Egyptian Legal RAG — Production Multi-Stage Dockerfile
# ============================================================================
# Stage 1: Build — install dependencies via uv (cached) and build wheel
# Stage 2: Runtime — minimal image, non-root user, healthcheck
# ============================================================================

# ------------------------------------------------------------------
# Stage 1 — Builder
# ------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for ultra-fast dependency resolution & caching
RUN pip install --no-cache-dir uv

# Copy dependency manifest first (maximises Docker layer cache)
COPY pyproject.toml ./

# Install runtime dependencies only (no editable install yet)
RUN uv pip install --system --no-cache .

# Copy application source
COPY src/ ./src/

# Install the project itself
RUN uv pip install --system --no-cache .

# ------------------------------------------------------------------
# Stage 2 — Runtime (minimal)
# ------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="MahmoudSalama7" \
      description="Egyptian Civil Code Legal RAG — Production API" \
      version="0.2.0"

# Create non-root user
RUN useradd --create-home --uid 1000 appuser

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code & data
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser data/processed/civil_code_chunks.json /app/data/processed/civil_code_chunks.json
COPY --chown=appuser:appuser data/processed/index_store/ /app/data/processed/index_store/

WORKDIR /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Container health check — lightweight HTTP probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "65"]
