# ═══════════════════════════════════════════════════════════════
# DOCKERFILE — MLOPS Clothes Project
# ═══════════════════════════════════════════════════════════════
#
# ✅ NEVER CHANGE:
#   - Multi-stage build structure (builder + final)
#   - Non-root user setup (appuser)
#   - PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED env vars
#   - Health check structure
#   - Gunicorn CMD pattern
#   - mkdir + chown block (folder permissions)
#
# 🔄 ALWAYS CHANGE:
#   - Python version (3.11) → match your local python version
#   - PORT number → if your app uses different port
#   - Gunicorn workers count → based on your server CPU cores
#     Formula: workers = (2 x CPU cores) + 1
#   - mkdir folders → add any folder your app writes to
# ═══════════════════════════════════════════════════════════════


# ───────────────────────────────────────────
# STAGE 1 — Builder
# ───────────────────────────────────────────

# 🔄 CHANGE: python version to match your local version
FROM python:3.11-slim AS builder

# ✅ NEVER CHANGE
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# ─────────────────────────────────────────────────────────────
# 🔄 CHANGE THIS BLOCK based on your dependency file:
#
# IF you use pyproject.toml  →  COPY pyproject.toml .
#                               RUN pip install --no-cache-dir --prefix=/install ".[dev]"
#
# IF you use requirements.txt → COPY requirements.txt .
#                               RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# ─────────────────────────────────────────────────────────────
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install ".[dev]"


# ───────────────────────────────────────────
# STAGE 2 — Final Image
# ───────────────────────────────────────────

# 🔄 CHANGE: must match builder python version above
FROM python:3.11-slim

# ✅ NEVER CHANGE
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 🔄 CHANGE: project name and version
ENV APP_NAME="mlops-clothes-project" \
    APP_VERSION="2.0.0" \
    PORT=8000

# ✅ NEVER CHANGE: copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# ✅ NEVER CHANGE: create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

# ✅ NEVER CHANGE: copy project files
COPY --chown=appuser:appgroup . .

# ─────────────────────────────────────────────────────────────
# ✅ NEVER CHANGE: folder permission fix
# Must create ALL writable folders BEFORE switching to appuser
# appuser cannot create new folders — only write to existing ones
#
# 🔄 CHANGE: add any folder your app writes to at runtime
#   Your config.py calls: Path("logs").mkdir(...)
#   So "logs" must exist and be owned by appuser before startup
# ─────────────────────────────────────────────────────────────
RUN mkdir -p /app/logs && \
    chown -R appuser:appgroup /app/logs

# ✅ NEVER CHANGE: switch to non-root user AFTER folder setup
USER appuser

# ✅ NEVER CHANGE
EXPOSE 8000

# ✅ NEVER CHANGE
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ✅ NEVER CHANGE: Gunicorn command structure
# 🔄 CHANGE: --workers based on CPU cores → formula: (2 x cores) + 1
# 🔄 CHANGE: app:app → yourfile:yourvariable if app.py is renamed
CMD ["gunicorn", "app:app", \
     "--workers", "3", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]