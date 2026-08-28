# Public, password-gated PL Predictor deployment — one image serving the
# built React frontend and the FastAPI backend from a single process/origin
# (see api/main.py's StaticFiles mount and api/auth.py's GuestAuthMiddleware).
# The private/full app (npm run dev + uvicorn --reload, no PUBLIC_MODE) is
# untouched by this file — it's only used for the public deployment.

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Baked in at build time, not runtime — Vite inlines import.meta.env values
# into the built bundle (see frontend/src/lib/publicMode.ts).
ENV VITE_PUBLIC_MODE=true
ENV VITE_API_BASE_URL=/api
RUN npm run build

FROM python:3.13-slim AS backend
WORKDIR /app

COPY pyproject.toml requirements-lock.txt ./
COPY src/ ./src/
# Pinned to this project's own known-working versions (same file `pip
# install -e . -c requirements-lock.txt` uses locally) rather than an
# unconstrained install, to avoid a newer pandas/xgboost/etc. silently
# changing behavior in the one environment nobody develops against directly.
RUN pip install --no-cache-dir . -c requirements-lock.txt

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn pl_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
