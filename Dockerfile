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

# Dependencies FIRST, keyed on pyproject.toml and requirements-lock.txt alone. The dependency layer is large (xgboost, pandas, scipy,
# scikit-learn) and used to sit AFTER `COPY src/`, so every commit rebuilt it with a new digest and the VPS kept a
# full extra copy per deploy. Now an unchanged dependency set reuses the SAME layer digest across commits and a deploy
# adds only the thin app layers.
# `pip install -e .` needs the package directory to exist to resolve it, so a stub stands in for the real source;
# the real source is copied right after and PYTHONPATH=/app/src (set below) is what the app imports from.
COPY pyproject.toml requirements-lock.txt ./
# Editable install, matching local dev exactly (README's own `pip install -e
# . -c requirements-lock.txt`) — NOT a cosmetic choice: config.py derives
# PROJECT_ROOT (and everything under it: MODELS_DIR, CACHE_DIR,
# FRONTEND_DIST_DIR) from `Path(__file__).resolve().parents[2]`. A
# non-editable install copies the package into site-packages, so that
# math would resolve to somewhere under site-packages instead of /app —
# silently breaking every path in the app (confirmed the hard way: the
# first deploy attempt served the API's plain JSON root instead of the
# built frontend, because FRONTEND_DIST_DIR pointed nowhere real).
# Pinned to this project's own known-working versions rather than an
# unconstrained install, to avoid a newer pandas/xgboost/etc. silently
# changing behavior in the one environment nobody develops against directly.
RUN mkdir -p src/pl_predictor && touch src/pl_predictor/__init__.py \
    && pip install --no-cache-dir -e . -c requirements-lock.txt \
    && rm -rf src
COPY src/ ./src/

# Ships with a real trained model immediately (models/*.json are
# git-tracked) instead of needing a full historical fetch+train before the
# first prediction can be served.
COPY models/ ./models/
# The precomputed Fixtures/Data Hub data this deployment actually serves
# (see public_snapshot.py's module docstring) — generated locally
# (`python -m pl_predictor.public_snapshot`) and committed, not built in
# this image. Must exist before building — run that command once if this
# COPY fails on a fresh checkout.
COPY data/public_snapshot.json ./data/public_snapshot.json
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn pl_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
