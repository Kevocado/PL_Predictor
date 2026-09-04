"""main.py — FastAPI app entry point.

Run with:
    uvicorn pl_predictor.api.main:app --reload --host 0.0.0.0 --port 8000

`--host 0.0.0.0` matters for reaching this from another device (phone over
Tailscale, or another machine on the LAN) — the default `127.0.0.1` only
accepts connections from the same machine.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import FRONTEND_DIST_DIR, PUBLIC_MODE, PUBLIC_SNAPSHOT_POLL_SECONDS
from .routes import (
    background_tracking_tick,
    maybe_auto_retrain,
    refresh_public_snapshot_from_remote,
    router,
    warm_caches,
)

# How often to check whether the current season has new played matches the
# model hasn't trained on yet (see routes.warm_caches's / maybe_auto_retrain's
# docstrings) — this is the answer to "when will a result affect my model":
# within one interval of it becoming available, automatically, no manual
# "Retrain models" click needed. Hourly is frequent enough for a football
# season (matches come in batches on matchdays, not continuously) without
# spending a retrain's ~10-30s of CPU needlessly often.
_AUTO_RETRAIN_INTERVAL_SECONDS = 3600
_TRACKING_INTERVAL_SECONDS = 300


async def _auto_retrain_loop():
    while True:
        await asyncio.sleep(_AUTO_RETRAIN_INTERVAL_SECONDS)
        await asyncio.to_thread(maybe_auto_retrain)


async def _tracking_loop():
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        await asyncio.to_thread(background_tracking_tick)


async def _initial_sync():
    await asyncio.to_thread(warm_caches)
    # Player-history fitting and per-fixture player prewarming can issue
    # dozens of remote requests and train several models. They are useful
    # once a fixture modal is opened, but must not contend with the first
    # dashboard/FPL request after local startup.


async def _public_snapshot_poll_loop():
    while True:
        await asyncio.to_thread(refresh_public_snapshot_from_remote)
        await asyncio.sleep(PUBLIC_SNAPSHOT_POLL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The public deployment serves everything from public_snapshot.py's
    # precomputed file (see routes.py's PUBLIC_MODE branches) and never
    # calls any of warm_caches/backfill/prewarm/retrain/tracking's
    # functions at all — running them anyway would rebuild the exact
    # live-serving state (FixtureFeatureContext, Elo/Pi, xG, ...) this
    # whole snapshot approach exists to avoid, which is what was OOM-
    # crashing the free-tier deployment in the first place.
    if PUBLIC_MODE:
        # Still needs *a* way to see each new snapshot without that live
        # computation — polling the raw file on GitHub (see routes.py::
        # refresh_public_snapshot_from_remote) rather than waiting for
        # Render's autoDeploy to rebuild the whole image on every data-only
        # commit, which used to mean a full container restart several
        # times a day on top of this host's own idle-sleep cycle.
        poll_task = asyncio.create_task(_public_snapshot_poll_loop())
        yield
        poll_task.cancel()
        return

    # Fire-and-forget in a background thread: warms every cache a real
    # request would otherwise pay for on first use (see warm_caches's
    # docstring), without delaying the server from accepting connections —
    # matters most for uvicorn --reload, which re-runs this on every file
    # save during development.
    asyncio.create_task(_initial_sync())
    retrain_task = asyncio.create_task(_auto_retrain_loop())
    tracking_task = asyncio.create_task(_tracking_loop())
    yield
    retrain_task.cancel()
    tracking_task.cancel()


app = FastAPI(title="PL Predictor API", lifespan=lifespan)

# Wide open on purpose: this server is only ever meant to be reached over a
# private network (localhost, LAN, or a personal Tailscale tailnet) — never
# exposed to the public internet — so there's no real origin to restrict to,
# and restricting it would just break access from a phone/other device. The
# public deployment (see PUBLIC_MODE branches throughout routes.py) is
# read-only and has no admin surface reachable at all (see
# routes.py::_admin_only) — there's nothing left for a login to protect,
# so it's a plain public site with no guest password.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Only present in the public Docker deployment (see repo-root Dockerfile),
# which builds frontend/dist before starting the server — local dev never
# has this directory, and keeps using `npm run dev` + this app's plain JSON
# root below exactly as before.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:

    @app.get("/")
    def root():
        return {"status": "ok", "docs": "/docs"}
