"""main.py — FastAPI app entry point.

Run with:
    uvicorn pl_predictor.api.main:app --reload --host 0.0.0.0 --port 8000

`--host 0.0.0.0` matters for reaching this from another device (phone over
Tailscale, or another machine on the LAN) — the default `127.0.0.1` only
accepts connections from the same machine.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(title="PL Predictor API")

# Wide open on purpose: this server is only ever meant to be reached over a
# private network (localhost, LAN, or a personal Tailscale tailnet) — never
# exposed to the public internet — so there's no real origin to restrict to,
# and restricting it would just break access from a phone/other device.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
