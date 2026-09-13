"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.api.ws import router as ws_router

app = FastAPI(
    title="Bushfire UAV MARL Simulator",
    description=(
        "Research-grade multi-agent UAV wildfire detection simulator. "
        "Not an operational wildfire prediction system."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(ws_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Bushfire UAV MARL Simulator",
        "version": __version__,
        "disclaimer": "Research simulation / simplified wildfire propagation model.",
    }
