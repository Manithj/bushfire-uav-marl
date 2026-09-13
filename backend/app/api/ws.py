"""WebSocket stream of simulation state."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.simulation.manager import MANAGER

router = APIRouter()


@router.websocket("/ws/simulations/{simulation_id}")
async def simulation_ws(websocket: WebSocket, simulation_id: str) -> None:
    if MANAGER.get(simulation_id) is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    queue = await MANAGER.subscribe(simulation_id)
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        MANAGER.unsubscribe(simulation_id, queue)
