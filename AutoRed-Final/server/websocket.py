import json
import logging
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("autored_server")


class WebSocketManager:
    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self._connections:
            self._connections[run_id] = set()
        self._connections[run_id].add(websocket)
        logger.info(f"[WS] Connected: run_id={run_id}, total_connections={len(self._connections[run_id])}")

    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self._connections:
            self._connections[run_id].discard(websocket)
            remaining = len(self._connections[run_id])
            logger.info(f"[WS] Disconnected: run_id={run_id}, remaining={remaining}")
            if not self._connections[run_id]:
                del self._connections[run_id]
                logger.info(f"[WS] Cleaned up empty connection set for run_id={run_id}")

    async def send_attempt(self, run_id: str, attempt: dict):
        if run_id not in self._connections:
            logger.warning(f"[WS] send_attempt: NO connections for run_id={run_id} (message DROPPED)")
            return
        conns = len(self._connections[run_id])
        message = json.dumps({"type": "attempt_update", "run_id": run_id, "attempt": attempt})
        disconnected = set()
        # Iterate over a copy to avoid "Set changed size during iteration"
        for ws in set(self._connections[run_id]):
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"[WS] send_attempt: Failed to send to WebSocket: {e}")
                disconnected.add(ws)
        self._connections[run_id] -= disconnected
        logger.info(f"[WS] Sent attempt_update: run_id={run_id}, attempt={attempt.get('attempt_number')}, delivered_to={conns - len(disconnected)}/{conns}")

    async def send_run_complete(self, run_id: str, run: dict):
        if run_id not in self._connections:
            logger.warning(f"[WS] send_run_complete: NO connections for run_id={run_id} (message DROPPED)")
            return
        conns = len(self._connections[run_id])
        message = json.dumps({"type": "run_complete", "run_id": run_id, "run": run})
        sent = 0
        for ws in set(self._connections[run_id]):
            try:
                await ws.send_text(message)
                sent += 1
            except Exception as e:
                logger.warning(f"[WS] send_run_complete: Failed to send to WebSocket: {e}")
        logger.info(f"[WS] Sent run_complete: run_id={run_id}, delivered_to={sent}/{conns}")

    async def broadcast(self, run_id: str, data: dict):
        if run_id not in self._connections:
            logger.warning(f"[WS] broadcast: NO connections for run_id={run_id} (message DROPPED)")
            return
        message = json.dumps(data)
        for ws in set(self._connections[run_id]):
            try:
                await ws.send_text(message)
            except Exception:
                pass


ws_manager = WebSocketManager()
