"""FastAPI application – WebSocket /ws + static file serving.

Run:  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .trainer import BabelTrainer, TrainStats

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="Babel – emergent symbolic communication", version="1.0")


# ── Static files ──────────────────────────────────────────────────────────────

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "babel-whispering-dart.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ── Message builders ─────────────────────────────────────────────────────────

def _stats_msg(s: TrainStats) -> dict:
    return {
        "type":         "stats",
        "epoch":        s.epoch,
        "listener_acc": s.listener_acc,
        "rival_acc":    s.rival_acc,
        "consistency":  s.consistency,
        "entropy":      s.entropy,
        "topsim":       s.topsim,
        "lexicon":      s.lexicon,
    }


def _round_msg(r: dict) -> dict:
    return {"type": "round", **r}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    trainer = BabelTrainer(seed=42)
    evolve_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()
    loop = asyncio.get_event_loop()

    async def safe_send(payload: dict) -> None:
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except Exception:
                pass

    async def _evolve_loop() -> None:
        """Train one step, sample one round, push both to client, repeat."""
        while True:
            stats = await loop.run_in_executor(None, trainer.train_step)
            rnd   = await loop.run_in_executor(None, trainer.sample_round)
            await safe_send(_round_msg(rnd))
            await safe_send(_stats_msg(stats))
            await asyncio.sleep(0.08)   # ≈ 12 visible rounds / sec max

    try:
        while True:
            # Short timeout so we keep serving the evolve loop while still
            # being able to receive incoming commands promptly.
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            data = json.loads(raw)
            cmd  = data.get("cmd", "")

            # ── whisper ───────────────────────────────────────────────────
            if cmd == "whisper":
                stats = await loop.run_in_executor(None, trainer.train_step)
                rnd   = await loop.run_in_executor(None, trainer.sample_round)
                await safe_send(_round_msg(rnd))
                await safe_send(_stats_msg(stats))

            # ── evolve ────────────────────────────────────────────────────
            elif cmd == "evolve":
                on = bool(data.get("on", False))
                if on and evolve_task is None:
                    evolve_task = asyncio.create_task(_evolve_loop())
                elif not on and evolve_task is not None:
                    evolve_task.cancel()
                    evolve_task = None

            # ── skip ──────────────────────────────────────────────────────
            elif cmd == "skip":
                n = int(data.get("n", 25))
                stats = await loop.run_in_executor(None, lambda: trainer.skip(n))
                await safe_send(_stats_msg(stats))

            # ── judge_start ───────────────────────────────────────────────
            elif cmd == "judge_start":
                if evolve_task is not None:
                    evolve_task.cancel()
                    evolve_task = None
                first_round = await loop.run_in_executor(None, trainer.start_judge)
                await safe_send(first_round)

            # ── judge_answer ──────────────────────────────────────────────
            elif cmd == "judge_answer":
                idx    = int(data.get("index", 0))
                result = await loop.run_in_executor(None, trainer.answer_judge, idx)
                await safe_send(result)
                # Automatically advance to the next round if session not finished
                if not result["finished"]:
                    await asyncio.sleep(0.1)
                    nxt = await loop.run_in_executor(None, trainer.judge_next_round)
                    if nxt:
                        await safe_send(nxt)

            # ── reset ─────────────────────────────────────────────────────
            elif cmd == "reset":
                seed = int(data.get("seed", 42))
                if evolve_task is not None:
                    evolve_task.cancel()
                    evolve_task = None
                await loop.run_in_executor(None, lambda: trainer.reset(seed=seed))
                stats = trainer._make_stats()
                await safe_send(_stats_msg(stats))

    except WebSocketDisconnect:
        log.info("Client disconnected.")
    except Exception as exc:
        log.exception("WebSocket error: %s", exc)
    finally:
        if evolve_task is not None:
            evolve_task.cancel()
        trainer._logfile.close()
