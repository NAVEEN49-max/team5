"""FastAPI server: runs the simulation in the background and serves the React dashboard.
Run:  uvicorn main:app --reload --port 8000   ->  http://localhost:8000
"""
import asyncio, contextlib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from engine import World, NODES

state = {"w": World(), "speed": 4, "paused": False}      # speed = simulated seconds per 0.25 s frame


async def loop():
    while True:
        if not state["paused"]:
            for _ in range(state["speed"]): state["w"].step()
        await asyncio.sleep(0.25)


@contextlib.asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="Adaptive Intelligent Traffic Network", lifespan=lifespan)


class Incident(BaseModel):          # also the hook for a video accident detector: POST {u,v,closed:true}
    u: int
    v: int
    closed: bool = True


@app.get("/api/state")
def get_state(): return {**state["w"].view(), "speed": state["speed"], "paused": state["paused"]}


@app.post("/api/scenario/{name}")
def scenario(name: str):
    if name not in ("normal", "surge", "closure", "emergency", "multiple"): raise HTTPException(404, "unknown scenario")
    state["w"].apply(name); return {"ok": True}


@app.post("/api/reset")
def reset(): state["w"] = World(); return {"ok": True}


@app.post("/api/speed/{n}")
def speed(n: int): state["speed"] = max(1, min(n, 40)); return {"speed": state["speed"]}


@app.post("/api/pause")
def pause(): state["paused"] = not state["paused"]; return {"paused": state["paused"]}


@app.post("/api/incident")
def incident(i: Incident):
    w = state["w"]
    if (i.u, i.v) not in w.a.E: raise HTTPException(400, "no such road")
    w.set_closed(i.u, i.v, i.closed); return {"ok": True}


@app.get("/api/route")
def route(src: int, dst: int):
    if not (0 <= src < NODES and 0 <= dst < NODES): raise HTTPException(400, "bad node")
    a, b = state["w"].a, state["w"].b
    pa, ca = a.route(src, dst, True); pb, cb = b.route(src, dst, False)
    return {"adaptive": {"path": pa, "eta": round(ca) if pa else None}, "static": {"path": pb, "eta": round(cb) if pb else None}}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="ui")
