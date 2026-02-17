# app.py
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
from datetime import datetime, timedelta
from sqlalchemy import func
from db import init_db, SessionLocal, TrainPass, CarEvent, EngineSighting
from tracker import event_queue, ocr_queue, tracker_loop
from ocr_worker import ocr_loop

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup():
    init_db()
    # Kick off background loops
    asyncio.create_task(tracker_loop())
    asyncio.create_task(ocr_loop(ocr_queue, event_queue))

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        msg = await event_queue.get()
        await ws.send_json(msg)

@app.get("/api/summary/daily")
def summary_daily():
    """
    Returns last 14 days with per-direction breakdown:
    [
      {"day":"2026-02-16","trains_EB":3,"trains_WB":2,"cars_EB":210,"cars_WB":175},
      ...
    ]
    """
    db = SessionLocal()
    try:
        start = datetime.utcnow() - timedelta(days=14)
        # Aggregate trains & cars per day and direction
        rows = (db.query(func.date(TrainPass.start_ts).label("day"),
                         TrainPass.direction.label("dir"),
                         func.count(TrainPass.id).label("trains"),
                         func.sum(TrainPass.total_railcars).label("cars"))
                  .filter(TrainPass.start_ts >= start)
                  .group_by(func.date(TrainPass.start_ts), TrainPass.direction)
                  .order_by(func.date(TrainPass.start_ts))
                  .all())

        # reshape to day rows with EB/WB fields
        by_day = {}
        for day, dird, ntr, ncars in rows:
            key = str(day)
            if key not in by_day:
                by_day[key] = {"day": key, "trains_EB":0, "trains_WB":0, "cars_EB":0, "cars_WB":0}
            if dird == "EB":
                by_day[key]["trains_EB"] = int(ntr or 0)
                by_day[key]["cars_EB"] = int(ncars or 0)
            elif dird == "WB":
                by_day[key]["trains_WB"] = int(ntr or 0)
                by_day[key]["cars_WB"] = int(ncars or 0)

        # fill gaps with zeros
        days = sorted(by_day.keys())
        return {"data": [by_day[d] for d in days]}
    finally:
        db.close()

@app.get("/api/trains/recent")
def trains_recent():
    db = SessionLocal()
    try:
        rows = (db.query(TrainPass)
                  .order_by(TrainPass.start_ts.desc())
                  .limit(20).all())
        out = []
        for tp in rows:
            engines = [e.engine_number for e in tp.engines]
            out.append({
                "train_id": tp.train_id,
                "start_ts": tp.start_ts.isoformat(),
                "end_ts": tp.end_ts.isoformat() if tp.end_ts else None,
                "direction": tp.direction,
                "locomotives": tp.total_locomotives,
                "railcars": tp.total_railcars,
                "engine_numbers": engines
            })
        return {"data": out}
    finally:
        db.close()

@app.get("/api/engines/by_direction")
def engines_by_direction():
    db = SessionLocal()
    try:
        start = datetime.utcnow() - timedelta(days=30)
        rows = (db.query(EngineSighting.engine_number, TrainPass.direction, func.count(EngineSighting.id))
                  .join(TrainPass, EngineSighting.train_pass_id == TrainPass.id)
                  .filter(TrainPass.start_ts >= start, TrainPass.direction.in_(["EB","WB"]))
                  .group_by(EngineSighting.engine_number, TrainPass.direction)
                  .order_by(EngineSighting.engine_number, TrainPass.direction)
                  .all())
        data = [{"engine": r[0], "direction": r[1], "count": int(r[2])} for r in rows]
        return {"data": data}
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)