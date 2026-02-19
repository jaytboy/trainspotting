import asyncio, time
from collections import defaultdict, deque
from datetime import datetime, timezone
from db import SessionLocal, TrainPass, CarEvent
from typing import Optional

# Globals shared with app.py
event_queue = asyncio.Queue()  # real-time events → dashboard
ocr_queue = asyncio.Queue()    # locomotive crops → OCR worker

# CONFIG
MODEL_PATH = "best.pt"
LINE_X = 320                   # vertical count line in pixels (imgsz width assumed 640)
CLASS_MAP = {0: "locomotive", 1: "railcar"}
CONF = 0.25
IMG_SIZE = 640
START_FRAMES = 6               # frames with detections to call "train started"
END_TIMEOUT_S = 8.0
MIN_TRACK_FRAMES = 2           # require ≥2 frames before counting a track
PIXELS_PER_FOOT = 12.0

# EB/WB mapping for a vertical count line at x = LINE_X
# dx = cx - prev_cx : positive means left->right
def lr_to_compass(dx: float) -> str:
    return "EB" if dx > 0 else "WB"

class TrainSession:
    def __init__(self):
        self.active = False
        self.train_id: Optional[str] = None
        self.counts = defaultdict(int)
        self.counted_ids = set()
        self.last_center_x = {}
        self.track_age = defaultdict(int)   # track_id -> frames seen
        self.last_detection_time = 0.0
        self.start_buffer = deque(maxlen=START_FRAMES)
        self.dx_buffer = deque(maxlen=30)   # collect dx for direction estimate
        self.speeds = [] # list of calculated speeds (mph)
        self.last_ts = {} # track_id -> last timestamp

    def start(self):
        self.active = True
        self.train_id = f"TP_{time.strftime('%Y%m%d_%H%M%S')}"
        self.counts.clear()
        self.counted_ids.clear()
        self.last_center_x.clear()
        self.track_age.clear()
        self.start_buffer.clear()
        self.dx_buffer.clear()
        self.speeds.clear()
        self.last_ts.clear()
        self.last_detection_time = time.time()

    def maybe_end(self):
        return self.active and (time.time() - self.last_detection_time) > END_TIMEOUT_S

    def direction(self) -> Optional[str]:
        if len(self.dx_buffer) < 5:
            return None
        import numpy as np
        avg = float(np.mean(self.dx_buffer))
        return lr_to_compass(avg)

train = TrainSession()

async def tracker_loop():
    # Give the server a moment to start up before we block the loop with heavy imports/init
    await asyncio.sleep(1.0)
    
    # Offload heavy imports to thread to avoid blocking loop just in case
    def load_heavy_imports():
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import select_device
        return cv2, YOLO, select_device

    cv2, YOLO, select_device = await asyncio.to_thread(load_heavy_imports)
    print("OpenCV loaded and ready to capture video") # User requested this specific message for OpenCV load/ready

    global train
    # Select device (GPU/NPU/CPU)
    device = select_device('')
    print(f"Using device: {device}")

    # Load model in thread with device
    model = await asyncio.to_thread(YOLO, MODEL_PATH)
    model.to(device)
    
    cap = 0  # USB camera index
    last_frame = None

    # Use stream=True for generator; but we also need the raw frame for OCR crops.
    # We'll open a parallel OpenCV capture for frames.
    # Open camera in thread
    cam = await asyncio.to_thread(cv2.VideoCapture, cap)
    
    if not cam.isOpened():
        raise RuntimeError("USB camera not found.")
    
    print("Camera loaded") # User requested this output

    # Warm-up read to know resolution; we’ll resize to IMG_SIZE width downstream if needed.
    ret, frame = await asyncio.to_thread(cam.read)
    if not ret:
        raise RuntimeError("Failed to grab frame from camera.")
    h0, w0 = frame.shape[:2]

    # Kick off model tracking generator (uses its own capture on source=0)
    # gen = model.track(source=cap, tracker="bytetrack.yaml",
    #                   conf=CONF, imgsz=IMG_SIZE, persist=True, stream=True)

    db = SessionLocal()

    try:
        while True:
            # Read frame explicitly in thread
            ret, raw = await asyncio.to_thread(cam.read)
            if not ret:
                # If camera fails, wait a bit and try again or break
                await asyncio.sleep(0.1)
                continue

            # Run inference in thread
            results = await asyncio.to_thread(model.track, raw, tracker="bytetrack.yaml",
                                conf=CONF, imgsz=IMG_SIZE, persist=True, verbose=False,
                                device=model.device)

            for r in results:
                now = time.time()

            has_boxes = r.boxes is not None and len(r.boxes) > 0

            # Train start logic: require a few consecutive "has detections"
            train.start_buffer.append(1 if has_boxes else 0)
            if not train.active and sum(train.start_buffer) >= START_FRAMES:
                train.start()
                # DB row
                tp = TrainPass(train_id=train.train_id, start_ts=datetime.now(timezone.utc))
                db.add(tp); db.commit()
                await event_queue.put({"event": "train_start", "train_id": train.train_id, "ts": now})

            if has_boxes:
                train.last_detection_time = now
                boxes = r.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id
                if ids is None:
                    continue
                ids = ids.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), cls_i, tid in zip(xyxy, clss, ids):
                    label = CLASS_MAP.get(cls_i, "unknown")
                    cx = (x1 + x2) / 2.0

                    # Track age
                    train.track_age[tid] += 1

                    # Keep dx history to estimate train's overall direction later
                    if tid in train.last_center_x:
                        dx = cx - train.last_center_x[tid]
                        train.dx_buffer.append(dx)

                    # Crossing check
                    if tid in train.last_center_x and train.track_age[tid] >= MIN_TRACK_FRAMES:
                        prev_cx = train.last_center_x[tid]
                        crossed = (prev_cx < LINE_X <= cx) or (prev_cx > LINE_X >= cx)
                        if crossed and tid not in train.counted_ids and label in ("locomotive", "railcar"):
                            train.counted_ids.add(tid)
                            train.counts[label] += 1

                            # Per-object direction (EB/WB)
                            dx = cx - prev_cx
                            direction_compass = lr_to_compass(dx)

                            # Ensure train is active before logging event
                            if not train.active:
                                # Logic duplicated from START_FRAMES check, but inline here to save the event
                                train.start()
                                tp = TrainPass(train_id=train.train_id, start_ts=datetime.now(timezone.utc))
                                db.add(tp); db.commit()
                                await event_queue.put({"event": "train_start", "train_id": train.train_id, "ts": now})

                            # Persist CarEvent with EB/WB
                            tp = db.query(TrainPass).filter_by(train_id=train.train_id).one()
                            ev = CarEvent(train_pass_id=tp.id, track_id=int(tid),
                                        klass=label, direction=direction_compass)
                            db.add(ev); db.commit()

                            # Calculate speed
                            speed_mph = 0.0
                            
                            # Only calculate speed if object is fully inside the frame
                            frame_h, frame_w = raw.shape[:2]
                            is_fully_inside = (x1 > 1) and (x2 < frame_w - 1)

                            if tid in train.last_ts and is_fully_inside:
                                dt = now - train.last_ts[tid]
                                if dt > 0:
                                    # distance in feet = dx (pixels) / PIXELS_PER_FOOT
                                    # speed in fps = (dx / PIXELS_PER_FOOT) / dt
                                    # speed in mph = fps * 0.681818
                                    # Use absolute dx for speed magnitude
                                    dist_ft = abs(dx) / PIXELS_PER_FOOT
                                    fps = dist_ft / dt
                                    speed_mph = fps * 0.681818
                                    if speed_mph > 0.1 and speed_mph < 150: # valid range filter
                                        train.speeds.append(speed_mph)

                            # Queue loco crop for OCR
                            if label == "locomotive":
                                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                                x1i = max(0, x1i); y1i = max(0, y1i)
                                x2i = min(raw.shape[1], x2i); y2i = min(raw.shape[0], y2i)
                                crop = raw[y1i:y2i, x1i:x2i].copy()
                                await ocr_queue.put({
                                    "train_id": train.train_id,
                                    "track_id": int(tid),
                                    "image": crop
                                })

                            # Calculate current average speed for display
                            avg_speed = 0.0
                            if train.speeds:
                                avg_speed = sum(train.speeds) / len(train.speeds)

                            # Live update event (now includes EB/WB)
                            await event_queue.put({
                                "event": "count",
                                "train_id": train.train_id,
                                "track_id": int(tid),
                                "class": label,
                                "direction": direction_compass,     # EB / WB
                                "speed_mph": round(speed_mph, 1),
                                "avg_speed_mph": round(avg_speed, 1),
                                "totals": dict(train.counts),
                                "ts": time.time()
                            })

                    train.last_center_x[tid] = cx
                    train.last_ts[tid] = now

            # Train end?
            if train.maybe_end():
                tp = db.query(TrainPass).filter_by(train_id=train.train_id).one()
                tp.end_ts = datetime.now(timezone.utc)
                tp.direction = train.direction()
                tp.total_locomotives = train.counts.get("locomotive", 0)
                tp.total_railcars = train.counts.get("railcar", 0)
                if train.speeds:
                     tp.avg_speed_mph = sum(train.speeds) / len(train.speeds)
                db.add(tp); db.commit()

                await event_queue.put({
                    "event": "train_end",
                    "train_id": train.train_id,
                    "ts": now,
                    "direction": tp.direction,
                    "final_totals": {
                        "locomotive": tp.total_locomotives,
                        "railcar": tp.total_railcars
                    }
                })
                # reset
                train = TrainSession()  # new instance resets state
    finally:
        db.close()
        cam.release()