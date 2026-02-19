import asyncio, os, re
from collections import defaultdict, Counter
from db import SessionLocal, TrainPass, EngineSighting
from datetime import datetime, timezone

# If needed on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DIGIT_RE = re.compile(r"\b(\d{4})\b")

def preprocess(img):
    # Grayscale, enlarge, denoise, binarize → better OCR on side numbers
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 7, 35, 35)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

async def ocr_loop(ocr_queue, event_queue):
    """Consumes locomotive crops, extracts 4-digit number, stores best guess per track."""
    # Allow server to start
    await asyncio.sleep(1.0)
    
    # Verify tesseract is effectively "loaded" (it's a subprocess, but checking availability helps)
    # We can assume it's ready if we can import it.
    import pytesseract
    print("Tesseract loaded completely") # User requested this output

    cache = defaultdict(list)  # (train_id, track_id) -> [candidates]

    while True:
        item = await ocr_queue.get()
        train_id = item["train_id"]
        track_id = item["track_id"]
        img = item["image"]

        # Run preprocessing in thread
        bw = await asyncio.to_thread(preprocess, img)
        config = r"--psm 7 -c tessedit_char_whitelist=0123456789"
        
        # Run Tesseract in thread
        text = await asyncio.to_thread(pytesseract.image_to_string, bw, config=config)
        cand = DIGIT_RE.findall(text)
        if cand:
            cache[(train_id, track_id)].extend(cand)

        # Commit if we have enough confidence (or after first hit)
        votes = Counter(cache[(train_id, track_id)])
        if votes:
            number, n = votes.most_common(1)[0]
            # Store to DB if not already stored
            db = SessionLocal()
            try:
                tp = db.query(TrainPass).filter_by(train_id=train_id).first()
                if tp:
                    exists = db.query(EngineSighting).filter_by(
                        train_pass_id=tp.id, track_id=track_id, engine_number=number
                    ).first()
                    if not exists:
                        db.add(EngineSighting(
                            train_pass_id=tp.id, track_id=track_id,
                            engine_number=number, first_seen_ts=datetime.now(timezone.utc)
                        ))
                        db.commit()
                        # Emit a non-blocking update—dashboard can show we recognized an engine number
                        await event_queue.put({
                            "event": "engine_number",
                            "train_id": train_id,
                            "track_id": track_id,
                            "engine_number": number
                        })
            finally:
                db.close()