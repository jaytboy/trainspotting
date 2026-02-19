import cv2
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# CONFIG
MODEL_PATH = "best.pt"
LINE_X = 320                   
CLASS_MAP = {0: "locomotive", 1: "railcar"}
CONF = 0.25
IMG_SIZE = 640
PIXELS_PER_FOOT = 12.0
MIN_TRACK_FRAMES = 2

class TrainSession:
    def __init__(self):
        self.last_center_x = {}
        self.track_age = defaultdict(int) 
        self.last_ts = {} 
        self.current_speeds = {} # track_id -> speed_mph

    def clear(self):
        self.last_center_x.clear()
        self.track_age.clear()
        self.last_ts.clear()
        self.current_speeds.clear()

def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: USB camera not found.")
        return

    # Window setup
    cv2.namedWindow("Train Tracker Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Train Tracker Debug", 1024, 768)

    session = TrainSession()
    
    # Tracking generator - use persist=True but don't pass source=0
    # We will pass frames manually
    print("Starting tracking loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking on the frame
        results = model.track(frame, tracker="bytetrack.yaml",
                            conf=CONF, imgsz=IMG_SIZE, persist=True, verbose=False)
        
        # Draw LINE_X
        h, w = frame.shape[:2]
        cv2.line(frame, (LINE_X, 0), (LINE_X, h), (0, 0, 255), 2)
        cv2.putText(frame, f"LINE_X {LINE_X}", (LINE_X + 5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        now = time.time()
        
        # Process results
        for r in results:
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id
            
            if ids is not None:
                ids = ids.cpu().numpy().astype(int)
                
                for (x1, y1, x2, y2), cls_i, tid in zip(xyxy, clss, ids):
                    label = CLASS_MAP.get(cls_i, "unknown")
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    # Update state
                    session.track_age[tid] += 1
                    
                    speed_str = "—"
                    
                    # Check if fully inside frame
                    frame_h, frame_w = frame.shape[:2]
                    is_fully_inside = (x1 > 1) and (x2 < frame_w - 1)
                    
                    # Calculate speed
                    if tid in session.last_ts and is_fully_inside:
                        dt = now - session.last_ts[tid]
                        # prevent divide by zero
                        if dt > 0:
                            if tid in session.last_center_x:
                                prev_cx = session.last_center_x[tid]
                                dx = cx - prev_cx
                                
                                # Speed calc
                                dist_ft = abs(dx) / PIXELS_PER_FOOT
                                fps = dist_ft / dt
                                speed_mph = fps * 0.681818
                                
                                # Filter unlikely speeds for display stability
                                if 0.1 < speed_mph < 150:
                                    session.current_speeds[tid] = speed_mph
                                    
                    # Get smoothed speed for display
                    if tid in session.current_speeds:
                         current_speed = session.current_speeds[tid]
                         speed_str = f"{current_speed:.1f} mph"

                    # Update history
                    session.last_center_x[tid] = cx
                    session.last_ts[tid] = now
                    
                    # VISUALIZATION
                    # Bounding Box
                    color = (0, 255, 0) if label == "locomotive" else (255, 100, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Centroid
                    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 255), -1)
                    
                    # Label: ID | Class | Speed
                    text = f"ID:{tid} {label} {speed_str}"
                    cv2.putText(frame, text, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show frame
        cv2.imshow("Train Tracker Debug", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
