# trainspotting

A real-time train detection and counting application using computer vision and OCR to track locomotives and railcars passing a fixed point.

## Features

- **Real-time Detection**: Uses YOLO object detection to identify locomotives and railcars from a live USB camera feed
- **Direction Tracking**: Automatically determines if trains are traveling EB (eastbound) or WB (westbound)
- **Locomotive OCR**: Extracts locomotive numbers from detected locomotives using Tesseract OCR
- **Database Storage**: Persists all train data, car events, and locomotive readings to SQLite
- **Live Dashboard**: WebSocket-based real-time dashboard showing train counts and engine numbers
- **REST API**: Multiple endpoints for querying historical data and statistics

## Requirements

- Python 3.7+
- USB camera connected to the system
- Tesseract-OCR installed (for Windows: https://github.com/UB-Mannheim/tesseract/wiki)

## Setup

### 1. Clone or Download Project
Clone:
```bash
git clone https://github.com/jaytboy/trainspotting
```
or download then enter project folder.
```bash
cd trainspotter
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract-OCR (Windows)
Download and run the installer from: https://github.com/UB-Mannheim/tesseract/wiki

If installed in a non-default location, update the path in `ocr_worker.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\Tesseract-OCR\tesseract.exe"
```

### 5. Run the Application
```bash
python app.py
```

The server will start at `http://0.0.0.0:8000`

## Usage

### Web Interface
Open your browser and navigate to:
```
http://localhost:8000/
```

### WebSocket (Real-time Updates)
The dashboard connects to the WebSocket endpoint at `/ws` for live train detection events.

### REST API Endpoints

#### Get Daily Summary (Last 14 Days)
```
GET /api/summary/daily
```
Returns daily train and car counts broken down by direction (EB/WB).

#### Get Recent Trains
```
GET /api/trains/recent
```
Returns the last 20 detected trains with details including locomotives, railcars, and detected engine numbers.

#### Get Engines by Direction (30 Days)
```
GET /api/engines/by_direction
```
Returns locomotive sightings grouped by engine number and direction for the past 30 days.

## Project Structure

```
trainspotter/
├── app.py              # FastAPI application & REST endpoints
├── tracker.py          # YOLO-based train/car detection & tracking
├── ocr_worker.py       # Locomotive number extraction (OCR)
├── db.py              # Database models (SQLAlchemy)
├── requirements.txt    # Python dependencies
├── best.pt            # YOLO model weights
├── train_counter.db   # SQLite database (created on first run)
├── static/
│   └── dashboard.js   # Frontend JavaScript
└── templates/
    └── index.html     # Web interface
```

## How It Works

1. **Detection**: The tracker reads frames from a USB camera and uses YOLOv8 (via Ultralytics) to detect locomotives and railcars
2. **Counting**: Objects crossing a vertical line at x=320 pixels are counted and tracked
3. **Direction**: Movement direction is determined by analyzing the x-coordinate change of tracked objects
4. **OCR**: When locomotives are detected, the crop is sent to the OCR worker which extracts the 4-digit locomotive number
5. **Storage**: All events, counts, and locomotive numbers are stored in SQLite
6. **Live Updates**: WebSocket events are pushed to the frontend dashboard in real-time

## Configuration

Key parameters can be adjusted in `tracker.py`:

- `MODEL_PATH`: Path to YOLO model weights (default: `best.pt`)
- `LINE_X`: Vertical count line position in pixels (default: 320)
- `CONF`: YOLO confidence threshold (default: 0.25)
- `IMG_SIZE`: Input image size for YOLO (default: 640)
- `START_FRAMES`: Consecutive frames needed to start a train session (default: 6)
- `END_TIMEOUT_S`: Seconds of inactivity to end a train session (default: 8.0)

## Troubleshooting

### Camera Not Found
Ensure your USB camera is properly connected and recognized by the system. Check the camera index in `tracker.py` (currently set to `0`).

### OCR Not Working
- Verify Tesseract is installed and the path is correctly configured in `ocr_worker.py`
- Ensure the locomotive images have sufficient quality and lighting

### Database Issues
Delete `train_counter.db` to start with a fresh database. It will be recreated on the next run.

## License

[Add your license information here]

## Contact

[Add contact information here]
