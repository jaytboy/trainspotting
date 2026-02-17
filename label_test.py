from ultralytics import YOLO

model = YOLO("best.pt")  # Load a pretrained model (you can specify the path to your trained model)
model.predict(source=0, save=True, conf=0.25, imgsz=640, device="cpu", show=True, vid_stride=1) # source=0 is a USB Camera
print("Live video labeling. Press Ctrl-C to stop.")
