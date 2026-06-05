from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")  # small model, good for RTX 2080

    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,           # GPU
        name="covid_lft",
        project="runs",
        patience=10,        # stop early if no improvement for 10 epochs
    )

    print("Training complete. Results saved to runs/covid_lft/")
