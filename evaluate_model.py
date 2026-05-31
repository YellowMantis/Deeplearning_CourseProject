from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO("runs/detect/runs/covid_lft-3/weights/best.pt")

    results = model.predict(
        source="test",
        save=True,
        save_txt=True,
        conf=0.25,
        project="runs/evaluate",
        name="test_predictions",
    )

    print(f"\nDone. Processed {len(results)} images.")
    print("Images with boxes drawn: runs/evaluate/test_predictions/")
