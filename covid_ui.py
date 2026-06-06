import threading
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

YOLO_MODEL  = "runs/detect/runs/covid_lft-3/weights/best.pt"
CLASS_MODEL = "project2/best_model.pth"
CLASSES     = ["invalid", "negative", "positive"]
RESULT_COLORS = {"positive": "#e74c3c", "negative": "#2ecc71", "invalid": "#f39c12"}
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── load models once at startup ──────────────────────────────────────────────
detector = YOLO(YOLO_MODEL)

classifier = models.resnet18()
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(CLASS_MODEL, map_location=DEVICE))
classifier.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_pipeline(img_path: str):
    results = detector(img_path, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None

    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    conf = float(best.conf[0])

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pad = 20
    crop = img.crop((max(0, x1 - pad), max(0, y1 - pad),
                     min(w, x2 + pad), min(h, y2 + pad)))
    crop.save("project2/last_crop.jpg")

    import torchvision.transforms.functional as TF
    angles = [0, 90, 180, 270]
    with torch.no_grad():
        tensors = torch.stack([transform(TF.rotate(crop, a)) for a in angles]).to(DEVICE)
        probs = torch.softmax(classifier(tensors), dim=1).mean(0).tolist()

    label = CLASSES[probs.index(max(probs))]
    return crop, label, probs


# ── GUI ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("COVID-19 Rapid Test Analyzer")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")
        self._current_probs = None
        self._current_crop = None
        self._build_ui()
        self._try_enable_dnd()
        self.bind("<Configure>", self._on_window_resize)

    def _build_ui(self):
        # title
        tk.Label(self, text="COVID-19 Test Analyzer", font=("Segoe UI", 18, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=(20, 4))

        # drop zone — stretches horizontally
        self.drop_frame = tk.Frame(self, bg="#313244", height=110,
                                   relief="flat", cursor="hand2")
        self.drop_frame.pack(fill="x", padx=24, pady=8)
        self.drop_frame.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="Drag & drop an image here\nor click to choose a file",
            font=("Segoe UI", 11), bg="#313244", fg="#a6adc8",
            justify="center", cursor="hand2"
        )
        self.drop_label.pack(expand=True)
        self.drop_frame.bind("<Button-1>", lambda e: self._choose_file())
        self.drop_label.bind("<Button-1>", lambda e: self._choose_file())

        # status
        self.status_var = tk.StringVar(value="")
        self.status_lbl = tk.Label(self, textvariable=self.status_var,
                                   font=("Segoe UI", 10), bg="#1e1e2e", fg="#89b4fa")
        self.status_lbl.pack()

        # cropped image display
        self.img_frame = tk.Frame(self, bg="#1e1e2e")
        self.img_frame.pack(pady=4)
        self.img_canvas = tk.Label(self.img_frame, bg="#1e1e2e")
        self.img_canvas.pack()

        # result label
        self.result_var = tk.StringVar(value="")
        self.result_lbl = tk.Label(self, textvariable=self.result_var,
                                   font=("Segoe UI", 22, "bold"),
                                   bg="#1e1e2e", fg="#cdd6f4")
        self.result_lbl.pack(pady=(8, 2))

        # probability bars — canvases stretch to fill window width
        self.bars_frame = tk.Frame(self, bg="#1e1e2e")
        self.bars_frame.pack(fill="x", padx=40, pady=(4, 20))
        self._bar_widgets = {}
        for cls in CLASSES:
            row = tk.Frame(self.bars_frame, bg="#1e1e2e")
            row.pack(fill="x", pady=3)
            lbl = tk.Label(row, text=f"{cls:10s}", width=10, anchor="w",
                           font=("Consolas", 10), bg="#1e1e2e", fg="#cdd6f4")
            lbl.pack(side="left")
            canvas = tk.Canvas(row, height=18, bg="#313244", highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True, padx=(4, 8))
            pct_lbl = tk.Label(row, text="", width=6, anchor="e",
                               font=("Consolas", 10), bg="#1e1e2e", fg="#a6adc8")
            pct_lbl.pack(side="left")
            canvas.bind("<Configure>", lambda e, c=cls: self._redraw_bar(c))
            self._bar_widgets[cls] = (canvas, pct_lbl)

        self._reset_bars()

    def _on_window_resize(self, event):
        if event.widget is self and self._current_crop is not None:
            self._update_image_display()

    def _update_image_display(self):
        if self._current_crop is None:
            return
        max_w = max(200, self.winfo_width() - 80)
        max_h = max(150, self.winfo_height() // 3)
        disp = self._current_crop.copy()
        ratio = min(max_w / disp.width, max_h / disp.height)
        if ratio < 1.0:
            disp = disp.resize((int(disp.width * ratio), int(disp.height * ratio)), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp)
        self.img_canvas.config(image=tk_img)
        self.img_canvas._img_ref = tk_img

    def _reset_bars(self):
        self._current_probs = None
        for cls, (canvas, pct_lbl) in self._bar_widgets.items():
            canvas.delete("all")
            pct_lbl.config(text="")

    def _redraw_bar(self, cls):
        if self._current_probs is None:
            return
        canvas, pct_lbl = self._bar_widgets[cls]
        prob = self._current_probs[CLASSES.index(cls)]
        bar_color = {"invalid": "#fab387", "negative": "#a6e3a1", "positive": "#f38ba8"}
        canvas.delete("all")
        bar_w = canvas.winfo_width()
        width = int(prob * bar_w)
        if width > 0:
            canvas.create_rectangle(0, 0, width, 18, fill=bar_color[cls], outline="")

    def _try_enable_dnd(self):
        try:
            from TkinterDnD2 import DND_FILES
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)
        except Exception as e:
            print(f"Drag and drop failed to enable: {e}")

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        self._run(path)

    def _choose_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")]
        )
        if path:
            self._run(path)

    def _run(self, img_path: str):
        self._reset_bars()
        self._current_crop = None
        self.result_var.set("")
        self.img_canvas.config(image="")
        self.status_var.set("Analyzing…")
        self.update()

        def worker():
            try:
                crop, label, probs = run_pipeline(img_path)
            except Exception as exc:
                self.after(0, lambda: self._show_error(str(exc)))
                return
            self.after(0, lambda: self._show_result(crop, label, probs))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, crop, label, probs):
        self.status_var.set("")

        if crop is None:
            self.result_var.set("No cassette detected")
            self.result_lbl.config(fg="#f38ba8")
            return

        self._current_crop = crop
        self._current_probs = probs
        self._update_image_display()

        color = RESULT_COLORS.get(label, "#cdd6f4")
        self.result_var.set(f"Result: {label.upper()}")
        self.result_lbl.config(fg=color)

        bar_color = {"invalid": "#fab387", "negative": "#a6e3a1", "positive": "#f38ba8"}
        for cls, prob in zip(CLASSES, probs):
            canvas, pct_lbl = self._bar_widgets[cls]
            canvas.delete("all")
            bar_w = canvas.winfo_width()
            width = int(prob * bar_w)
            if width > 0:
                canvas.create_rectangle(0, 0, width, 18,
                                        fill=bar_color[cls], outline="")
            pct_lbl.config(text=f"{prob*100:.1f}%")

    def _show_error(self, msg: str):
        self.status_var.set(f"Error: {msg}")


if __name__ == "__main__":
    try:
        from TkinterDnD2 import TkinterDnD

        class App2(App):
            def __init__(self):
                TkinterDnD.Tk.__init__(self)
                self.title("COVID-19 Rapid Test Analyzer")
                self.resizable(True, True)
                self.configure(bg="#1e1e2e")
                self._current_probs = None
                self._current_crop = None
                self._build_ui()
                self._try_enable_dnd()
                self.bind("<Configure>", self._on_window_resize)

        App2().mainloop()
    except Exception:
        App().mainloop()
