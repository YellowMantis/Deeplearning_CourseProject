import shutil
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

CROPS_DIR = Path("project2/crops")
BASE_DIR  = Path("project2")

counts = {"positive": 0, "negative": 0, "invalid": 0}

def dest(label):
    # every 5th image goes to test, rest go to train
    split = "test" if counts[label] % 5 == 4 else "train"
    return BASE_DIR / split / label

images = sorted(CROPS_DIR.glob("*.jpg"))
index  = [0]

root = tk.Tk()
root.title("Label Images — P=Positive  N=Negative  I=Invalid  S=Skip")
root.configure(bg="black")

img_label = tk.Label(root, bg="black")
img_label.pack(pady=10)

status = tk.Label(root, text="", bg="black", fg="white", font=("Arial", 13))
status.pack()

progress = tk.Label(root, text="", bg="black", fg="gray", font=("Arial", 11))
progress.pack(pady=5)

def show(i):
    if i >= len(images):
        status.config(text="All done!", fg="lime")
        img_label.config(image="")
        return
    img = Image.open(images[i])
    img.thumbnail((600, 600))
    photo = ImageTk.PhotoImage(img)
    img_label.config(image=photo)
    img_label.image = photo
    progress.config(text=f"{i+1} / {len(images)}  |  positive={counts['positive']}  negative={counts['negative']}  invalid={counts['invalid']}")
    status.config(text=images[i].name, fg="white")

def move(label):
    i = index[0]
    if i >= len(images):
        return
    dst = dest(label)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.move(str(images[i]), dst / images[i].name)
    counts[label] += 1
    index[0] += 1
    show(index[0])

def on_key(event):
    k = event.keysym.lower()
    if   k == "p": move("positive")
    elif k == "n": move("negative")
    elif k == "i": move("invalid")
    elif k == "s": index[0] += 1; show(index[0])  # skip

root.bind("<Key>", on_key)

btn_frame = tk.Frame(root, bg="black")
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Positive (P)", width=14, bg="#2e7d32", fg="white", font=("Arial",12), command=lambda: move("positive")).grid(row=0, column=0, padx=8)
tk.Button(btn_frame, text="Negative (N)", width=14, bg="#1565c0", fg="white", font=("Arial",12), command=lambda: move("negative")).grid(row=0, column=1, padx=8)
tk.Button(btn_frame, text="Invalid (I)",  width=14, bg="#b71c1c", fg="white", font=("Arial",12), command=lambda: move("invalid")).grid(row=0, column=2, padx=8)

show(0)
root.mainloop()
