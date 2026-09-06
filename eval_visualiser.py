import os
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

# Paths
csv_path = "/path/to/csv/file/storing/predicted/results.csv"
image_dir = "/path/to/directory/containing/all/test/set/images"

# Load CSV
df = pd.read_csv(csv_path)

# Filtering (Optional)
df = df[
    (df["gnd_ana"] == 0) &
    (df["gnd_mag"] == 0)
].reset_index(drop=True)

# Tkinter setup
root = tk.Tk()
root.title("Model Evaluation Visualiser")

# Global index
index = 0

# UI elements
img_label = tk.Label(root)
img_label.pack()

text_label = tk.Label(root, font=("Arial", 14))
text_label.pack()

def load_image(idx):
    row = df.iloc[idx]

    img_path = os.path.join(image_dir, row["frame"]) # + ".jpg"
    img = Image.open(img_path)

    # # Resize for display (optional)
    # img = img.resize((400, 400))

    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    text = (
        f"{row['frame']}\n"
        f"Pred: {row['pred_ana']}, {row['pred_mag']}, {row['pred_gain']}, {row['pred_centering']}, {row['pred_shadow']}\n"
        f"GND: {row['gnd_ana']}, {row['gnd_mag']}, {row['gnd_gain']}, {row['gnd_centering']}, {row['gnd_shadow']}\n"
        f"{idx+1}/{len(df)}"
    )

    text_label.config(text=text)

def next_image(event=None):
    global index
    if index < len(df) - 1:
        index += 1
        load_image(index)

def prev_image(event=None):
    global index
    if index > 0:
        index -= 1
        load_image(index)

# Key bindings
root.bind("<Right>", next_image)
root.bind("<Left>", prev_image)

# Load first image
if len(df) > 0:
    load_image(index)
else:
    text_label.config(text="No images found.")

# Run app
root.mainloop()