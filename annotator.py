import os
import csv
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

### This annotation tool uses: 
# Python 3.12.12
# pillow 12.0.0

#### User Instructions: 
## Before you run this script:
#   1. Adjust label maps to suit your annotation requirements.
#   2. Define annotation output filename and path using the `anno_file` variable.
## Run this script:
#   1. A window will appear to allow you to choose image folder you wish to annotate.
#   2. You must *enter* into the desired folder, not just select it. 
#   3. Once you are happy, select "OK" to begin annotation session.
## Annotation:
#   1. This is a keystroke annotator. Press a key that's in the label map and it will annotate the image.
#   2. Press left and right arrows on your keyboard to move between images. 
#   3. Current image filename and current annotation are displayed at bottom of screen.
#   4. Change existing annotation simply by annotating the image with the new label.
#   5. Annotations are auto-saved in csv file as you go. 

labels_ana = {
    "a": "AC",
    "c": "Cardiac",
    "e": "Cerebellum",
    "h": "HC",
    "f": "Femur",
    "n": "Lips and nose",
    "w": "Spine (coronal)",
    "s": "Spine (sagittal)",
    "q": "Other",
}

labels_mag = {
    "a": "good_mag",
    "f": "increase_mag",
    "r": "reduce_mag",
}

labels_gain = {
    "a": "good_gain",
    "f": "increase_gain",
    "r": "reduce_gain",
}

labels_centering = {
    "d": "good_centering",
    "f": "move_right",
    "s": "move_left",
    "e": "move_up",
    "c": "move_down",
}

labels_shadow = {
    "s": "shadow",
    "d": "no_shadow",
}

# Define annotation output filename below (the file will automatically store in the same directory as this .py file):
anno_file = "anno_ana.csv"

# Define what labels this annotation session will use: 
LABELS = labels_ana

class Annotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame-wise Annotator")

        self.folder = filedialog.askdirectory(title="Select frames folder")
        if not self.folder:
            root.quit()
            return

        self.images = sorted([
            f for f in os.listdir(self.folder)
            if f.lower().endswith(".jpg")
        ])

        self.index = 0
        self.annotations_file = os.path.join(".", anno_file)
        self.annotations = self.load_existing_annotations()

        self.label = tk.Label(root)
        self.label.pack()

        self.filename_label = tk.Label(root, text="", font=("Arial", 14))
        self.filename_label.pack()

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)

        for key in LABELS:
            self.root.bind(key, self.annotate)

        self.show_image()


    def load_existing_annotations(self):
        annotations = {}
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    annotations[row[0]] = row[1]
        return annotations

    def save_annotation(self, filename, label):
        stem = os.path.splitext(filename)[0]
        self.annotations[stem] = label
        with open(self.annotations_file, "w", newline='') as f:
            writer = csv.writer(f)
            for k, v in self.annotations.items():
                writer.writerow([k, v])

    def show_image(self):
        if self.index >= len(self.images):
            print("Done!")
            self.root.quit()
            return

        filename = self.images[self.index]
        stem = os.path.splitext(filename)[0]
        path = os.path.join(self.folder, filename)
        img = Image.open(path)
        img.thumbnail((900, 900))
        self.tk_img = ImageTk.PhotoImage(img)

        self.label.config(image=self.tk_img)

        ### ADDITION START - allows you to see current annotation of any image
        current_ann = self.annotations.get(stem, "unannotated")
        self.filename_label.config(
            text=f"{filename} | current annotation: {current_ann}  ({self.index+1}/{len(self.images)})"
        )
        ### ADDITON END

    def annotate(self, event):
        key = event.char
        if key in LABELS:
            filename = self.images[self.index]
            self.save_annotation(filename, LABELS[key])
            self.next_image()

    def next_image(self, event=None):
        self.index += 1
        self.show_image()

    def prev_image(self, event=None):
        if self.index > 0:
            self.index -= 1
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = Annotator(root)
    root.mainloop()