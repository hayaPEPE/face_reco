import tkinter as tk
from tkinter import filedialog
import cv2
import os
from glob import glob
from tkinter import ttk
import dlib

def browse_folder(entry):
    folder_selected = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder_selected)

def detect_faces(input_folder, output_folder, progress, width, height):
    detector = dlib.get_frontal_face_detector()
    image_files = glob(os.path.join(input_folder, '*.*'))

    total_files = len(image_files)
    progress['maximum'] = total_files
    progress['value'] = 0

    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for k, d in enumerate(faces):
            x, y, w, h = d.left(), d.top(), d.width(), d.height()
            face_img = img[y:y + h, x:x + w]
            resized_face = cv2.resize(face_img, (width, height))
            output_file = os.path.join(output_folder, f'face_{i}_{x}_{y}.jpg')
            cv2.imwrite(output_file, resized_face)

        progress['value'] = i + 1
        root.update()

def main():
    global root
    root = tk.Tk()
    root.title("顔抽出アプリ")

    input_label = tk.Label(root, text="画像フォルダ:")
    input_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    input_entry = tk.Entry(root, width=40)
    input_entry.grid(row=0, column=1, padx=5, pady=5)

    input_button = tk.Button(root, text="参照", command=lambda: browse_folder(input_entry))
    input_button.grid(row=0, column=2, padx=5, pady=5)

    output_label = tk.Label(root, text="出力フォルダ:")
    output_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

    output_entry = tk.Entry(root, width=40)
    output_entry.grid(row=1, column=1, padx=5, pady=5)

    output_button = tk.Button(root, text="参照", command=lambda: browse_folder(output_entry))
    output_button.grid(row=1, column=2, padx=5, pady=5)

    size_label = tk.Label(root, text="抽出画像サイズ(幅x高さ):")
    size_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

    size_entry = tk.Entry(root, width=40)
    size_entry.insert(0, "200x200")
    size_entry.grid(row=2, column=1, padx=5, pady=5)

    run_button = tk.Button(root, text="実行", command=lambda: detect_faces(input_entry.get(), output_entry.get(), progress, *map(int, size_entry.get().split('x'))))
    run_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()