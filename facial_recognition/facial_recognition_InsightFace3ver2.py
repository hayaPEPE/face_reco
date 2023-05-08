import tkinter as tk
from tkinter import filedialog
import cv2
import os
import os.path
from glob import glob
from tkinter import ttk
#import dlib
import imghdr
import insightface
#import numpy as np

def valid_image_file(filename):
    return imghdr.what(filename) is not None

def browse_folder(entry):
    folder_selected = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder_selected)

def get_output_filename(image_file, output_folder, ext, filename_option, face_count, face_count_local):
    base_filename = os.path.splitext(os.path.basename(image_file))[0]
    if filename_option == "連番":
        face_number_str = str(face_count).zfill(3)  # 連番を001, 002, 003...のような形式に変更
        return os.path.join(output_folder, f'{face_number_str}{ext}')
    elif filename_option == "元のファイル名と同じ":
        return os.path.join(output_folder, f'{base_filename}_{face_count_local}{ext}')
    else:  # "元のファイル名_crop" の場合
        return os.path.join(output_folder, f'{base_filename}_{face_count_local}_crop{ext}')

def add_padding(image, color=(0, 0, 0)):
    h, w = image.shape[:2]
    if h == w:
        return image
    padding_size = abs(h - w) // 2

    if h > w:
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT, value=color)
    else:
        padded_image = cv2.copyMakeBorder(image, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return padded_image


def detect_faces(input_folder, output_folder, progress, width, height, margin=0.3, filename_option="連番", output_mode="カラー"):
    # InsightFaceモデルの読み込み
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=-1, det_size=(640, 640))

    image_files = glob(os.path.join(input_folder, '*.*'))

    total_files = len(image_files)
    progress['maximum'] = total_files
    progress['value'] = 0

    face_count = 1  # 顔の連番カウントを1から始める
    face_count_local = 1
    skipped_count = 0  # スキップされた画像のカウント

    for i, image_file in enumerate(image_files):
        if not valid_image_file(image_file):
            print(f"Skipping invalid image file: {image_file}")
            skipped_count += 1
            progress['maximum'] = total_files - skipped_count
            continue  # 画像ファイルが無効な場合はスキップ

        img = cv2.imread(image_file)
        if img is None:
            print(f"Skipping unreadable image file: {image_file}")
            skipped_count += 1
            progress['maximum'] = total_files - skipped_count
            continue  # 画像が読み込めない場合はスキップ

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = model.get(img)

        if faces is None:
            continue

        for face in faces:
            bbox = face.bbox.astype(int).flatten()
            x, y, x_max, y_max = bbox

            w = x_max - x
            h = y_max - y

            x_min = max(x - int(w * margin), 0)
            y_min = max(y - int(h * margin), 0)
            x_max = min(x + w + int(w * margin), img.shape[1])
            y_max = min(y + h + int(h * margin), img.shape[0])

            face_img = img[y_min:y_max, x_min:x_max]
            face_img_padded = add_padding(face_img)
            resized_face = cv2.resize(face_img_padded, (width, height))

            ext = os.path.splitext(image_file)[1]
            output_file = get_output_filename(image_file, output_folder, ext, filename_option, face_count, face_count_local)

            if output_mode == "モノクロ":
                resized_face_gray = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(output_file, resized_face_gray)
            else:
                cv2.imwrite(output_file, resized_face)

            face_count += 1
            face_count_local += 1

        face_count_local += 1
        progress['value'] = i + 1 - skipped_count
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

    width_label = tk.Label(root, text="幅:")
    width_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

    width_entry = tk.Entry(root, width=10)
    width_entry.insert(0, "400")
    width_entry.grid(row=2, column=1, padx=5, pady=5)

    height_label = tk.Label(root, text="高さ:")
    height_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

    height_entry = tk.Entry(root, width=10)
    height_entry.insert(0, "400")
    height_entry.grid(row=3, column=1, padx=5, pady=5)

    margin_label = tk.Label(root, text="マージン:")
    margin_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)

    margin_entry = tk.Entry(root, width=10)
    margin_entry.insert(0, "0.3")
    margin_entry.grid(row=4, column=1, padx=5, pady=5)

    filename_option_label = tk.Label(root, text="ファイル名のオプション:")
    filename_option_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)

    filename_option_combobox = ttk.Combobox(root, values=["連番", "元のファイル名と同じ", "元のファイル名_crop"])
    filename_option_combobox.set("連番")
    filename_option_combobox.grid(row=5, column=1, padx=5, pady=5)

    output_mode_label = tk.Label(root, text="出力モード:")
    output_mode_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)

    output_mode_combobox = ttk.Combobox(root, values=["カラー", "モノクロ"])
    output_mode_combobox.set("カラー")
    output_mode_combobox.grid(row=6, column=1, padx=5, pady=5)

    start_button = tk.Button(root, text="開始", command=lambda: detect_faces(input_entry.get(), output_entry.get(), progress, int(width_entry.get()), int(height_entry.get()), float(margin_entry.get()), filename_option_combobox.get(), output_mode_combobox.get()))
    start_button.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

    progress = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
    progress.grid(row=8, column=0, columnspan=3, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()