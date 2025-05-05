import os
import cv2
import numpy as np
from pathlib import Path

def analyze_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge_strength = np.mean(cv2.Canny(gray, 100, 200))
    brightness = np.mean(gray)
    contrast = np.std(gray)

    b, g, r = cv2.split(img)
    rg_diff = np.mean(np.abs(r - g))
    rb_diff = np.mean(np.abs(r - b))

    return [sharpness, edge_strength, brightness, contrast, rg_diff, rb_diff]

def collect_features(folder_path):
    features = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = Path(root) / file
                result = analyze_image(img_path)
                if result:
                    features.append(result)

    return np.array(features)

def print_averages(label, data):
    if data.size == 0:
        print(f"\n⚠️ No data found for: {label}")
        return

    avg = np.mean(data, axis=0)
    print(f"\n📊 Averages for {label.upper()}:")
    print(f"   Sharpness      : {avg[0]:.2f}")
    print(f"   Edge Strength  : {avg[1]:.2f}")
    print(f"   Brightness     : {avg[2]:.2f}")
    print(f"   Contrast       : {avg[3]:.2f}")
    print(f"   R-G Diff       : {avg[4]:.2f}")
    print(f"   R-B Diff       : {avg[5]:.2f}")

if __name__ == "__main__":
    base_dir = Path("dataset")

    real_data = collect_features(base_dir / "real")
    fake_data = collect_features(base_dir / "fake")

    print_averages("REAL", real_data)
    print_averages("FAKE", fake_data)
