import cv2
import os
import glob
from datetime import datetime
from config import VIDEOS_DIR, DATASET_DIR as OUTPUT_DIR

# === Paths ===
VIDEO_DIR = VIDEOS_DIR
OUTPUT_DIR = OUTPUT_DIR

def get_unique_folder_path(base_dir, label, video_name):
    """Generate a unique folder path by adding a timestamp if the folder exists."""
    base_path = os.path.join(base_dir, label, video_name)
    
    # If folder exists, add timestamp
    if os.path.exists(base_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"{base_path}_{timestamp}"
    
    return base_path

def extract_frames(video_path, label):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = get_unique_folder_path(OUTPUT_DIR, label, video_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(save_path, f"{idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    print(f"✅ Extracted {idx} frames from {video_name} to {save_path}")

# === Main Loop ===
if __name__ == "__main__":
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))

    for video_path in video_files:
        name = os.path.basename(video_path).lower()
        if "fake" in name:
            label = "fake"
        elif "real" in name:
            label = "real"
        else:
            print(f"⚠️ Skipping {name} (could not determine if real or fake)")
            continue

        extract_frames(video_path, label)
