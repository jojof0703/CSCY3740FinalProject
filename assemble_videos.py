import os
import cv2
import subprocess
from pathlib import Path
from tkinter import filedialog, Tk
from datetime import datetime

FPS = 25

def choose_folder(title):
    root = Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path

def is_image(file_path):
    img = cv2.imread(str(file_path))
    return img is not None

def get_unique_output_path(output_dir, base_name):
    """Generate a unique output path by adding a timestamp if the file exists."""
    base_name = os.path.splitext(base_name)[0]  # Remove extension
    output_path = os.path.join(output_dir, f"{base_name}.mp4")
    
    # If file exists, add timestamp
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.mp4")
    
    return output_path

def frames_to_video(frame_folder, audio_file, output_video):
    files = sorted(Path(frame_folder).iterdir())
    frame_list = [f for f in files if is_image(f)]
    if not frame_list:
        print(f"❌ No valid frames in {frame_folder}")
        return

    first_frame = cv2.imread(str(frame_list[0]))
    height, width, _ = first_frame.shape

    temp_video = str(Path(output_video).with_suffix(".temp.mp4"))
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    for frame_path in frame_list:
        img = cv2.imread(str(frame_path))
        out.write(img)
    out.release()

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", audio_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_video)
    print(f"✅ Created: {output_video}")

def find_matching_wav(folder_name, audio_root):
    for wav in os.listdir(audio_root):
        if wav.lower().startswith(folder_name.lower()) and wav.lower().endswith(".wav"):
            return os.path.join(audio_root, wav)
    return None

def run_batch_conversion():
    frames_root = choose_folder("📁 Select FRAME folder (each subfolder = 1 video)")
    audio_root  = choose_folder("🎵 Select AUDIO folder (contains .wav files)")
    output_dir  = filedialog.askdirectory(title="💾 Select OUTPUT folder for .mp4 videos")

    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(frames_root):
        frame_path = os.path.join(frames_root, subdir)
        if not os.path.isdir(frame_path):
            continue

        wav_path = find_matching_wav(subdir, audio_root)
        if not wav_path:
            print(f"⚠️ Skipping {subdir}: No matching WAV")
            continue

        # Generate unique output path
        output_file = get_unique_output_path(output_dir, f"real_{subdir}")
        frames_to_video(frame_path, wav_path, output_file)

if __name__ == "__main__":
    run_batch_conversion()
