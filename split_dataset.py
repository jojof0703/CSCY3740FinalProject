import os
import shutil
import random
from glob import glob
from config import DATASET_DIR as SOURCE_DIR, SPLIT_DIR as DEST_DIR

# Paths
SPLIT_RATIO = 0.8  # 80% train, 20% val

def get_video_frames(video_path):
    """Get all frames from a video folder."""
    frames = glob(os.path.join(video_path, "*.jpg"))  # Adjust extension if needed
    return frames

def split_class(label):
    src_path = os.path.join(SOURCE_DIR, label)
    video_folders = os.listdir(src_path)
    random.shuffle(video_folders)  # Shuffle videos, not frames
    
    # Calculate split index based on number of videos
    split_index = int(len(video_folders) * SPLIT_RATIO)
    train_videos = video_folders[:split_index]
    val_videos = video_folders[split_index:]
    
    print(f"\n📊 {label.upper()} Split Statistics:")
    print(f"Total videos: {len(video_folders)}")
    print(f"Training videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    
    # Process training videos
    print(f"\n📁 Processing training videos for {label}...")
    for video_name in train_videos:
        src = os.path.join(src_path, video_name)
        dst = os.path.join(DEST_DIR, "train", label, video_name)
        shutil.copytree(src, dst)
        frames = get_video_frames(src)
        print(f"✅ Copied {label}/{video_name} ({len(frames)} frames) to train/")
    
    # Process validation videos
    print(f"\n📁 Processing validation videos for {label}...")
    for video_name in val_videos:
        src = os.path.join(src_path, video_name)
        dst = os.path.join(DEST_DIR, "val", label, video_name)
        shutil.copytree(src, dst)
        frames = get_video_frames(src)
        print(f"✅ Copied {label}/{video_name} ({len(frames)} frames) to val/")

def main():
    # Create base directories
    for split_type in ["train", "val"]:
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(DEST_DIR, split_type, label), exist_ok=True)

    # Split each class
    split_class("real")
    split_class("fake")
    
    # Print final statistics
    print("\n📊 Final Dataset Statistics:")
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            path = os.path.join(DEST_DIR, split, label)
            videos = os.listdir(path)
            total_frames = sum(len(get_video_frames(os.path.join(path, v))) for v in videos)
            print(f"{split}/{label}: {len(videos)} videos, {total_frames} frames")
    
    print("\n🎉 Done splitting dataset!")

if __name__ == "__main__":
    main()
