import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define all paths relative to project root
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "split")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Model weights path
MODEL_WEIGHTS = os.path.join(CHECKPOINTS_DIR, "best_model_weights.h5")

# Training and validation directories
TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val") 