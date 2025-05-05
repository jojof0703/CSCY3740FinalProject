import os
import sys
import subprocess
import numpy as np
from PIL import Image
import random
import shutil
from termcolor import colored
from ascii_magic import AsciiArt
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import glob
from config import TRAIN_DIR, VAL_DIR, SPLIT_DIR as base_dir, PROJECT_ROOT

def install_packages():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "termcolor", "ascii_magic"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please run manually:")
        print("pip install termcolor ascii_magic")
        return False

def get_terminal_size():
    """Get terminal size for proper display."""
    size = shutil.get_terminal_size()
    return size.columns, size.lines

def analyze_directory(directory):
    """Analyze a directory and return statistics about the dataset."""
    stats = {
        'real': {'count': 0, 'sizes': [], 'samples': []},
        'fake': {'count': 0, 'sizes': [], 'samples': []}
    }
    
    for label in ['real', 'fake']:
        label_dir = os.path.join(directory, label)
        if not os.path.exists(label_dir):
            print(f"⚠️ Warning: {label_dir} does not exist!")
            continue
            
        files = os.listdir(label_dir)
        stats[label]['count'] = len(files)
        
        # Get sample of images for visualization
        sample_files = random.sample(files, min(3, len(files)))
        for file in sample_files:
            try:
                img_path = os.path.join(label_dir, file)
                stats[label]['samples'].append(img_path)
            except Exception as e:
                print(f"⚠️ Error processing {file}: {str(e)}")
    
    return stats

def display_ascii_image(image_path, width=50):
    """Display an image as ASCII art in the terminal."""
    try:
        my_art = AsciiArt.from_image(image_path)
        print(my_art.to_ascii(columns=width))
    except Exception as e:
        print(f"⚠️ Error displaying image: {str(e)}")

def display_distribution(train_stats, val_stats):
    """Display the distribution of classes in train and validation sets."""
    print("\n" + "="*80)
    print(colored("📊 Dataset Distribution", "cyan", attrs=['bold']))
    print("="*80)
    
    # Training set
    print("\n" + colored("Training Set:", "yellow"))
    print(f"Real images: {colored(train_stats['real']['count'], 'green')}")
    print(f"Fake images: {colored(train_stats['fake']['count'], 'red')}")
    print(f"Total: {colored(train_stats['real']['count'] + train_stats['fake']['count'], 'blue')}")
    
    # Validation set
    print("\n" + colored("Validation Set:", "yellow"))
    print(f"Real images: {colored(val_stats['real']['count'], 'green')}")
    print(f"Fake images: {colored(val_stats['fake']['count'], 'red')}")
    print(f"Total: {colored(val_stats['real']['count'] + val_stats['fake']['count'], 'blue')}")

def display_samples(stats, set_name):
    """Display sample images from each class."""
    print("\n" + "="*80)
    print(colored(f"🖼️ Sample Images from {set_name} Set", "cyan", attrs=['bold']))
    print("="*80)
    
    for label in ['real', 'fake']:
        print(f"\n{colored(label.upper(), 'yellow' if label == 'real' else 'red')} samples:")
        for img_path in stats[label]['samples']:
            display_ascii_image(img_path)
            print("-"*80)

def analyze_video_frames(video_path):
    frames = glob.glob(os.path.join(video_path, "*.jpg"))
    if not frames:
        return None
        
    # Sample a few frames for analysis
    sample_frames = np.random.choice(frames, min(5, len(frames)), replace=False)
    
    stats = {
        'sharpness': [],
        'brightness': [],
        'contrast': [],
        'color_diff': []
    }
    
    for frame_path in sample_frames:
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate brightness and contrast
        brightness = np.mean(gray)
        contrast = gray.std()
        
        # Calculate color differences
        R, G, B = cv2.split(img)
        color_diff = np.mean([np.abs(R-G).mean(), np.abs(R-B).mean(), np.abs(G-B).mean()])
        
        stats['sharpness'].append(sharpness)
        stats['brightness'].append(brightness)
        stats['contrast'].append(contrast)
        stats['color_diff'].append(color_diff)
    
    return {k: np.mean(v) for k, v in stats.items()}

def main():
    # Try to import required packages
    try:
        from termcolor import colored
        from ascii_magic import AsciiArt
    except ImportError:
        if not install_packages():
            print("\n⚠️ Please install the required packages and run the script again.")
            return
        
        try:
            from termcolor import colored
            from ascii_magic import AsciiArt
        except ImportError:
            print("\n❌ Failed to import packages even after installation.")
            print("Please try installing manually: pip install termcolor ascii_magic")
            return

    print(colored("🔍 Analyzing dataset...", "cyan", attrs=['bold']))
    print("="*80)
    
    train_stats = analyze_directory(TRAIN_DIR)
    val_stats = analyze_directory(VAL_DIR)
    
    # Display distribution
    display_distribution(train_stats, val_stats)
    
    # Display samples
    display_samples(train_stats, "Training")
    display_samples(val_stats, "Validation")
    
    print("\n" + colored("✅ Analysis complete!", "green", attrs=['bold']))

    base_dir = r"C:\dfd2\split"
    categories = ['train', 'val']
    labels = ['real', 'fake']
    
    results = defaultdict(lambda: defaultdict(list))
    
    for category in categories:
        for label in labels:
            dir_path = os.path.join(base_dir, category, label)
            videos = os.listdir(dir_path)
            
            print(f"\nAnalyzing {category}/{label} videos...")
            for video in videos:
                video_path = os.path.join(dir_path, video)
                stats = analyze_video_frames(video_path)
                if stats:
                    for key, value in stats.items():
                        results[category][f"{label}_{key}"].append(value)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    for category in categories:
        print(f"\n{category.upper()} SET:")
        for label in labels:
            for metric in ['sharpness', 'brightness', 'contrast', 'color_diff']:
                key = f"{label}_{metric}"
                values = results[category][key]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    print(f"{label} {metric}: {mean:.2f} ± {std:.2f}")
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(['sharpness', 'brightness', 'contrast', 'color_diff']):
        plt.subplot(2, 2, i+1)
        for label in labels:
            train_values = results['train'][f"{label}_{metric}"]
            val_values = results['val'][f"{label}_{metric}"]
            
            plt.hist(train_values, alpha=0.5, label=f"Train {label}")
            plt.hist(val_values, alpha=0.5, label=f"Val {label}")
            
        plt.title(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "dataset_analysis.png"))
    print("\nAnalysis plot saved to dataset_analysis.png")

if __name__ == "__main__":
    main() 