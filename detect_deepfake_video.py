import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, Input
import glob
from datetime import datetime
from tqdm import tqdm
import face_recognition
from config import MODEL_WEIGHTS as WEIGHTS_PATH, OUTPUT_DIR

# === Paths ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Feature extraction ===
def extract_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Sharpness metrics
    sharpness = laplacian.var()
    edge_strength = np.mean(np.abs(laplacian))
    local_sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    
    # Brightness and contrast
    brightness = np.mean(img_gray)
    contrast = img_gray.std()
    
    # Color metrics
    R, G, B = cv2.split(img)
    rg_diff = np.mean(np.abs(R - G))
    rb_diff = np.mean(np.abs(R - B))
    gb_diff = np.mean(np.abs(G - B))
    
    # Color consistency
    color_std = np.std([R.mean(), G.mean(), B.mean()])
    color_range = np.max([R.max(), G.max(), B.max()]) - np.min([R.min(), G.min(), B.min()])
    
    # Texture analysis
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    return np.array([
        sharpness, edge_strength, local_sharpness,
        brightness, contrast,
        rg_diff, rb_diff, gb_diff,
        color_std, color_range,
        texture_strength
    ], dtype=np.float32)

# === Model Architecture ===
def build_model():
    # Image input
    img_input = tf.keras.Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
    base_model.trainable = False

    x = base_model(img_input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Feature input
    feature_input = tf.keras.Input(shape=(11,))
    feature_dense = tf.keras.layers.Dense(64, activation='relu')(feature_input)
    feature_dense = tf.keras.layers.BatchNormalization()(feature_dense)
    feature_dense = tf.keras.layers.Dropout(0.2)(feature_dense)

    # Combine paths
    combined = tf.keras.layers.Concatenate()([x, feature_dense])

    # Enhanced dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Add attention mechanism
    attention = tf.keras.layers.Dense(128, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Single output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[img_input, feature_input], outputs=output)

# === File Picker ===
def select_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def get_output_path(input_path):
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{name}_analyzed_{timestamp}{ext}"
    return os.path.join(OUTPUT_DIR, output_name)

def process_frame(frame, final_prediction, final_confidence):
    # Detect faces
    face_locations = face_recognition.face_locations(frame)
    
    # Draw boxes and labels for each face
    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        # Draw box and label
        color = (0, 0, 255) if final_prediction > 0.5 else (0, 255, 0)  # Red for fake, green for real
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Add final verdict and confidence at the top of the frame
        verdict = "Fake" if final_prediction > 0.5 else "Real"
        label = f"{verdict}: {final_confidence*100:.1f}%"
        cv2.putText(frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
        
    predictions = []
    frame_count = 0
    max_frames = 50  # Sample 50 frames
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling interval
    if total_frames < max_frames:
        sampling_interval = 1
    else:
        sampling_interval = total_frames // max_frames
    
    # Create output video
    output_path = get_output_path(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("📊 Processing frames...")
    with tqdm(total=max_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % sampling_interval == 0:
                # Process frame
                img = cv2.resize(frame, (224, 224)) / 255.0
                features = extract_features(frame)
                
                pred = model.predict([np.expand_dims(img, axis=0), 
                                    np.expand_dims(features, axis=0)], verbose=0)
                predictions.append(pred[0][0])
                
                pbar.update(1)
                
                if len(predictions) >= max_frames:
                    break
    
    cap.release()
    
    if not predictions:
        return None, None, None
    
    # Calculate final prediction and confidence
    final_prediction = np.mean(predictions)
    final_confidence = final_prediction
    
    # Reopen video for processing with final results
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with final results
        processed_frame = process_frame(frame, final_prediction, final_confidence)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    
    return final_prediction, np.std(predictions), output_path

def main():
    print("🔍 Loading model...")
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    
    # Let user select a video file
    video_path = select_video()
    if not video_path:
        print("❌ No video file selected.")
        return
        
    print(f"\n🎥 Processing selected video: {os.path.basename(video_path)}")
    
    # Analyze the video
    final_prediction, std_pred, output_path = analyze_video(video_path, model)
    if final_prediction is None:
        print("❌ Error: Could not process video")
        return
    
    print("\n📊 Results:")
    print("=" * 40)
    print(f"Confidence Score: {final_prediction*100:.2f}% ± {std_pred*100:.2f}%")
    print(f"Final Decision: {'Fake' if final_prediction > 0.5 else 'Real'}")
    print(f"Output saved to: {output_path}")
    print("=" * 40)

if __name__ == "__main__":
    main()
