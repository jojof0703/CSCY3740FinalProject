import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob
from collections import defaultdict
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, Input
from datetime import datetime
import face_recognition
from config import VIDEOS_DIR, MODEL_WEIGHTS, OUTPUT_DIR

# Paths
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    # Process frames without a progress bar
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
    model.load_weights(MODEL_WEIGHTS)
    
    # Get list of test videos
    test_videos = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))
    if not test_videos:
        print("❌ No test videos found in the videos directory.")
        return
    
    # Randomly select 25 real and 25 fake videos
    real_videos = [v for v in test_videos if "real" in os.path.basename(v).lower()]
    fake_videos = [v for v in test_videos if "fake" in os.path.basename(v).lower()]
    
    if len(real_videos) < 25 or len(fake_videos) < 25:
        print("❌ Not enough videos for testing. Need at least 25 real and 25 fake videos.")
        return
    
    selected_real = np.random.choice(real_videos, 25, replace=False)
    selected_fake = np.random.choice(fake_videos, 25, replace=False)
    test_videos = list(selected_real) + list(selected_fake)
    
    print(f"\n🎥 Testing on {len(test_videos)} videos...")
    
    # Run the test 5 times
    all_results = []
    for run in range(5):
        print(f"\n📊 Test Run #{run + 1}")
        print("=" * 40)
        
        # Test each video with a single progress bar for all videos
        results = []
        with tqdm(total=len(test_videos), desc="Processing videos") as pbar:
            for video_path in test_videos:
                final_prediction, std_pred, output_path = analyze_video(video_path, model)
                
                if final_prediction is not None:
                    is_fake = "fake" in os.path.basename(video_path).lower()
                    predicted_fake = final_prediction > 0.5
                    correct = (is_fake and predicted_fake) or (not is_fake and not predicted_fake)
                    
                    results.append((is_fake, predicted_fake, final_prediction))
                pbar.update(1)
        
        # Calculate overall statistics for this run
        if results:
            is_fake, predicted_fake, confidences = zip(*results)
            is_fake = np.array(is_fake)
            predicted_fake = np.array(predicted_fake)
            confidences = np.array(confidences)
            
            accuracy = np.mean(is_fake == predicted_fake)
            real_accuracy = np.mean(predicted_fake[~is_fake] == False)
            fake_accuracy = np.mean(predicted_fake[is_fake] == True)
            
            print("\n📈 Overall Statistics:")
            print("=" * 40)
            print(f"Total Videos: {len(results)}")
            print(f"Overall Accuracy: {accuracy*100:.2f}%")
            print(f"Real Video Accuracy: {real_accuracy*100:.2f}%")
            print(f"Fake Video Accuracy: {fake_accuracy*100:.2f}%")
            print("=" * 40)
            
            all_results.append((accuracy, real_accuracy, fake_accuracy))
    
    # Calculate and display average results across all runs
    if all_results:
        accuracies, real_accuracies, fake_accuracies = zip(*all_results)
        avg_accuracy = np.mean(accuracies)
        avg_real_accuracy = np.mean(real_accuracies)
        avg_fake_accuracy = np.mean(fake_accuracies)
        
        print("\n📈 Average Results Across All Runs:")
        print("=" * 40)
        print(f"Average Overall Accuracy: {avg_accuracy*100:.2f}%")
        print(f"Average Real Video Accuracy: {avg_real_accuracy*100:.2f}%")
        print(f"Average Fake Video Accuracy: {avg_fake_accuracy*100:.2f}%")
        print("=" * 40)

if __name__ == "__main__":
    main() 