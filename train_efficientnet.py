import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from datetime import datetime
import os
import numpy as np
import glob
import gc
import cv2
from config import TRAIN_DIR, VAL_DIR, CHECKPOINTS_DIR

# === Paths ===
WEIGHTS_DIR = CHECKPOINTS_DIR
FINAL_WEIGHTS = os.path.join(WEIGHTS_DIR, "efficientnetb0_weights.h5")

# === Data Augmentation ===
def augment_image(img):
    # Random brightness adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    
    # Random contrast adjustment
    contrast_factor = np.random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
    
    # Random Gaussian noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

# === Enhanced Feature Extraction ===
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

# === Memory-efficient video-level metrics ===
class VideoLevelMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_dir):
        super().__init__()
        self.val_dir = val_dir
        self.val_videos = self._get_video_paths(val_dir)
        self.best_val_accuracy = 0.0
        
    def _get_video_paths(self, base_dir):
        videos = []
        for label in ['real', 'fake']:
            label_dir = os.path.join(base_dir, label)
            for video in os.listdir(label_dir):
                videos.append((os.path.join(label_dir, video), 0 if label == 'real' else 1))
        return videos
    
    def on_epoch_end(self, epoch, logs=None):
        correct = 0
        total = len(self.val_videos)
        val_loss = 0
        
        for video_path, true_label in self.val_videos:
            frames = glob.glob(os.path.join(video_path, "*.jpg"))
            if not frames:
                continue
                
            # Sample a subset of frames to save memory
            if len(frames) > 10:
                frames = np.random.choice(frames, 10, replace=False)
                
            # Get predictions for sampled frames
            predictions = []
            for frame_path in frames:
                img = cv2.imread(frame_path)
                img_array = cv2.resize(img, (224, 224)) / 255.0
                features = extract_features(img)
                pred = self.model.predict([np.expand_dims(img_array, axis=0), 
                                        np.expand_dims(features, axis=0)], verbose=0)
                predictions.append(pred[0][0])
                val_loss += self.model.evaluate([np.expand_dims(img_array, axis=0), 
                                               np.expand_dims(features, axis=0)], 
                                              np.array([true_label]), verbose=0)[0]
                
                # Clear memory
                del img, img_array, features
                gc.collect()
            
            # Average predictions for the video
            video_pred = np.mean(predictions)
            pred_label = 1 if video_pred > 0.5 else 0
            
            if pred_label == true_label:
                correct += 1
        
        video_accuracy = correct / total
        avg_val_loss = val_loss / total
        
        # Update best accuracy
        if video_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = video_accuracy
            print(f"\n🎉 New best validation accuracy: {video_accuracy:.4f}")
        
        print(f"\nVideo-level validation accuracy: {video_accuracy:.4f}")
        print(f"Video-level validation loss: {avg_val_loss:.4f}")
        logs['val_loss'] = avg_val_loss
        logs['val_accuracy'] = video_accuracy

# === Learning Rate Schedule ===
def lr_schedule(epoch):
    if epoch < 2:
        return 5e-5
    elif epoch < 4:
        return 2e-5
    else:
        return 1e-5

# === Memory-efficient data generators ===
def get_video_frames(video_path):
    return glob.glob(os.path.join(video_path, "*.jpg"))

def video_generator(directory, batch_size=8, is_validation=False):
    while True:
        videos = []
        for label in ['real', 'fake']:
            label_dir = os.path.join(directory, label)
            for video in os.listdir(label_dir):
                videos.append((os.path.join(label_dir, video), 0 if label == 'real' else 1))
        
        if not is_validation:
            np.random.shuffle(videos)
        
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i+batch_size]
            batch_images = []
            batch_features = []
            batch_labels = []
            
            for video_path, label in batch_videos:
                frames = get_video_frames(video_path)
                if not frames:
                    continue
                    
                # For validation, use a subset of frames
                if is_validation:
                    if len(frames) > 4:  # Increased from 3
                        frames = np.random.choice(frames, 4, replace=False)
                else:
                    # For training, randomly select one frame
                    frames = [np.random.choice(frames)]
                
                for frame_path in frames:
                    img = cv2.imread(frame_path)
                    img_array = cv2.resize(img, (224, 224)) / 255.0
                    features = extract_features(img)
                    
                    batch_images.append(img_array)
                    batch_features.append(features)
                    batch_labels.append(label)
                    
                    # Clear memory
                    del img, img_array, features
                    gc.collect()
            
            if batch_images:
                yield [np.array(batch_images), np.array(batch_features)], np.array(batch_labels)
                # Clear memory
                del batch_images, batch_features, batch_labels
                gc.collect()

# === Enhanced Model Architecture ===
def build_model():
    # Image input
    img_input = Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    base_model.trainable = False

    x = base_model(img_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Feature input with enhanced features
    feature_input = Input(shape=(11,))
    feature_dense = layers.Dense(64, activation='relu')(feature_input)
    feature_dense = layers.BatchNormalization()(feature_dense)
    feature_dense = layers.Dropout(0.2)(feature_dense)

    # Combine paths
    combined = layers.Concatenate()([x, feature_dense])

    # Enhanced dense layers
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Add attention mechanism
    attention = layers.Dense(128, activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Single output
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=[img_input, feature_input], outputs=output)

# === Configure TensorFlow for CPU ===
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

os.makedirs(WEIGHTS_DIR, exist_ok=True)
model = build_model()

# === Callbacks ===
callbacks = [
    LearningRateScheduler(lr_schedule),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(WEIGHTS_DIR, 'best_model_weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    VideoLevelMetrics(VAL_DIR)
]

# === Training ===
print("\n🔁 Starting training...")

# Calculate class weights
train_real_count = len(os.listdir(os.path.join(TRAIN_DIR, 'real')))
train_fake_count = len(os.listdir(os.path.join(TRAIN_DIR, 'fake')))
total = train_real_count + train_fake_count
class_weight = {
    0: total / (2 * train_real_count),  # real
    1: total / (2 * train_fake_count)   # fake
}

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Calculate steps per epoch based on number of videos
train_videos = sum([len(os.listdir(os.path.join(TRAIN_DIR, label))) for label in ['real', 'fake']])
val_videos = sum([len(os.listdir(os.path.join(VAL_DIR, label))) for label in ['real', 'fake']])

# Adjusted batch size and steps
steps_per_epoch = train_videos // 8
validation_steps = val_videos // 8

print(f"\n📊 Training Configuration:")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Batch size: 8")
print(f"Using CPU with 4 threads")
print(f"Initial learning rate: 5e-5")
print(f"Class weights: {class_weight}")
print(f"Using enhanced feature extraction and attention mechanism")

model.fit(
    video_generator(TRAIN_DIR, batch_size=8),
    steps_per_epoch=steps_per_epoch,
    validation_data=video_generator(VAL_DIR, batch_size=8, is_validation=True),
    validation_steps=validation_steps,
    epochs=30,
    class_weight=class_weight,
    callbacks=[
        LearningRateScheduler(lr_schedule),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(WEIGHTS_DIR, 'best_model_weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        VideoLevelMetrics(VAL_DIR)
    ],
    verbose=1
)

# Save final weights
model.save_weights(FINAL_WEIGHTS)
print(f"\n✅ Final weights saved to: {FINAL_WEIGHTS}")
