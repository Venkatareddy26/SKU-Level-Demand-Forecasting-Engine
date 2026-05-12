"""
Fine-tune existing banana leaf disease model for better accuracy.
This script loads your existing model and trains it a bit more.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Configuration
MODEL_PATH = 'saved_models/banana_leaf_model.h5'
IMPROVED_MODEL_PATH = 'saved_models/banana_leaf_model_improved.h5'
DATA_DIR = 'data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch for fine-tuning
FINE_TUNE_EPOCHS = 20  # Additional epochs

print("="*60)
print("FINE-TUNING EXISTING MODEL")
print("="*60)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please train the model first using train.py")
    exit(1)

# Load existing model
print(f"\n✓ Loading existing model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("✓ Model loaded successfully!")

# Show current model summary
print("\nCurrent Model Architecture:")
model.summary()

# Enhanced data augmentation for fine-tuning
print("\n✓ Setting up enhanced data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
print("\n✓ Loading training data...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("✓ Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Unfreeze some layers for fine-tuning
print("\n✓ Unfreezing layers for fine-tuning...")
# Unfreeze the last 20 layers
for layer in model.layers[-20:]:
    layer.trainable = True

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Compile with lower learning rate for fine-tuning
print("\n✓ Compiling model with lower learning rate...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Callbacks for better training
callbacks = [
    # Save best model
    ModelCheckpoint(
        IMPROVED_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Stop if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# Evaluate before fine-tuning
print("\n" + "="*60)
print("BEFORE FINE-TUNING")
print("="*60)
before_results = model.evaluate(val_generator, verbose=0)
print(f"Validation Loss: {before_results[0]:.4f}")
print(f"Validation Accuracy: {before_results[1]:.4f}")
if len(before_results) > 2:
    print(f"Validation Precision: {before_results[2]:.4f}")
    print(f"Validation Recall: {before_results[3]:.4f}")

# Fine-tune the model
print("\n" + "="*60)
print("STARTING FINE-TUNING")
print("="*60)
print(f"Training for {FINE_TUNE_EPOCHS} additional epochs...")

history = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluate after fine-tuning
print("\n" + "="*60)
print("AFTER FINE-TUNING")
print("="*60)

# Load best model
best_model = keras.models.load_model(IMPROVED_MODEL_PATH)
after_results = best_model.evaluate(val_generator, verbose=0)

print(f"Validation Loss: {after_results[0]:.4f}")
print(f"Validation Accuracy: {after_results[1]:.4f}")
if len(after_results) > 2:
    print(f"Validation Precision: {after_results[2]:.4f}")
    print(f"Validation Recall: {after_results[3]:.4f}")

# Show improvement
print("\n" + "="*60)
print("IMPROVEMENT SUMMARY")
print("="*60)
acc_improvement = (after_results[1] - before_results[1]) * 100
loss_improvement = (before_results[0] - after_results[0]) * 100

print(f"Accuracy Improvement: {acc_improvement:+.2f}%")
print(f"Loss Improvement: {loss_improvement:+.2f}%")

if acc_improvement > 0:
    print("\n✅ Model improved! Saved to:", IMPROVED_MODEL_PATH)
    print("\nTo use the improved model:")
    print("1. Rename it: mv saved_models/banana_leaf_model_improved.h5 saved_models/banana_leaf_model.h5")
    print("2. Or update your app to use: banana_leaf_model_improved.h5")
else:
    print("\n⚠️ No significant improvement. Original model might already be optimal.")
    print("Consider:")
    print("- Adding more training data")
    print("- Trying a different architecture")
    print("- Adjusting hyperparameters")

print("\n" + "="*60)
print("FINE-TUNING COMPLETE!")
print("="*60)
