import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"          # <-- point this to your PlantVillage folder
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 10
MODEL_DIR   = "detection/ml_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Check dataset exists, else build a tiny demo dataset ───────────────────
if not os.path.isdir(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
    print(" 'dataset/' folder not found or empty.")
    print("  Download the PlantVillage dataset from Kaggle and place it in 'dataset/'.")
    print(" Generating a tiny synthetic dataset so the script still runs end-to-end (demo only).")

    from PIL import Image
    demo_classes = ["Healthy", "Bacterial_Spot", "Early_Blight", "Late_Blight"]
    rng = np.random.default_rng(42)
    for cls in demo_classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(20):
            arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(cls_dir, f"{cls}_{i}.jpg"))

# ── 2. Data Generators (with augmentation) ─────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

num_classes = len(train_gen.class_indices)
print(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")

# ── 3. Build Model: MobileNetV2 base + custom head ─────────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze pretrained layers initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── 4. Train (head only) ────────────────────────────────────────────────────────
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
)

# ── 5. Fine-tune: unfreeze top layers of base model for better accuracy ────────
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(val_gen)
print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

# ── 7. Save model + class labels ─────────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, "plant_disease_model.h5"))

class_indices = train_gen.class_indices
class_names = {v: k for k, v in class_indices.items()}  # index -> label
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\nModel saved to {MODEL_DIR}/plant_disease_model.h5")
print(f"Class labels saved to {MODEL_DIR}/class_names.json")

# ── 8. Plot training curves (optional, for your report/portfolio) ───────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Loss")

plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
print(f"Training curves saved to {MODEL_DIR}/training_curves.png")