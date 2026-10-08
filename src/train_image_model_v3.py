from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "train"

MODEL_OUT = PROJECT_DIR / "src" / "coffee_roast_model_v3.keras"
BEST_MODEL = PROJECT_DIR / "src" / "coffee_roast_model_v3_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "Light",
    "Medium",
    "Dark"
]

VALIDATION_SPLIT = 0.20


# ============================================================
# START
# ============================================================

print("=" * 70)
print("COFFEE ROAST CLASSIFIER - V3")
print("=" * 70)

print("Project :", PROJECT_DIR)
print("Train   :", TRAIN_DIR)
print()


# ============================================================
# CHECK DATASET
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"ไม่พบโฟลเดอร์ train:\n{TRAIN_DIR}"
    )

print("Checking dataset...")

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

for class_name in CLASS_NAMES:

    class_dir = TRAIN_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์:\n{class_dir}"
        )

    count = sum(
        1
        for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in extensions
    )

    print(f"{class_name:8s}: {count} images")

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print()

print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Class order:")
print(train_ds.class_names)
print()


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ],
    name="data_augmentation"
)


# ============================================================
# MOBILE NET V2
# ============================================================

print("Loading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

x = data_augmentation(inputs)

# รับภาพ 0-255
# ไม่ต้อง /255 จากภายนอก
x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    3,
    activation="softmax",
    name="classification"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="coffee_roast_classifier_v3"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL CREATED")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    BEST_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# STAGE 1
# ============================================================

print()
print("=" * 70)
print("STAGE 1 - TRAIN CLASSIFIER")
print("=" * 70)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# STAGE 2
# ============================================================

print()
print("=" * 70)
print("STAGE 2 - FINE TUNING")
print("=" * 70)

base_model.trainable = True

# Freeze everything except last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNorm frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model = keras.models.load_model(
    BEST_MODEL
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print()
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# SAVE
# ============================================================

best_model.save(
    MODEL_OUT
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Model saved:")
print(MODEL_OUT)

print()
print("Class order:")
print(CLASS_NAMES)

print()
print("Input:")
print("Raw image pixels 0-255")
print("Do NOT divide by 255 before prediction.")