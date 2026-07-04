import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# =========================
# CONFIGURACIÓN
# =========================

DATASET_PATH = "dataset"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMAGE_SIZE = 96
BATCH_SIZE = 32

# =========================
# DATA AUGMENTATION
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# GENERADORES
# =========================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# =========================
# MODELO CNN MEJORADO
# =========================

model = Sequential([

    # BLOQUE 1
    Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),
    Dropout(0.25),

    # BLOQUE 2
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),
    Dropout(0.25),

    # BLOQUE 3
    Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),
    Dropout(0.3),

    # REDUCCIÓN MODERNA
    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    BatchNormalization(),

    Dropout(0.5),

    # SOLO 5 EMOCIONES
    Dense(5, activation='softmax')
])

# =========================
# COMPILACIÓN
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# =========================
# ENTRENAMIENTO
# =========================

print("🚀 Entrenando modelo Emotia con FER2013 + RAF-DB...")

history = model.fit(
    train_generator,
    epochs=40,
    validation_data=test_generator,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

# =========================
# GUARDADO
# =========================

model.save("modelo_emociones.h5")

print("✅ Modelo guardado correctamente")