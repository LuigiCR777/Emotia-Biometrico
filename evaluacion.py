import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(
    "modelo_emotia_rafdb.h5"
)

# =========================
# CLASES
# =========================

clases_modelo = [
    'angry',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# =========================
# DATASET TEST
# =========================

DATASET_PATH = "dataset"

TEST_DIR = os.path.join(
    DATASET_PATH,
    "test"
)

IMAGE_SIZE = 96
BATCH_SIZE = 32

# =========================
# GENERADOR
# =========================

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    color_mode='rgb',

    shuffle=False
)

# =========================
# PREDICCIÓN
# =========================

print("🔍 Evaluando modelo...")

y_pred = model.predict(
    test_generator
)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = test_generator.classes

# =========================
# MÉTRICAS
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred_classes
)

precision = precision_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

conf_matrix = confusion_matrix(
    y_true,
    y_pred_classes
)

# =========================
# RESULTADOS
# =========================

print("\n==============================")
print("📊 RESULTADOS DE EVALUACIÓN")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\n==============================")
print("📄 Classification Report")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=clases_modelo
    )
)

print("\n==============================")
print("🧩 Confusion Matrix")
print("==============================")

print(conf_matrix)