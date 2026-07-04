import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# =========================
# CONFIG
# =========================

IMAGE_SIZE = 96

CONFIDENCE_THRESHOLD = 0.60

WINDOW_SIZE = 10

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(
    "modelo_emociones.h5"
)

print("✅ Modelo cargado correctamente")

# =========================
# CLASES
# =========================

clases_modelo = [
    "Angry",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# historial para suavizado temporal
historial_predicciones = deque(maxlen=WINDOW_SIZE)

# =========================
# MEDIAPIPE
# =========================

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

# =========================
# CÁMARA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir cámara")
    exit()

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detection.process(img_rgb)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            ih, iw, _ = frame.shape

            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            # margen extra para capturar mejor expresión
            padding = 20

            x = max(0, x - padding)
            y = max(0, y - padding)

            w = min(iw - x, w + padding * 2)
            h = min(ih - y, h + padding * 2)

            face_roi = frame[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            try:

                # BGR -> RGB
                face_roi = cv2.cvtColor(
                    face_roi,
                    cv2.COLOR_BGR2RGB
                )

                # resize según entrenamiento
                face_roi = cv2.resize(
                    face_roi,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                # normalización
                face_roi = face_roi.astype(
                    "float32"
                ) / 255.0

                face_roi = np.expand_dims(
                    face_roi,
                    axis=0
                )

            except:
                continue

            # =========================
            # PREDICCIÓN
            # =========================

            preds = model.predict(
                face_roi,
                verbose=0
            )[0]

            historial_predicciones.append(preds)

            promedio = np.mean(
                historial_predicciones,
                axis=0
            )

            pred_idx = np.argmax(
                promedio
            )

            confianza = promedio[pred_idx]

            # emoción incierta
            if confianza < CONFIDENCE_THRESHOLD:

                etiqueta = "No clara"

            else:

                etiqueta = clases_modelo[pred_idx]

            texto = f"{etiqueta} ({confianza*100:.1f}%)"

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255,0,0),
                2
            )

            cv2.putText(
                frame,
                texto,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2
            )

    cv2.imshow(
        "Emotia - Reconocimiento Emocional",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()