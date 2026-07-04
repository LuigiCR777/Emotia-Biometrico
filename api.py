from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import io
import mediapipe as mp
import cv2

# =========================
# CONFIG
# =========================

IMAGE_SIZE = 96
CONFIDENCE_THRESHOLD = 0.35

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza",
    "Sorpresa"
]

app = FastAPI(title="Emotia Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MEDIAPIPE
# =========================

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# =========================
# MODELO
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo_emociones.h5"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Modelo cargado correctamente")

# =========================
# ENDPOINT
# =========================

@app.post("/predict")
async def predict_emotion(
    image: UploadFile = File(...)
):

    try:
        image_bytes = await image.read()

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        img_np = np.array(
            pil_image
        )

        results = face_detection.process(
            img_np
        )

        # === NUEVA MODIFICACIÓN: RESPUESTA HOMOGÉNEA SI NO HAY ROSTRO ===
        if not results.detections:
            return {
                "status": "error",
                "emocion": "No detectado",
                "probabilidad": 0.0,
                "segunda_emocion": "Ninguna",
                "probabilidades": {emocion: 0.0 for emocion in EMOCIONES},
                "message": "No se detectó rostro"
            }

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        iw, ih = pil_image.size

        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)

        # más margen
        pad_w = int(w * 0.35)
        pad_h = int(h * 0.35)

        x = max(0, x - pad_w)
        y = max(0, y - pad_h)

        w = min(iw - x, w + pad_w * 2)
        h = min(ih - y, h + pad_h * 2)

        face_crop = pil_image.crop(
            (x, y, x + w, y + h)
        )

        face_crop.save(
            os.path.join(
                BASE_DIR,
                "que_ve_la_ia.jpg"
            )
        )

        # =========================
        # PREPROCESAMIENTO
        # =========================

        face_crop = face_crop.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        img = np.array(
            face_crop
        )

        # mejorar contraste
        lab = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2LAB
        )

        l, a, b = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(
            l
        )

        lab = cv2.merge(
            (l, a, b)
        )

        img = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2RGB
        )

        img = img.astype(
            "float32"
        ) / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

        # =========================
        # PREDICCIÓN
        # =========================

        pred = model.predict(
            img,
            verbose=0
        )[0]

        pred_idx = np.argmax(
            pred
        )

        confianza = float(
            pred[pred_idx]
        )

        top2 = np.argsort(
            pred
        )[-2:][::-1]

        print("\n==========")
        print("Predicción:")
        for i in range(len(EMOCIONES)):
            print(
                f"{EMOCIONES[i]}: {pred[i]:.4f}"
            )

        if confianza < CONFIDENCE_THRESHOLD:
            emocion = EMOCIONES[
                top2[0]
            ]
        else:
            emocion = EMOCIONES[
                pred_idx
            ]

        probabilidades = {
            EMOCIONES[i]: round(
                float(pred[i]),
                4
            )
            for i in range(
                len(EMOCIONES)
            )
        }

        return {
            "status": "success",
            "emocion": emocion,
            "probabilidad": round(
                confianza,
                4
            ),
            "segunda_emocion": EMOCIONES[top2[1]],
            "probabilidades": probabilidades
        }

    except Exception as e:
        # === EN CASO DE ERROR CRÍTICO, TAMBIÉN RESPONDEMOS VALORES DE CONTROL ===
        return {
            "status": "error",
            "emocion": "Error",
            "probabilidad": 0.0,
            "segunda_emocion": "Ninguna",
            "probabilidades": {emocion: 0.0 for emocion in EMOCIONES},
            "message": str(e)
        }