"""
FastAPI Gateway для Triton Inference Server (Классификация изображений).
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from pydantic import BaseModel
from typing import List, Union, Optional
import tritonclient.grpc as grpcclient
import numpy as np
import asyncio
import json
import os
import base64
import io
from PIL import Image
from fastapi import Form

# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════

TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

# ═══════════════════════════════════════════════════════════════════
# СОЗДАНИЕ ПРИЛОЖЕНИЯ
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Image Classification API",
    description="API для классификации изображений через Triton",
    version="1.1.0"
)

# ═══════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ═══════════════════════════════════════════════════════════════════

triton_client = None
class_names = []
input_name = None
output_name = None

# ═══════════════════════════════════════════════════════════════════
# МОДЕЛИ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════

class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    all_scores: dict

# ═══════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    global triton_client, class_names, input_name, output_name

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
    
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)

    for attempt in range(30):
        try:
            if triton_client.is_server_live() and triton_client.is_model_ready(MODEL_NAME):
                metadata = triton_client.get_model_metadata(MODEL_NAME)
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
                print(f"✅ API Ready. Model: {MODEL_NAME}")
                return
        except Exception:
            await asyncio.sleep(2)
    print("⚠️ Triton not reached during startup, will retry on request.")

# ═══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════

def preprocess_image(img: Image.Image):
    img = img.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ═══════════════════════════════════════════════════════════════════
# ЭНДПОИНТЫ
# ═══════════════════════════════════════════════════════════════════

@app.get("/health")  # HTTP GET запрос на /health
async def health():
    """
    Health check — проверка состояния сервиса.
    Используется Docker'ом, Kubernetes, load balancer'ами для мониторинга.
    """
    try:
        return {
            # Статус: healthy если Triton отвечает, иначе unhealthy
            "status": "healthy" if triton_client.is_server_live() else "unhealthy",
            # Готова ли модель принимать запросы
            "model_ready": triton_client.is_model_ready(MODEL_NAME),
            # Имена слоёв модели (для отладки)
            "input_name": input_name,
            "output_name": output_name
        }
    except Exception as e:
        # Если ошибка при проверке — сервис нездоров
        return {"status": "unhealthy", "error": str(e)}

@app.get("/classes")
async def get_classes():
    return {"classes": class_names}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...)  # Делаем обязательным
):
    try:
        # 1. Загрузка изображения
        content = await file.read()
        img = Image.open(io.BytesIO(content))

        # 2. Препроцессинг
        data = preprocess_image(img)

        # 3. Инференс
        input_name_str = str(input_name)  # Принудительно в строку
        inputs = [grpcclient.InferInput(input_name_str, list(data.shape), "FP32")]
        inputs[0].set_data_from_numpy(data)

        output_name_str = str(output_name)  # И output тоже
        outputs = [grpcclient.InferRequestedOutput(output_name_str)]

        result = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        logits = result.as_numpy(output_name_str)[0]
        print(class_names)
        # 4. Постпроцессинг
        idx = np.argmax(logits)
        return {
            "class_name": class_names[idx] if class_names else str(idx),
            "confidence": float(logits[idx]),
            "all_scores": {class_names[i]: float(logits[i]) for i in range(len(logits))}
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)  # В логи контейнера
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "traceback": error_trace
            }
        )