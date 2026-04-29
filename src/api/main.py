import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import random
import time

import torch
from PIL import Image
import io
from models.factory import ModelFactory

app = FastAPI(title="NTD Classifier API", description="API para classificação de Doenças Tropicais Negligenciadas")

# Configuração CORS para permitir que o frontend converse com a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, defina a URL do seu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MODELS = [
    "microsoft/resnet-50",
    "google/efficientnet-b3",
    "google/vit-base-patch16-224",
    "openai/clip-vit-base-patch16",
    "google/siglip-base-patch16-224"
]

# Classes que estamos treinando no momento
CLASSES = [
    "microscopy_parasite_babesia",
    "microscopy_parasite_plasmodium",
    "microscopy_parasite_trichomonad",
    "microscopy_parasite_leishmania",
    "microscopy_parasite_rbcs",
    "microscopy_parasite_trypanosome",
    "microscopy_parasite_leukocyte",
    "microscopy_parasite_toxoplasma"
]

@app.get("/")
def read_root():
    return {"status": "API Online", "models": ALLOWED_MODELS}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Modelo não suportado.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo inválido. Envie uma imagem.")

    # Lendo o arquivo da memória
    image_bytes = await file.read()
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model, processor = ModelFactory.create(model_name, num_classes=len(CLASSES))
    # Carregue os pesos aqui: model.load_state_dict(torch.load(f"output/results/{model_name.replace('/','_')}/saved_model/best...pth"))
    model.eval()
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.pixel_values)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Pega o Top-5
    top_prob, top_indices = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        idx = top_indices[i].item()
        results.append({
            "class_name": CLASSES[idx],
            "confidence": float(top_prob[i].item())
        })
    
    # Ordena por confiança
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return {
        "model_used": model_name,
        "filename": file.filename,
        "predictions": results
    }