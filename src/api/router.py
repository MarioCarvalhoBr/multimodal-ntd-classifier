from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from services.inference import run_inference, ALLOWED_MODELS

# Cria o roteador
router = APIRouter()

@router.get("/")
def read_root():
    return {"status": "API Online", "models": ALLOWED_MODELS}

@router.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    # Validações de segurança
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Modelo não suportado.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo inválido. Envie uma imagem.")

    # Lê os bytes
    image_bytes = await file.read()
    
    # Chama o serviço de IA (agora isolado!)
    try:
        results = run_inference(image_bytes, model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no PyTorch: {str(e)}")

    return {
        "model_used": model_name,
        "filename": file.filename,
        "predictions": results
    }