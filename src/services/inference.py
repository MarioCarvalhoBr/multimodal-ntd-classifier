import io
import torch
from PIL import Image
from functools import lru_cache
import sys
from pathlib import Path

# Garante que as importações a partir de 'src' funcionem
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.factory import ModelFactory

ALLOWED_MODELS = [
    "microsoft/resnet-50",
    "google/efficientnet-b3",
    "google/vit-base-patch16-224",
    "openai/clip-vit-base-patch16",
    "google/siglip-base-patch16-224"
]

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

# OTIMIZAÇÃO: @lru_cache impede que o PyTorch recarregue o modelo do HD a cada clique!
@lru_cache(maxsize=3)
def get_model(model_name: str):
    print(f"[*] Carregando modelo {model_name} na memória pela primeira vez...")
    model, processor = ModelFactory.create(model_name, num_classes=len(CLASSES))
    # TODO: No futuro, carregue seus pesos treinados aqui:
    # model.load_state_dict(torch.load(f"output/results/{model_name.replace('/','_')}/saved_model/best...pth"))
    model.eval()
    return model, processor

def run_inference(image_bytes: bytes, model_name: str) -> list:
    """Executa a inferência e retorna a lista predições ordenadas."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model, processor = get_model(model_name)
    
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
    
    return sorted(results, key=lambda x: x["confidence"], reverse=True)