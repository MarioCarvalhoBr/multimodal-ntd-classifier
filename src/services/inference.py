import io
import torch
from PIL import Image
from functools import lru_cache
import sys
from pathlib import Path
import os


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

# Must be sorted: NTDDataset assigns label IDs in alphabetical order
CLASSES = sorted([
    "microscopy_parasite_babesia",
    "microscopy_parasite_plasmodium",
    "microscopy_parasite_trichomonad",
    "microscopy_parasite_leishmania",
    "microscopy_parasite_rbcs",
    "microscopy_parasite_trypanosome",
    "microscopy_parasite_leukocyte",
    "microscopy_parasite_toxoplasma",
])


@lru_cache(maxsize=3)
def get_model(model_name: str):
    print(f"[*] Carregando arquitetura e pesos do {model_name}...")
    
    # 1. Instancia a arquitetura vazia com as 8 classes
    model, processor = ModelFactory.create(model_name, num_classes=len(CLASSES))
    
    # 2. Formata o caminho baseado na sua árvore de diretórios (substituindo '/' por '_')
    safe_name = model_name.replace('/', '_')
    weight_path = Path(f"output/results/{safe_name}/saved_model/best_{safe_name}.pth")
    
    # 3. Trava de segurança: Previne o erro 500 silencioso
    if not weight_path.exists():
        raise FileNotFoundError(f"Pesos não encontrados no disco! Caminho esperado: {weight_path}")
        
    # 4. Mapeamento de Memória Seguro (CRÍTICO PARA API)
    # Garante que os tensores carreguem na memória certa, independente de onde foram treinados
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[*] Carregando state_dict de {weight_path} para {device}...")
    # weights_only=True aumenta a segurança da API contra injeção de código em arquivos .pth
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # 5. Move o modelo para a Placa de Vídeo (ou CPU) e ativa modo de Teste
    model.to(device)
    model.eval()
    
    # Retornamos o device junto para sabermos para onde enviar a imagem depois
    return model, processor, device

def run_inference(image_bytes: bytes, model_name: str) -> list:
    """Executa a inferência real com a imagem enviada pelo Frontend."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Busca o modelo do cache
    model, processor, device = get_model(model_name)
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # CRÍTICO: Move os pixels da imagem para o mesmo device (GPU/CPU) onde o modelo está
        pixel_values = inputs.pixel_values.to(device)
        
        outputs = model(pixel_values)
        
        # O [0] pega a primeira imagem do batch (já que estamos enviando 1 por vez)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Extrai o Top-5
    top_prob, top_indices = torch.topk(probabilities, 5)
    
    results = []
    for i in range(5):
        idx = top_indices[i].item()
        results.append({
            "class_name": CLASSES[idx],
            "confidence": float(top_prob[i].item())
        })
    
    return sorted(results, key=lambda x: x["confidence"], reverse=True)