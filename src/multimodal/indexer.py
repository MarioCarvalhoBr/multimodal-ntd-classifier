# FUTURE WORK

import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from utils.logger import logger
from .config import (
    VECTOR_BASE_PATH, DB_PATH, COLLECTION_PREFIX,
    MODEL_DIMENSIONS, INFO_TEXT, FENOTIPOS,
    SUPPORTED_EXTENSIONS, ALLOWED_MODELS,
)


def get_collection_name(model_name: str) -> str:
    model_tag = model_name.replace("/", "_").replace("-", "_")
    return f"{COLLECTION_PREFIX}_{model_tag}_v1"


def _load_backbone(model_name: str, device: str):
    if "clip" in model_name.lower():
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def _image_embedding(model, processor, img_path: str, device: str) -> list:
    img = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        if hasattr(out, 'pooler_output'):
            embeds = out.pooler_output
        elif hasattr(out, 'image_embeds'):
            embeds = out.image_embeds
        else:
            embeds = out
        return F.normalize(embeds, dim=1).cpu().numpy()[0].tolist()


def _text_embedding(model, processor, text: str, device: str) -> list:
    with torch.no_grad():
        inputs = processor(
            text=[text], padding="max_length",
            truncation=True, max_length=64, return_tensors="pt"
        ).to(device)
        out = model.get_text_features(**inputs)
        if hasattr(out, 'pooler_output'):
            embeds = out.pooler_output
        elif hasattr(out, 'text_embeds'):
            embeds = out.text_embeds
        else:
            embeds = out
        return F.normalize(embeds, dim=1).cpu().numpy()[0].tolist()


def build_index(model_name: str, force: bool = False) -> None:
    """Index vector_base images and text descriptions into Qdrant.

    Structure expected under VECTOR_BASE_PATH:
        {class_name}/
            IMAGEM/
                *.jpg  (or .jpeg, .png)
            informações.txt
    """
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não suportado para indexação multimodal. "
            f"Permitidos: {ALLOWED_MODELS}"
        )

    if not VECTOR_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Pasta vector_base não encontrada: {VECTOR_BASE_PATH}\n"
            "Crie a estrutura: vector_base/{class}/IMAGEM/*.jpg + informações.txt"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[*] Carregando backbone para indexação: {model_name} [{device.upper()}]")

    model, processor = _load_backbone(model_name, device)

    collection_name = get_collection_name(model_name)
    dim = MODEL_DIMENSIONS[model_name]
    client = QdrantClient(path=DB_PATH)

    try:
        if client.collection_exists(collection_name):
            if force:
                client.delete_collection(collection_name)
                logger.info(f"[*] Coleção apagada para reindexação: {collection_name}")
            else:
                logger.info(
                    f"[+] Coleção '{collection_name}' já existe. "
                    "Use --force-reindex para recriar."
                )
                return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"[+] Coleção criada: {collection_name} (dim={dim})")

        points = []
        point_id = 0

        for class_name in sorted(os.listdir(VECTOR_BASE_PATH)):
            class_path = VECTOR_BASE_PATH / class_name
            if not class_path.is_dir():
                continue

            logger.info(f"  -> Indexando: {class_name}")

            # Index images from IMAGEM subfolder
            for fen in FENOTIPOS:
                fen_path = class_path / fen
                if not fen_path.exists():
                    logger.info(f"     [!] Subfolder '{fen}' não encontrado em {class_path}")
                    continue
                for img_name in sorted(os.listdir(fen_path)):
                    if not img_name.lower().endswith(SUPPORTED_EXTENSIONS):
                        continue
                    img_full = str(fen_path / img_name)
                    try:
                        vector = _image_embedding(model, processor, img_full, device)
                        points.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={"class_name": class_name, "type": "IMAGEM", "source": img_full},
                        ))
                        point_id += 1
                    except Exception as e:
                        logger.info(f"     [!] Erro na imagem {img_name}: {e}")

            # Index class text description
            info_path = class_path / INFO_TEXT
            if info_path.exists():
                try:
                    raw_text = info_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if raw_text:
                        vector = _text_embedding(model, processor, raw_text, device)
                        points.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={"class_name": class_name, "type": "TEXTO", "source": str(info_path)},
                        ))
                        point_id += 1
                except Exception as e:
                    logger.info(f"     [!] Erro ao indexar texto de {class_name}: {e}")

        if points:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"[+] Indexação concluída: {point_id} vetores em '{collection_name}'.")
        else:
            logger.info(
                "[!] Nenhum vetor indexado. "
                "Verifique se vector_base/{class}/IMAGEM/ contém imagens."
            )
    finally:
        client.close()
