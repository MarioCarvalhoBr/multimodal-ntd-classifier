import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient

from models.factory import ModelFactory
from utils.logger import logger
from .config import DB_PATH, PESOS, TOP_K, ALLOWED_MODELS
from .indexer import get_collection_name


class MultimodalPredictor:
    """Late-fusion predictor: trained linear probe (0.4) + image VDB (0.3) + text VDB (0.3).

    Usage:
        predictor = MultimodalPredictor(model_name, num_classes, classes, device)
        preds = predictor.predict_batch(pixel_values_tensor)  # returns list[int]
        predictor.close()
    """

    def __init__(self, model_name: str, num_classes: int, classes: list, device: str):
        if model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' não suportado para predição multimodal. "
                f"Permitidos: {ALLOWED_MODELS}"
            )

        self.classes = classes
        self.device = device

        # Fine-tuned model (frozen backbone + linear head loaded from checkpoint)
        model_tag = model_name.replace("/", "_")
        ckpt_path = f"output/results/{model_tag}/saved_model/best_{model_tag}.pth"
        logger.info(f"[*] Carregando modelo treinado: {ckpt_path}")

        self.model, self.processor = ModelFactory.create(model_name, num_classes)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device).eval()

        # Qdrant
        collection_name = get_collection_name(model_name)
        self._qdrant = QdrantClient(path=DB_PATH)
        self._collection = collection_name

        if not self._qdrant.collection_exists(collection_name):
            self._qdrant.close()
            raise RuntimeError(
                f"Coleção '{collection_name}' não encontrada no VectorDB. "
                "Execute o script com --build-index primeiro."
            )
        logger.info(f"[+] VectorDB pronto: coleção '{collection_name}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backbone_embeddings(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Extract L2-normalised backbone embeddings from the frozen backbone."""
        with torch.no_grad():
            out = self.model.backbone.get_image_features(pixel_values)
            if hasattr(out, 'pooler_output'):
                embeds = out.pooler_output
            elif hasattr(out, 'image_embeds'):
                embeds = out.image_embeds
            else:
                embeds = out
            return F.normalize(embeds, dim=1).cpu().numpy()

    def _vdb_scores(self, vector: list) -> tuple[dict, dict]:
        """Return per-class image and text similarity scores from Qdrant."""
        hits = self._qdrant.query_points(
            collection_name=self._collection,
            query=vector,
            limit=TOP_K * len(self.classes),
        ).points

        img_acc = {c: 0.0 for c in self.classes}
        txt_acc = {c: 0.0 for c in self.classes}

        for hit in hits:
            cls = hit.payload.get("class_name", "")
            if cls not in img_acc:
                continue
            if hit.payload["type"] == "IMAGEM":
                img_acc[cls] += hit.score
            else:
                txt_acc[cls] += hit.score

        return img_acc, txt_acc

    @staticmethod
    def _normalize(scores: dict) -> dict:
        """Normalise a class→score dict so values sum to 1."""
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_batch(self, pixel_values: torch.Tensor) -> list[int]:
        """Run late-fusion prediction on a batch.

        Args:
            pixel_values: preprocessed image tensor (B, C, H, W) on self.device.

        Returns:
            List of predicted class indices, length B.
        """
        batch_size = pixel_values.shape[0]

        # 1. Fine-tuned model: softmax probabilities  [weight MODELO = 0.4]
        with torch.no_grad():
            logits = self.model(pixel_values)
            model_probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 2. Backbone embeddings for VectorDB search
        embeddings = self._backbone_embeddings(pixel_values)

        final_preds: list[int] = []

        for i in range(batch_size):
            img_acc, txt_acc = self._vdb_scores(embeddings[i].tolist())
            norm_img = self._normalize(img_acc)
            norm_txt = self._normalize(txt_acc)

            # Late fusion: MODELO * 0.4 + IMAGEM * 0.3 + TEXTO * 0.3
            fused = [
                PESOS["MODELO"] * float(model_probs[i][j])
                + PESOS["IMAGEM"] * norm_img[self.classes[j]]
                + PESOS["TEXTO"]  * norm_txt[self.classes[j]]
                for j in range(len(self.classes))
            ]

            final_preds.append(int(np.argmax(fused)))

        return final_preds

    def close(self) -> None:
        self._qdrant.close()
