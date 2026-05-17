# FUTURE WORK
from pathlib import Path

# Only these two VLMs are supported for multimodal fusion
ALLOWED_MODELS = [
    "google/siglip-base-patch16-224",
    "openai/clip-vit-base-patch16",
]

# Output embedding dimension for each backbone
MODEL_DIMENSIONS = {
    "google/siglip-base-patch16-224": 768,
    "openai/clip-vit-base-patch16": 512,
}

# Path to the vector knowledge base (images + informações.txt per class)
VECTOR_BASE_PATH = Path("dataset/processed/Dataset-NTD-V1/vector_base")

# Persistent Qdrant storage path
DB_PATH = "output/multimodal_vector_db"

# Qdrant collection name prefix
COLLECTION_PREFIX = "ntd"

# Filename for class textual descriptions inside each class folder
INFO_TEXT = "informações.txt"

# Single fenotipo used in NTD vector base (images subfolder name)
FENOTIPOS = ["IMAGEM"]

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Late-fusion weights: must sum to 1.0
PESOS = {
    "MODELO": 0.5,   # fine-tuned linear probe score
    "IMAGEM": 0.3,   # cosine similarity to indexed images (VectorDB)
    "TEXTO":  0.2,   # cosine similarity to indexed text descriptions (VectorDB)
}

# Number of nearest neighbours to retrieve per query
TOP_K = 8
