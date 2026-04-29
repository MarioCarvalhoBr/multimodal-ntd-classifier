import sys
from pathlib import Path

# Ajuste de path para o servidor enxergar as pastas
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router

# Importando a sua classe de configurações dinâmicas
from config.config import load_config

# Carrega as configs (do Pydantic e/ou config.yaml)
settings = load_config()

# Inicialização do app com o /docs configurado
app = FastAPI(
    title="CADTN - Classificador Multimodal API",
    description="API robusta para inferência de Doenças Tropicais Negligenciadas.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Acoplando as rotas
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "server.server:app", 
        host=settings.api_host, 
        port=settings.api_port, 
        reload=True, 
        log_level=settings.log_level.lower()
    )