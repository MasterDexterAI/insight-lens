import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
CHROMA_DB_PATH = str(DATA_DIR / "chromadb")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

# ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")
OLLAMA_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "qwen3.5:2b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# florence-2
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE_DEVICE = os.getenv("FLORENCE_DEVICE","cpu")
FLORENCE_ATTN_IMPLEMENTATION = "eager"

# image handling
MAX_IMAGE_SIZE = 1536

# retrieval
CHROMA_COLLECTION_NAME = "insight_lens_pages"
TOP_K_RESULTS = 5
