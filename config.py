import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hospital_ai_copilot")

# ==========================================
# APPLICATION CONSTANTS
# ==========================================
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_FILE_TYPES = ["application/pdf"]

# ==========================================
# AI PIPELINE CONSTANTS
# ==========================================
AI_MODEL_NAME = "openai/gpt-oss-20b"
FAISS_TOP_K = 3
HISTORICAL_PATIENT_COUNT = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")