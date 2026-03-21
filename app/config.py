from pydantic import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    # === LLM Configuration ===
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistral")

    # === Retrieval Configuration ===
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # === Vector Store Configuration ===
    VECTOR_DB_PATH: str = "data/processed/vector_store"

    class Config:
        env_file = ".env"


# Create a single settings instance
settings = Settings()