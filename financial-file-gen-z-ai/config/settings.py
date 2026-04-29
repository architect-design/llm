import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    PROJECT_NAME: str = "FinGen Studio"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///financial_data.db")
    MODEL_CHECKPOINT: str = os.getenv("MODEL_CHECKPOINT", "models/checkpoints/slm_v1.pt")

    # ACH Constants
    ACH_RECORD_LENGTH: int = 94
    ACH_BLOCKING_FACTOR: int = 10

    # VCF Constants (Simplified for demo)
    VCF_RECORD_LENGTH: int = 120


settings = Settings()