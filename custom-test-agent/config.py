import os

# Database Configuration
DB_URI = "postgresql://postgres:password@localhost:5432/testdb"

# Model Configuration
# Using a small model for efficiency. Change to 'mistralai/Mistral-7B-v0.1' for better results.
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./fine_tuned_adapter"

# Device Configuration
DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"