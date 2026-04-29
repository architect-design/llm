import torch
import json
import os
from models.slm_architecture import TinyFinancialSLM
from config.settings import settings


class ModelInference:
    def __init__(self):
        self.device = "cpu"

        # Model definition must match training architecture
        self.vocab_size = 100
        self.embed_dim = 32
        self.hidden_dim = 64

        self.model = TinyFinancialSLM(self.vocab_size, self.embed_dim, self.hidden_dim)

        # Attempt to load trained weights
        self.loaded = False
        if os.path.exists(settings.MODEL_CHECKPOINT):
            try:
                # Load the state dict
                self.model.load_state_dict(torch.load(settings.MODEL_CHECKPOINT, map_location=self.device))
                self.model.eval()  # Set to evaluation mode
                self.loaded = True
                print(f"✅ SLM Loaded successfully from {settings.MODEL_CHECKPOINT}")
            except Exception as e:
                print(f"❌ Error loading model: {e}. Falling back to simulation.")
        else:
            print(f"⚠️ No model file found at {settings.MODEL_CHECKPOINT}. Running in Simulation Mode.")
            print("Run 'python models/train.py' to train and save the model.")

    def generate_transactions(self, prompt):
        """
        If model is loaded, runs actual inference (mocked logic here for JSON output).
        If not loaded, runs rule-based simulation.
        """

        # Determine number of transactions from prompt (Simple parsing)
        num_tx = 1
        words = prompt.split()
        for word in words:
            if word.isdigit():
                num_tx = int(word)
                break

        generated_data = []

        # NOTE: In a real SLM, you would tokenize the prompt and run self.model(input_ids)
        # Here we simulate the output format regardless of load status for this demo.

        for i in range(num_tx):
            # SLM Output Logic (Simulated)
            tx = {
                "routing": "021000021",
                "account": f"8888{str(i).zfill(5)}",
                "amount": 1500.00 + (i * 50),
                "name": f"AI GENERATED USER {i + 1}",
                "credit": "debit" not in prompt.lower()
            }
            generated_data.append(tx)

        return generated_data


# Singleton instance
inference_engine = ModelInference()