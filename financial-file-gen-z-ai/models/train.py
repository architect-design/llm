import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse

# --- PATH FIX START ---
# Add the project root directory to sys.path to allow imports like 'from models...' or 'from config...'
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)
# --- PATH FIX END ---

from models.slm_architecture import TinyFinancialSLM
from config.settings import settings


def train_model(force_retrain=False):
    """
    Trains the SLM model and saves it to the path defined in settings.
    Skips training if the model file already exists, unless force_retrain is True.
    """

    # 1. Check if model already exists
    if os.path.exists(settings.MODEL_CHECKPOINT) and not force_retrain:
        print(f"✅ Trained model found at '{settings.MODEL_CHECKPOINT}'.")
        print("Skipping training. Use '--force' argument to retrain.")
        return

    # 2. Ensure the checkpoint directory exists
    checkpoint_dir = os.path.dirname(settings.MODEL_CHECKPOINT)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created directory: {checkpoint_dir}")

    # 3. Model Configuration
    vocab_size = 100  # Should match inference definition
    embed_dim = 32
    hidden_dim = 64
    epochs = 10

    print("🚀 Starting training process...")
    model = TinyFinancialSLM(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop (Simulated Data)
    # In production, load your actual JSON dataset here
    for epoch in range(epochs):
        # Simulate inputs and targets for demonstration
        inputs = torch.randint(0, vocab_size, (2, 10))  # Batch size 2
        targets = torch.randint(0, vocab_size, (2, 10))

        optimizer.zero_grad()
        outputs, _ = model(inputs)

        # Reshape for CrossEntropyLoss: (N, C, ...)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    # 5. Save the Model State Dictionary
    torch.save(model.state_dict(), settings.MODEL_CHECKPOINT)
    print(f"🎉 Training complete. Model saved to {settings.MODEL_CHECKPOINT}")


if __name__ == "__main__":
    # Argument parser to allow force retraining
    parser = argparse.ArgumentParser(description="Train Financial SLM")
    parser.add_argument('--force', action='store_true', help="Force retrain even if model exists")
    args = parser.parse_args()

    train_model(force_retrain=args.force)