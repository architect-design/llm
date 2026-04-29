"""
Training Pipeline for Financial SLM
Auto-selects Transformer (PyTorch) or Bigram fallback (numpy)
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Detect backend
try:
    import torch
    BACKEND = "transformer"
except ImportError:
    BACKEND = "bigram"
    print("PyTorch not found — using Bigram SLM fallback (numpy)")

from data.generator import ACHGenerator, VCFGenerator

DEFAULT_CONFIG = {
    "ach": {
        "model_config": "small",
        "n_training_files": 300,
        "n_val_files": 40,
        "block_size": 512,
        "batch_size": 4,
        "max_epochs": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "eval_every": 50,
        "save_dir": os.path.join(BASE_DIR, "trained_models"),
    },
    "vcf": {
        "model_config": "small",
        "n_training_files": 300,
        "n_val_files": 40,
        "block_size": 512,
        "batch_size": 4,
        "max_epochs": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "eval_every": 50,
        "save_dir": os.path.join(BASE_DIR, "trained_models"),
    }
}


class Trainer:
    def __init__(self, file_type: str = "ACH", config: dict = None):
        self.file_type = file_type.upper()
        self.cfg = config or DEFAULT_CONFIG[file_type.lower()]
        self.backend = BACKEND
        os.makedirs(self.cfg["save_dir"], exist_ok=True)
        print(f"Trainer: {self.file_type}, backend={self.backend}")

    def generate_training_data(self, callback=None):
        gen = ACHGenerator() if self.file_type == "ACH" else VCFGenerator()
        n_train = self.cfg["n_training_files"]
        n_val = self.cfg["n_val_files"]
        train_files, val_files = [], []

        for i in range(n_train + n_val):
            f = gen.generate_file() if self.file_type == "ACH" else gen.generate_file()
            (train_files if i < n_train else val_files).append(f)
            if callback and i % 50 == 0:
                prog = int((i / (n_train + n_val)) * 30)
                callback(f"Generated {i}/{n_train+n_val} files...", prog)

        return train_files, val_files

    def train(self, callback=None, status_file=None):
        def report(msg, progress=None):
            print(msg)
            if callback:
                callback(msg, progress)

        report("Generating training data...", 5)
        train_files, val_files = self.generate_training_data(report)
        report(f"Generated {len(train_files)} train / {len(val_files)} val files", 30)

        if self.backend == "transformer":
            return self._train_transformer(train_files, val_files, report)
        else:
            return self._train_bigram(train_files, val_files, report)

    def _train_bigram(self, train_files, val_files, report):
        from model.bigram_fallback import BigramSLM

        report("Training Bigram SLM (numpy backend)...", 40)
        model = BigramSLM(order=5)
        model.file_type = self.file_type

        def cb(msg, prog):
            report(msg, prog)

        model.train(train_files, callback=cb)
        report("Training complete!", 95)

        # Evaluate perplexity on val set (approx)
        val_score = self._bigram_eval(model, val_files[:20])
        report(f"Val perplexity (approx): {val_score:.2f}", 97)

        save_path = os.path.join(self.cfg["save_dir"], f"{self.file_type.lower()}_model.pkl")
        model.save(save_path)
        report("Model saved!", 100)

        return {
            "model_path": save_path,
            "backend": "bigram",
            "best_val_loss": val_score,
            "train_losses": [val_score],
        }

    def _bigram_eval(self, model, val_files):
        import numpy as np
        total_log = 0.0
        total_chars = 0
        for text in val_files:
            for j in range(len(text) - model.order):
                ctx = text[j: j + model.order]
                nxt = text[j + model.order]
                if ctx in model.counts and nxt in model.counts[ctx]:
                    total = sum(model.counts[ctx].values())
                    p = model.counts[ctx][nxt] / total
                    total_log += np.log(max(p, 1e-10))
                    total_chars += 1
        if total_chars == 0:
            return 999.0
        avg_nll = -total_log / total_chars
        return float(np.exp(avg_nll))

    def _train_transformer(self, train_files, val_files, report):
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from model.transformer import FinancialSLM
        from model.tokenizer import FinancialTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = FinancialTokenizer()

        report("Tokenizing data...", 35)
        train_inputs, train_targets = tokenizer.create_training_batch(
            train_files, block_size=self.cfg["block_size"], file_type=self.file_type
        )
        val_inputs, val_targets = tokenizer.create_training_batch(
            val_files, block_size=self.cfg["block_size"], file_type=self.file_type
        )
        report(f"Tokenized: {train_inputs.shape[0]} train, {val_inputs.shape[0]} val batches", 40)

        model = FinancialSLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=self.cfg["block_size"],
            config=self.cfg["model_config"]
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=self.cfg["learning_rate"],
                          weight_decay=self.cfg["weight_decay"], betas=(0.9, 0.95))
        n_steps = len(train_inputs) * self.cfg["max_epochs"]
        scheduler = CosineAnnealingLR(optimizer, T_max=max(n_steps, 1), eta_min=1e-5)

        batch_size = self.cfg["batch_size"]
        best_val_loss = float('inf')
        train_losses = []

        for epoch in range(self.cfg["max_epochs"]):
            model.train()
            epoch_loss, n_batch, step = 0.0, 0, 0
            perm = torch.randperm(len(train_inputs))
            train_inputs_s = train_inputs[perm]
            train_targets_s = train_targets[perm]

            for i in range(0, len(train_inputs_s), batch_size):
                xb = train_inputs_s[i:i+batch_size].to(device)
                yb = train_targets_s[i:i+batch_size].to(device)
                optimizer.zero_grad()
                _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batch += 1
                step += 1

            avg_loss = epoch_loss / max(n_batch, 1)
            train_losses.append(avg_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, min(len(val_inputs), 16), batch_size):
                    xv = val_inputs[i:i+batch_size].to(device)
                    yv = val_targets[i:i+batch_size].to(device)
                    _, vloss = model(xv, yv)
                    val_loss += vloss.item()
            val_loss /= max(min(16, len(val_inputs)) // batch_size, 1)

            progress = 45 + int(((epoch + 1) / self.cfg["max_epochs"]) * 50)
            report(f"Epoch {epoch+1}/{self.cfg['max_epochs']} | train={avg_loss:.4f} | val={val_loss:.4f}", progress)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.cfg["save_dir"], f"{self.file_type.lower()}_model.pt")
                model.save(save_path, tokenizer, self.file_type)

        tok_path = os.path.join(self.cfg["save_dir"], f"{self.file_type.lower()}_tokenizer.json")
        tokenizer.save(tok_path)
        report("Training complete!", 100)

        return {
            "model_path": os.path.join(self.cfg["save_dir"], f"{self.file_type.lower()}_model.pt"),
            "tokenizer_path": tok_path,
            "backend": "transformer",
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
        }


class FileGenerator:
    """Generate files using a trained model (transformer or bigram)"""

    def __init__(self, file_type: str, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(BASE_DIR, "trained_models")
        self.file_type = file_type.upper()
        self.model_dir = model_dir

        # Try transformer model first
        pt_path = os.path.join(model_dir, f"{file_type.lower()}_model.pt")
        pkl_path = os.path.join(model_dir, f"{file_type.lower()}_model.pkl")

        if BACKEND == "transformer" and os.path.exists(pt_path):
            self._load_transformer(pt_path)
            self.backend = "transformer"
        elif os.path.exists(pkl_path):
            self._load_bigram(pkl_path)
            self.backend = "bigram"
        elif os.path.exists(pt_path):
            self._load_transformer(pt_path)
            self.backend = "transformer"
        else:
            raise FileNotFoundError(f"No trained model found. Train the model first.")

    def _load_transformer(self, path):
        from model.transformer import FinancialSLM
        from model.tokenizer import FinancialTokenizer
        self.model, checkpoint = FinancialSLM.load(path)
        self.model.eval()
        tok_path = path.replace("_model.pt", "_tokenizer.json")
        self.tokenizer = FinancialTokenizer.load(tok_path) if os.path.exists(tok_path) else FinancialTokenizer()

    def _load_bigram(self, path):
        from model.bigram_fallback import BigramSLM
        self.model = BigramSLM.load(path)
        self.tokenizer = None

    def generate_with_seed(self, temperature=0.8, max_tokens=2000, **kwargs) -> str:
        if self.backend == "transformer":
            import torch
            seed = "1" if self.file_type == "ACH" else "VCF|"
            tokens = self.tokenizer.encode(seed, add_special_tokens=False)
            idx = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                generated = self.model.generate(
                    idx, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=40, top_p=0.9,
                    stop_token=self.tokenizer.char2idx.get("<EOS>")
                )
            return self.tokenizer.decode(generated[0].tolist())
        else:
            return self.model.generate(temperature=temperature, max_chars=max_tokens)

