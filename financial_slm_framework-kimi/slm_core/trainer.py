"""
financial_slm_framework/slm_core/trainer.py
Training pipeline for the Financial SLM.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import random
import os


class FinancialDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_seq_len: int = 2048, augment: bool = True):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record['text']
        record_type = record.get('record_type', 0)
        is_valid = record.get('is_valid', 1)

        if self.augment and is_valid and random.random() < 0.15:
            text = self._corrupt_record(text)
            is_valid = 0

        tokens = self.tokenizer.encode(text, record_type=record_type,
                                        max_length=self.max_seq_len, pad_to_max_length=True)
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'record_type': torch.tensor(record_type, dtype=torch.long),
            'is_valid': torch.tensor(is_valid, dtype=torch.long)
        }

    def _corrupt_record(self, text: str) -> str:
        corruption_type = random.choice(['truncate', 'insert', 'replace', 'swap'])
        chars = list(text)
        if corruption_type == 'truncate' and len(chars) > 10:
            chars = chars[:-random.randint(1, 5)]
        elif corruption_type == 'insert':
            pos = random.randint(0, len(chars))
            chars.insert(pos, random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        elif corruption_type == 'replace' and len(chars) > 0:
            pos = random.randint(0, len(chars) - 1)
            chars[pos] = random.choice('0123456789')
        elif corruption_type == 'swap' and len(chars) > 1:
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)


class FinancialSLMTrainer:
    def __init__(self, model: nn.Module, tokenizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4, weight_decay: float = 0.01, validation_weight: float = 0.3):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.validation_weight = validation_weight
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_ID)
        self.validation_criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            record_types = batch['record_type'].to(self.device)
            is_valid = batch['is_valid'].to(self.device)

            outputs = self.model(input_ids, record_type_ids=record_types, return_validation=True)

            gen_logits = outputs['generation_logits'][:, :-1, :].contiguous()
            gen_targets = input_ids[:, 1:].contiguous()
            gen_loss = self.generation_criterion(gen_logits.view(-1, gen_logits.size(-1)), gen_targets.view(-1))

            val_logits = outputs['validation_logits']
            val_loss = self.validation_criterion(val_logits, is_valid)

            loss = gen_loss + self.validation_weight * val_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                record_types = batch['record_type'].to(self.device)
                is_valid = batch['is_valid'].to(self.device)

                outputs = self.model(input_ids, record_type_ids=record_types, return_validation=True)

                gen_logits = outputs['generation_logits'][:, :-1, :].contiguous()
                gen_targets = input_ids[:, 1:].contiguous()
                gen_loss = self.generation_criterion(gen_logits.view(-1, gen_logits.size(-1)), gen_targets.view(-1))

                val_logits = outputs['validation_logits']
                val_loss = self.validation_criterion(val_logits, is_valid)
                loss = gen_loss + self.validation_weight * val_loss

                preds = val_logits.argmax(dim=-1)
                correct += (preds == is_valid).sum().item()
                total += is_valid.size(0)

                total_loss += loss.item()
                num_batches += 1
        return {'loss': total_loss / max(num_batches, 1), 'accuracy': correct / max(total, 1)}

    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None,
              epochs: int = 10, save_dir: str = './checkpoints', save_every: int = 1):
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            self.history['train_loss'].append(train_loss)
            log_msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}"
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                log_msg += f" | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))
            print(log_msg)
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))

    def save_checkpoint(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history}, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
