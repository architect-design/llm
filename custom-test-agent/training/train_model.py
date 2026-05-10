import sys

sys.path.append('..')  # Add root to path

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import config


def main():
    # 1. Load Model and Tokenizer
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        load_in_4bit=True,  # Quantization for memory efficiency
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare for LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 3. Load Dataset
    print("Loading dataset...")
    data = load_dataset("json", data_files="../training_data/dataset.jsonl", split="train")

    # Simple formatting function
    def format_prompt(example):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

    def tokenize_function(example):
        prompt = format_prompt(example)
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=256)

    tokenized_data = data.map(tokenize_function, batched=False)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="epoch"
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    print("Starting training...")
    trainer.train()

    # 6. Save Adapter
    print("Saving adapter...")
    model.save_pretrained(config.ADAPTER_PATH)
    tokenizer.save_pretrained(config.ADAPTER_PATH)
    print(f"Model saved to {config.ADAPTER_PATH}")


if __name__ == "__main__":
    main()