import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import config


class CustomLLM:
    def __init__(self):
        print("Loading Custom Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)

        # Load Base Model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Load Custom Adapter (if exists)
        try:
            self.model = PeftModel.from_pretrained(self.base_model, config.ADAPTER_PATH)
            print("Custom Adapter loaded successfully.")
        except Exception:
            print("No adapter found, using base model.")
            self.model = self.base_model

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500,
            temperature=0.1
        )

    def generate_sql(self, schema_info, feature_name):
        prompt = f"""
        ### Instruction:
        You are a QA Engineer. Generate SQL INSERT statements for the following tables.
        Feature: {feature_name}

        ### Schema:
        {schema_info}

        ### Response:
        (Provide only SQL code)
        """

        result = self.pipe(prompt)
        # Extracting the generated text after the prompt
        generated_text = result[0]['generated_text'].replace(prompt, "").strip()
        return generated_text