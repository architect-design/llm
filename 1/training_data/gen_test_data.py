import markovify
import os
import random
import argparse
from pathlib import Path

class FileTypeGenerator:
    def __init__(self, train_dir="training_data"):
        self.models = {}
        self.train_dir = Path(train_dir)
        for file_type in ["ach", "vcf"]:
            self._train(file_type)
    
    def _train(self, file_type):
        """Train a character‑based Markov model on all samples of given type."""
        text = ""
        sample_dir = self.train_dir / file_type
        if not sample_dir.exists():
            raise FileNotFoundError(f"Missing {sample_dir}/ – add training files")
        for fpath in sample_dir.glob("*"):
            with open(fpath, "r", encoding="utf-8") as f:
                text += f.read() + "\n\n"  # separate files with blank line
        # Build model (state size = 3 characters)
        model = markovify.Text(text, state_size=3)
        self.models[file_type] = model
    
    def generate(self, file_type, num_files=1, max_chars=1000):
        """Generate one or more test files of the given type."""
        model = self.models.get(file_type.lower())
        if not model:
            raise ValueError(f"Unknown type: {file_type}. Use 'ach' or 'vcf'.")
        generated = []
        for _ in range(num_files):
            # Generate until we get a plausible length and proper start/end
            while True:
                out = model.make_short_sentence(max_chars, max_overlap_ratio=0.3, tries=100)
                if out:
                    if file_type == "vcf" and out.strip().upper().startswith("BEGIN:VCARD"):
                        break
                    elif file_type == "ach" and out.strip().startswith("101"):
                        break
            # Add a final newline and ensure END:VCARD for vcf
            if file_type == "vcf" and "END:VCARD" not in out.upper():
                out += "\nEND:VCARD"
            generated.append(out)
        return generated if num_files > 1 else generated[0]

def main():
    parser = argparse.ArgumentParser(description="Generate test ACH or VCF files")
    parser.add_argument("type", choices=["ach", "vcf"], help="File type to generate")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of files")
    parser.add_argument("-o", "--output-dir", default="generated", help="Output directory")
    args = parser.parse_args()
    
    # Train the model (first run reads training_data/)
    generator = FileTypeGenerator("training_data")
    
    # Generate
    outputs = generator.generate(args.type, args.num)
    if args.num == 1:
        outputs = [outputs]
    
    Path(args.output_dir).mkdir(exist_ok=True)
    for i, content in enumerate(outputs, 1):
        ext = "ach" if args.type == "ach" else "vcf"
        out_file = Path(args.output_dir) / f"test_{args.type}_{i}.{ext}"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Generated: {out_file}")

if __name__ == "__main__":
    main()