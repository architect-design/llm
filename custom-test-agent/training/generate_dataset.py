import json
import os

# Create output directory
os.makedirs("training_data", exist_ok=True)


def generate_sample_entry():
    # This is a template. In production, vary these heavily.
    schema = """
    CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR, role VARCHAR);
    """
    instruction = "Generate test data for an admin user login test."
    output = "INSERT INTO users (id, email, role) VALUES (1, 'admin@test.com', 'admin');"

    # Format for standard instruction tuning
    return {
        "instruction": instruction,
        "input": schema,
        "output": output
    }


def main():
    dataset = []
    # Generate 100 variations (In reality, you need 1000+ for fine-tuning)
    for i in range(100):
        entry = generate_sample_entry()
        # Simple variation logic for demo
        entry['input'] = entry['input'].replace("users", f"table_{i}")
        dataset.append(entry)

    # Save as JSONL
    with open("training_data/dataset.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print("Dataset generated at training_data/dataset.jsonl")


if __name__ == "__main__":
    main()