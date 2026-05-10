Custom Test Data Generation Agent
An autonomous AI agent that generates and executes test data for specific application features using a locally fine-tuned Large Language Model.

Features
Privacy First: No data leaves your environment. The model runs locally.
Intelligent Data: Understands Foreign Key relationships and data types automatically.
Self-Contained: Includes training pipeline to create your own custom model.

Architecture
Schema Introspector: Reads database structure (SQLAlchemy).
Custom LLM: A fine-tuned Mistral/Llama model optimized for SQL generation.
Execution Engine: Inserts data, runs tests, and rolls back changes.
Prerequisites
Python 3.10+
PostgreSQL (or other SQL DB)
NVIDIA GPU (Recommended for training, CPU works for inference but is slower)

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

Usage Workflow
Step 1: Generate Training Data
Create a synthetic dataset to teach the model how to write SQL for your specific domain.
python -m training.generate_dataset

Step 2: Fine-Tune the Model
Train the adapter weights (LoRA) on your generated dataset.
python -m training.train_model
Note: This creates a folder ./fine_tuned_adapter containing your custom model weights.

Step 3: Run the Agent
Execute the agent against your database.
python main.py --feature "User Login" --tables "users,roles"

Configuration
Edit config.py to set your Database URI and Model paths.

---

### 2. `requirements.txt`

```text
torch>=2.0.0
transformers>=4.34.0
peft>=0.4.0
accelerate>=0.23.0
bitsandbytes>=0.41.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
tqdm
datasets