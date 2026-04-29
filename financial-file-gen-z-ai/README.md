# FinGen Studio: Financial File Generator & Validator

FinGen Studio is a hybrid AI-powered application designed to generate and validate ACH (NACHA) and VISA (VCF) financial files. It utilizes a custom Small Language Model (SLM) to understand user intent and generate structured data, while employing a deterministic Python engine to ensure strict compliance with financial specifications.

**Python** | **Streamlit** | **PyTorch**

## 🚀 Key Features
- **Hybrid Architecture:** Uses SLMs for data context (JSON) and Python for strict formatting (Fixed-Width) to guarantee file validity.
- **Multi-Format Support:** Handles ACH (NACHA) and VISA (VCF) file standards.
- **Validation Engine:** Upload existing files to validate structure, hash totals, and field compliance.
- **Database Integration:** Generate files directly from database records (SQLAlchemy supported).
- **Interactive UI:** User-friendly Streamlit interface for non-technical users.

## 🏗️ Architecture Overview

This project solves the "hallucination problem" of LLMs in financial contexts by splitting the task:

- **The Brain (SLM):** A lightweight Transformer/LSTM model that interprets natural language prompts (e.g., "Generate payroll for 10 employees") and outputs structured JSON data containing the transaction details.
- **The Formatter (Python):** A deterministic engine that takes the JSON and rigorously formats it into NACHA/VCF standards. It handles:
  - Fixed-width padding.
  - Entry hash calculations.
  - Trace number generation.
  - Blocking factors (padding files to multiples of 10).
- **The Validator:** A parser that checks uploaded files against strict compliance rules (record lengths, routing number checksums, etc.).

## 📋 Prerequisites

- Python 3.9 or higher
- pip package installer

## ⚙️ Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/financial-file-gen.git
cd financial-file-gen
```

### Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment
Copy the example environment file and configure your database connection string.
```bash
cp .env.example .env
```

Open `.env` and update the `DATABASE_URL` if you plan to connect to a real database (PostgreSQL, MySQL, etc.). By default, it uses a local SQLite file.

## 🏃 Running the Application

Start the web UI using Streamlit:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## 🔧 Usage Guide

### 1. Generating Files via AI (SLM)
- Open the "Generate Files" tab.
- Select "Option A: Use AI (SLM)".
- Choose your file type (ACH or VCF).
- Enter a natural language prompt, for example:
  - "Generate 5 payroll credit transactions for IT department."
  - "Create a file with 3 debit transactions for vendor payments."
- Click "Generate via AI".
- The model generates JSON data.
- The Python engine formats it into a valid file.
- Click "Download File" to save the result.

### 2. Generating Files via Database
- Open the "Generate Files" tab.
- Select "Option B: Use Database".
- Enter a SQL query that fetches columns mapping to: `routing_number`, `account_number`, `amount`, `recipient_name`.
  - Example: `SELECT * FROM payments WHERE status='pending'`
- Click "Pull from Database". The app will map DB rows to the file format automatically.

### 3. Validating Files
- Open the "Validate Files" tab.
- Upload an existing `.ach`, `.vcf`, or `.txt` file.
- The system will detect the file type and run compliance checks.
- Errors (e.g., "Invalid length on Line 5") will be displayed in the UI.

## 🧠 Training the Model

The repository includes a simulated inference engine. To train the SLM on your proprietary data:

1. Prepare your dataset in `data/sample_ach.json` as a list of transaction objects.
2. Modify `models/train.py` to load your specific data logic.
3. Run the training script:

```bash
python models/train.py
```

The model weights (`slm_v1.pt`) will be saved to `models/checkpoints/`.
Restart the Streamlit app to use your newly trained model.

## 📁 Project Structure

```
financial-file-gen/
├── app.py                  # Main Streamlit UI Entry point
├── requirements.txt        # Python dependencies
├── .env                    # Configuration (DB URL)
│
├── config/
│   └── settings.py         # Global constants and env loader
│
├── core/
│   ├── ach_handler.py      # ACH formatting and validation logic
│   ├── vcf_handler.py      # VCF formatting and validation logic
│   └── db_connector.py     # Database connection logic (SQLAlchemy)
│
├── models/
│   ├── slm_architecture.py # PyTorch Model definition
│   ├── inference.py        # Logic to run predictions
│   └── train.py            # Script to train the model
│
└── tests/
    └── test_ach_handler.py # Unit tests for file compliance
```

## ⚡ Model Training Flow

This script checks if a model file already exists. It will skip training unless you explicitly use a `--force` flag.

### How to Use the Updated Flow

#### First Run (Training)
Run the training script from your terminal:

```bash
python models/train.py
```

**Output:** It will create `models/checkpoints/slm_v1.pt` and save the weights.

#### Subsequent Runs (Skipping Training)
Run the command again:

```bash
python models/train.py
```

**Output:** It will print "✅ Trained model found... Skipping training." and exit immediately.

#### Force Retraining
If you change your data or want to retrain, use the force flag:

```bash
python models/train.py --force
```

#### Running the App
```bash
streamlit run app.py
```

The console will confirm that the model was loaded from the file, ensuring your startup is fast because no training happens at runtime.