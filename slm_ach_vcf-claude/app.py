"""
Financial SLM Web Application
Flask app providing UI for ACH/VCF file generation, validation, and model training
"""

import os
import sys
import json
import time
import threading
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from validators.ach_validator import ACHValidator
from validators.vcf_validator import VCFValidator
from data.generator import ACHGenerator, VCFGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MODEL_DIR'] = os.path.join(BASE_DIR, 'trained_models')
app.config['STATUS_FILE'] = os.path.join(BASE_DIR, 'training_status.json')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

# Global training state
training_state = {
    "running": False,
    "file_type": None,
    "message": "Idle",
    "progress": 0,
    "result": None,
    "error": None,
    "started_at": None,
}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/validate', methods=['POST'])
def validate_file():
    """Validate uploaded ACH or VCF file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_type = request.form.get('file_type', '').upper()

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file_type not in ('ACH', 'VCF'):
        return jsonify({"error": "file_type must be ACH or VCF"}), 400

    try:
        content = file.read().decode('utf-8', errors='replace')

        if file_type == 'ACH':
            validator = ACHValidator()
            report = validator.validate(content)
        else:
            validator = VCFValidator()
            report = validator.validate(content)

        result = report.to_dict()
        result['file_name'] = secure_filename(file.filename)
        result['file_type'] = file_type
        result['file_size'] = len(content)
        result['validated_at'] = datetime.now().isoformat()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Validation error: {str(e)}"}), 500


@app.route('/api/generate', methods=['POST'])
def generate_file():
    """Generate a synthetic ACH or VCF file"""
    data = request.get_json() or {}
    file_type = data.get('file_type', 'ACH').upper()
    use_model = data.get('use_model', False)

    if file_type not in ('ACH', 'VCF'):
        return jsonify({"error": "file_type must be ACH or VCF"}), 400

    try:
        if use_model:
            # Use trained SLM for generation
            model_path = os.path.join(app.config['MODEL_DIR'],
                                       f"{file_type.lower()}_model.pt" if os.path.exists(os.path.join(app.config["MODEL_DIR"], f"{file_type.lower()}_model.pt")) else f"{file_type.lower()}_model.pkl")
            if not os.path.exists(model_path):
                return jsonify({
                    "error": f"No trained {file_type} model found. Please train the model first.",
                    "suggestion": "Use the Rule-based generator or train the model via the Training tab."
                }), 404

            try:
                from trainer import FileGenerator
                gen = FileGenerator(file_type, app.config['MODEL_DIR'])
                temperature = float(data.get('temperature', 0.8))
                max_tokens = int(data.get('max_tokens', 1500))
                content = gen.generate_with_seed(temperature=temperature, max_tokens=max_tokens)
            except Exception as model_err:
                return jsonify({"error": f"Model generation error: {str(model_err)}"}), 500

        else:
            # Use rule-based generator (always available)
            if file_type == 'ACH':
                gen = ACHGenerator()
                content = gen.generate_file(
                    num_batches=int(data.get('num_batches', 2)),
                    entries_per_batch=int(data.get('entries_per_batch', 5)),
                    sec_code=data.get('sec_code', None),
                )
            else:
                gen = VCFGenerator()
                content = gen.generate_file(
                    num_transactions=int(data.get('num_transactions', 20)),
                )

        return jsonify({
            "content": content,
            "file_type": file_type,
            "generated_at": datetime.now().isoformat(),
            "method": "model" if use_model else "rule-based",
            "char_count": len(content),
            "line_count": len(content.splitlines()),
        })

    except Exception as e:
        return jsonify({"error": f"Generation error: {str(e)}"}), 500


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training in a background thread"""
    global training_state

    if training_state["running"]:
        return jsonify({
            "error": "Training already in progress",
            "current": training_state
        }), 409

    data = request.get_json() or {}
    file_type = data.get('file_type', 'ACH').upper()

    if file_type not in ('ACH', 'VCF'):
        return jsonify({"error": "file_type must be ACH or VCF"}), 400

    config = {
        "model_config": data.get('model_config', 'nano'),
        "n_training_files": int(data.get('n_files', 200)),
        "n_val_files": int(data.get('n_val_files', 30)),
        "block_size": 512,
        "batch_size": int(data.get('batch_size', 4)),
        "max_epochs": int(data.get('max_epochs', 5)),
        "learning_rate": float(data.get('learning_rate', 3e-4)),
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "eval_every": 50,
        "save_dir": app.config['MODEL_DIR'],
    }

    training_state = {
        "running": True,
        "file_type": file_type,
        "message": "Initializing training...",
        "progress": 0,
        "result": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
    }

    def run_training():
        global training_state
        try:
            from trainer import Trainer
            trainer = Trainer(file_type, config)

            def progress_callback(msg, progress):
                training_state["message"] = msg
                training_state["progress"] = progress or training_state["progress"]

            result = trainer.train(callback=progress_callback)
            training_state["result"] = result
            training_state["message"] = f"Training complete! Best val loss: {result['best_val_loss']:.4f}"
            training_state["progress"] = 100

        except Exception as e:
            training_state["error"] = str(e)
            training_state["message"] = f"Training failed: {str(e)}"
        finally:
            training_state["running"] = False

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({"status": "started", "file_type": file_type, "config": config})


@app.route('/api/train/status', methods=['GET'])
def training_status():
    """Get current training status"""
    return jsonify(training_state)


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available trained models"""
    models = {}
    for ft in ('ach', 'vcf'):
        pt_path = os.path.join(app.config['MODEL_DIR'], f"{ft}_model.pt")
        pkl_path = os.path.join(app.config['MODEL_DIR'], f"{ft}_model.pkl")
        model_path = pt_path if os.path.exists(pt_path) else pkl_path
        models[ft.upper()] = {
            "available": os.path.exists(model_path),
            "path": model_path,
            "backend": "transformer" if os.path.exists(pt_path) else "bigram",
            "size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
            "modified": datetime.fromtimestamp(
                os.path.getmtime(model_path)).isoformat()
            if os.path.exists(model_path) else None,
        }
    return jsonify(models)


@app.route('/api/validate/text', methods=['POST'])
def validate_text():
    """Validate raw text content (not file upload)"""
    data = request.get_json() or {}
    content = data.get('content', '')
    file_type = data.get('file_type', 'ACH').upper()

    if not content.strip():
        return jsonify({"error": "No content provided"}), 400

    try:
        if file_type == 'ACH':
            report = ACHValidator().validate(content)
        else:
            report = VCFValidator().validate(content)

        return jsonify({**report.to_dict(), "file_type": file_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/download', methods=['POST'])
def download_file():
    """Download generated content as a file"""
    data = request.get_json() or {}
    content = data.get('content', '')
    file_type = data.get('file_type', 'ACH').upper()
    filename = data.get('filename', f'{file_type.lower()}_generated.txt')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        tmp_path = f.name

    return send_file(tmp_path, as_attachment=True, download_name=filename,
                     mimetype='text/plain')


@app.route('/api/spec', methods=['GET'])
def get_spec():
    """Return specification summary for ACH and VCF formats"""
    return jsonify({
        "ACH": {
            "full_name": "NACHA ACH (Automated Clearing House)",
            "record_length": 94,
            "blocking_factor": 10,
            "record_types": {
                "1": "File Header",
                "5": "Batch Header",
                "6": "Entry Detail",
                "7": "Addenda",
                "8": "Batch Control",
                "9": "File Control"
            },
            "sec_codes": ["PPD", "CCD", "CTX", "WEB", "TEL", "COR", "IAT"],
            "service_class_codes": {"200": "Mixed", "220": "Credits", "225": "Debits"},
        },
        "VCF": {
            "full_name": "VISA VCF (VisaNet Custom File)",
            "format": "Pipe-delimited or fixed-width",
            "record_types": {
                "Header": "File identification and metadata",
                "Transaction": "Individual payment/auth records",
                "Trailer": "File totals and control counts"
            },
            "transaction_codes": {
                "05": "Authorization Request",
                "06": "Financial Transaction",
                "10": "Full Reversal",
                "25": "Chargeback"
            },
            "pan_validation": "Luhn algorithm",
            "currency_standard": "ISO 4217",
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  Financial SLM Web Application")
    print("  ACH NACHA & VISA VCF Generator/Validator")
    print("=" * 60)
    print(f"  Starting server at http://localhost:5000")
    print(f"  Model directory: {app.config['MODEL_DIR']}")
    app.run(debug=True, host='0.0.0.0', port=5000)


# ══════════════════════════════════════════════════════════════════════════════
# ORACLE-SPECIFIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _get_oracle_config(data: dict) -> "OracleConfig":
    from db.oracle_connector import OracleConfig
    cfg = OracleConfig()
    # Allow per-request overrides (connection-test screen)
    for field in ("host","port","service","user","password","schema","mock"):
        if field in data:
            if field == "port":
                setattr(cfg, field, int(data[field]))
            elif field == "mock":
                setattr(cfg, field, str(data[field]).lower() == "true")
            else:
                setattr(cfg, field, str(data[field]))
    return cfg


@app.route('/api/oracle/test', methods=['POST'])
def oracle_test_connection():
    """Test Oracle DB connectivity."""
    data = request.get_json() or {}
    try:
        from db.oracle_connector import OracleConnectionPool
        OracleConnectionPool.reset()          # force reconnect with new params
        cfg = _get_oracle_config(data)
        from db.oracle_connector import get_pool
        pool = get_pool(cfg)
        ok, msg = pool.test_connection()
        health = pool.health()
        return jsonify({"success": ok, "message": msg, "health": health})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/oracle/health', methods=['GET'])
def oracle_health():
    """Return Oracle pool health without making a new connection."""
    try:
        from db.oracle_connector import get_pool
        pool = get_pool()
        return jsonify(pool.health())
    except Exception as e:
        return jsonify({"error": str(e), "mode": "unknown"}), 500


@app.route('/api/oracle/generate', methods=['POST'])
def oracle_generate_ach():
    """Generate ACH file from Oracle pending transactions."""
    data = request.get_json() or {}
    try:
        from db.oracle_connector import OracleConnectionPool
        cfg = _get_oracle_config(data)
        OracleConnectionPool.reset()
        from data.oracle_ach_generator import OracleACHGenerator
        gen = OracleACHGenerator(cfg)
        result = gen.generate(
            company_id      = data.get("company_id"),
            sec_code        = data.get("sec_code"),
            effective_date  = data.get("effective_date"),
            max_transactions= int(data.get("max_transactions", 500)),
            save_audit      = bool(data.get("save_audit", True)),
            save_corpus     = bool(data.get("save_corpus", True)),
        )
        return jsonify({
            "content":      result["content"],
            "file_name":    result["file_name"],
            "batch_count":  result["batch_count"],
            "entry_count":  result["entry_count"],
            "total_debit":  result["total_debit"]  / 100,
            "total_credit": result["total_credit"] / 100,
            "source":       result["source"],
            "file_id":      result["file_id"],
            "transaction_count": len(result["transactions"]),
            "generated_at": datetime.now().isoformat(),
            "file_type":    "ACH",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/oracle/companies', methods=['GET'])
def oracle_companies():
    """List companies from Oracle ACH_COMPANIES table."""
    sec = request.args.get("sec_code")
    try:
        from db.ach_repository import ACHRepository
        repo = ACHRepository()
        companies = repo.get_companies(sec_code=sec)
        return jsonify([{
            "company_id":          c.company_id,
            "company_name":        c.company_name.strip(),
            "company_id_number":   c.company_id_number.strip(),
            "company_entry_desc":  c.company_entry_desc.strip(),
            "sec_code":            c.sec_code,
            "service_class_code":  c.service_class_code,
        } for c in companies])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/oracle/train', methods=['POST'])
def oracle_train():
    """Train the ACH SLM using Oracle data (+ synthetic augmentation)."""
    global training_state
    if training_state["running"]:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json() or {}
    cfg_overrides = {
        "model_config":       data.get("model_config",        "nano"),
        "n_oracle_files":     int(data.get("n_oracle_files",  200)),
        "n_synthetic_files":  int(data.get("n_synthetic_files", 50)),
        "n_val_files":        int(data.get("n_val_files",      30)),
        "max_epochs":         int(data.get("max_epochs",        5)),
        "batch_size":         int(data.get("batch_size",        4)),
        "learning_rate":      float(data.get("learning_rate",  3e-4)),
        "save_dir":           app.config["MODEL_DIR"],
    }

    training_state = {
        "running": True,
        "file_type": "ACH",
        "message": "Initialising Oracle-aware training...",
        "progress": 0,
        "result": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "source": "oracle",
    }

    def run():
        global training_state
        try:
            from data.oracle_trainer import OracleAwareTrainer
            oracle_cfg = _get_oracle_config(data)
            trainer = OracleAwareTrainer(oracle_cfg, cfg_overrides)

            def cb(msg, p):
                training_state["message"]  = msg
                training_state["progress"] = p or training_state["progress"]

            result = trainer.train(callback=cb)
            training_state["result"]  = result
            training_state["message"] = (
                f"Oracle training complete! val_loss={result['best_val_loss']:.4f}"
            )
            training_state["progress"] = 100
        except Exception as e:
            training_state["error"]   = str(e)
            training_state["message"] = f"Oracle training failed: {e}"
        finally:
            training_state["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "source": "oracle", "config": cfg_overrides})

