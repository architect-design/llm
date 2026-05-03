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
# CHROMADB ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_STORE = os.path.join(BASE_DIR, 'db', 'chromadb_store')


def _get_chroma_gen():
    from data.chroma_ach_generator import ChromaACHGenerator
    return ChromaACHGenerator(CHROMA_STORE)


def _get_chroma_repo():
    from db.ach_repository import ACHRepository
    return ACHRepository(CHROMA_STORE)


@app.route('/api/chroma/health', methods=['GET'])
def chroma_health():
    """Return ChromaDB health — collection sizes and store path."""
    try:
        from db.chroma_client import get_client
        client = get_client(CHROMA_STORE)
        return jsonify(client.health())
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/api/chroma/test', methods=['POST'])
def chroma_test():
    """Test ChromaDB connectivity and return full health info."""
    try:
        from db.chroma_client import ChromaDBClient, get_client
        ChromaDBClient.reset()
        client = get_client(CHROMA_STORE)
        ok, msg = client.test_connection()
        return jsonify({"success": ok, "message": msg, "health": client.health()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/chroma/seed', methods=['POST'])
def chroma_seed():
    """Seed ChromaDB with synthetic transactions for demo / testing."""
    data = request.get_json() or {}
    n        = int(data.get('n', 50))
    sec_code = data.get('sec_code')
    try:
        gen   = _get_chroma_gen()
        added = gen.seed_transactions(n=n, sec_code=sec_code)
        return jsonify({
            "success": True,
            "added":   added,
            "message": f"Seeded {added} transactions into ChromaDB",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chroma/generate', methods=['POST'])
def chroma_generate_ach():
    """Generate an ACH file from ChromaDB PENDING transactions."""
    data = request.get_json() or {}
    try:
        gen    = _get_chroma_gen()
        result = gen.generate(
            company_id       = data.get('company_id'),
            sec_code         = data.get('sec_code'),
            effective_date   = data.get('effective_date'),
            max_transactions = int(data.get('max_transactions', 500)),
            save_audit       = bool(data.get('save_audit', True)),
            save_corpus      = bool(data.get('save_corpus', True)),
        )
        return jsonify({
            "content":           result["content"],
            "file_name":         result["file_name"],
            "batch_count":       result["batch_count"],
            "entry_count":       result["entry_count"],
            "total_debit":       result["total_debit"]  / 100,
            "total_credit":      result["total_credit"] / 100,
            "source":            result["source"],
            "file_id":           result["file_id"],
            "transaction_count": len(result["transactions"]),
            "generated_at":      datetime.now().isoformat(),
            "file_type":         "ACH",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chroma/companies', methods=['GET'])
def chroma_companies():
    """List companies stored in ach_companies collection."""
    sec = request.args.get('sec_code')
    try:
        repo      = _get_chroma_repo()
        companies = repo.get_companies(sec_code=sec)
        return jsonify([{
            "company_id":         c.company_id,
            "company_name":       c.company_name.strip(),
            "company_id_number":  c.company_id_number.strip(),
            "company_entry_desc": c.company_entry_desc.strip(),
            "sec_code":           c.sec_code,
            "service_class_code": c.service_class_code,
        } for c in companies])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chroma/odfi', methods=['GET'])
def chroma_odfi():
    """List all ODFI configurations."""
    try:
        repo  = _get_chroma_repo()
        odfis = repo.list_odfi()
        return jsonify([{
            "id":               o.id,
            "routing_number":   o.routing_number,
            "bank_name":        o.bank_name.strip(),
            "immediate_dest":   o.immediate_dest,
            "immediate_origin": o.immediate_origin,
        } for o in odfis])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chroma/corpus/stats', methods=['GET'])
def chroma_corpus_stats():
    """Return training corpus statistics from ChromaDB."""
    try:
        repo  = _get_chroma_repo()
        stats = repo.corpus_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chroma/file-log', methods=['GET'])
def chroma_file_log():
    """Return recent file generation audit log entries."""
    limit = int(request.args.get('limit', 20))
    try:
        repo    = _get_chroma_repo()
        entries = repo.get_file_log(limit=limit)
        return jsonify(entries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chroma/train', methods=['POST'])
def chroma_train():
    """Start ChromaDB-aware ACH SLM training."""
    global training_state
    if training_state["running"]:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json() or {}
    cfg  = {
        "model_config":       data.get("model_config",       "nano"),
        "n_corpus_files":     int(data.get("n_corpus_files",  200)),
        "n_chroma_files":     int(data.get("n_chroma_files",   50)),
        "n_synthetic_files":  int(data.get("n_synthetic_files", 50)),
        "n_val_files":        int(data.get("n_val_files",      30)),
        "max_epochs":         int(data.get("max_epochs",        5)),
        "batch_size":         int(data.get("batch_size",        4)),
        "learning_rate":      float(data.get("learning_rate",  3e-4)),
        "save_dir":           app.config["MODEL_DIR"],
    }

    training_state = {
        "running":    True,
        "file_type":  "ACH",
        "message":    "Initialising ChromaDB-aware training...",
        "progress":   0,
        "result":     None,
        "error":      None,
        "started_at": datetime.now().isoformat(),
        "source":     "chromadb",
    }

    def run():
        global training_state
        try:
            from data.chroma_trainer import ChromaAwareTrainer
            trainer = ChromaAwareTrainer(CHROMA_STORE, cfg)

            def cb(msg, p):
                training_state["message"]  = msg
                training_state["progress"] = p or training_state["progress"]

            result = trainer.train(callback=cb)
            training_state["result"]   = result
            training_state["message"]  = (
                f"ChromaDB training complete! val_loss={result['best_val_loss']:.4f}"
            )
            training_state["progress"] = 100
        except Exception as e:
            training_state["error"]   = str(e)
            training_state["message"] = f"ChromaDB training failed: {e}"
        finally:
            training_state["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "source": "chromadb", "config": cfg})



# ══════════════════════════════════════════════════════════════════════════════
# VCF — MULTI-CATEGORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _vcf_gen():
    from data.vcf.vcf_generator import VCFGenerator
    return VCFGenerator()

def _vcf_repo():
    from db.vcf_repository import VCFRepository
    return VCFRepository(CHROMA_STORE)


@app.route('/api/vcf/categories', methods=['GET'])
def vcf_categories():
    """Return all supported VCF categories with their transaction codes."""
    from data.vcf.vcf_models import VCFCategory, TRANSACTION_CODES, MCC_BY_CATEGORY
    result = {}
    for cat in VCFCategory:
        codes = TRANSACTION_CODES.get(cat, {})
        mccs  = [(m[0], m[1]) for m in MCC_BY_CATEGORY.get(cat, [])][:5]
        result[cat.value] = {
            "description":       cat.value.replace("_"," ").title(),
            "transaction_codes": codes,
            "sample_mccs":       mccs,
        }
    return jsonify(result)


@app.route('/api/vcf/generate', methods=['POST'])
def vcf_generate():
    """Generate a VCF file — mixed or single category, with audit + corpus saving."""
    data = request.get_json() or {}
    try:
        from data.vcf.vcf_models import VCFCategory
        cat_names  = data.get('categories', [])
        categories = None
        if cat_names:
            try:
                categories = [VCFCategory[c.upper()] for c in cat_names]
            except KeyError as ke:
                return jsonify({"error": f"Unknown category: {ke}"}), 400

        gen     = _vcf_gen()
        content = gen.generate_file(
            num_transactions = int(data.get('num_transactions', 20)),
            categories       = categories,
            force_approved   = bool(data.get('force_approved', False)),
        )

        stats = {}
        try:
            from validators.vcf_validator import VCFValidator
            rpt   = VCFValidator().validate(content)
            stats = rpt.statistics
        except Exception:
            pass

        # Persist to ChromaDB
        repo = _vcf_repo()
        file_name = f"VCF_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vcf"
        cats_used = list(stats.get('category_distribution', {}).keys())
        meta = {
            "transaction_count": stats.get("transaction_count", 0),
            "total_amount":      stats.get("total_amount", 0.0),
            "categories":        ",".join(cats_used),
            "acquirer_bin":      content.split("|")[2] if "|" in content else "",
            "generation_method": "VCF_GENERATOR",
            "modifier":          "A",
        }
        file_id = None
        if data.get('save_audit', True):
            file_id = repo.log_vcf_file(file_name, content, meta)
        if data.get('save_corpus', True):
            import random
            repo.save_vcf_corpus(
                content,
                category   = cats_used[0] if cats_used else "MIXED",
                split      = "TRAIN" if random.random() < 0.85 else "VAL",
                file_log_id= file_id or "",
            )

        return jsonify({
            "content":           content,
            "file_name":         file_name,
            "file_type":         "VCF",
            "file_id":           file_id,
            "transaction_count": stats.get("transaction_count", 0),
            "total_amount":      stats.get("total_amount", 0.0),
            "approved_count":    stats.get("approved_count", 0),
            "declined_count":    stats.get("declined_count", 0),
            "approval_rate_pct": stats.get("approval_rate_pct", 0.0),
            "category_distribution": stats.get("category_distribution", {}),
            "generated_at":      datetime.now().isoformat(),
            "method":            "vcf-generator",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vcf/generate/category', methods=['POST'])
def vcf_generate_category():
    """Generate a VCF file for a single specific category."""
    data = request.get_json() or {}
    cat_name = data.get('category','PURCHASE').upper()
    try:
        from data.vcf.vcf_models import VCFCategory
        cat = VCFCategory[cat_name]
    except KeyError:
        return jsonify({"error": f"Unknown category '{cat_name}'"}), 400

    try:
        gen     = _vcf_gen()
        content = gen.generate_by_category(cat, n=int(data.get('n', 20)))
        return jsonify({
            "content":    content,
            "category":   cat.value,
            "file_type":  "VCF",
            "line_count": len(content.splitlines()),
            "char_count": len(content),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vcf/validate', methods=['POST'])
def vcf_validate_endpoint():
    """Validate a VCF file and return category-aware report."""
    data    = request.get_json() or {}
    content = data.get('content','').strip()
    if not content:
        return jsonify({"error": "No content provided"}), 400
    try:
        from validators.vcf_validator import VCFValidator
        rpt = VCFValidator().validate(content)
        result = rpt.to_dict()
        # Save validation result to ChromaDB
        repo = _vcf_repo()
        repo.log_vcf_file(
            data.get('file_name','vcf_validated.vcf'),
            content[:500],  # snippet only
            {"transaction_count": rpt.statistics.get('transaction_count',0),
             "total_amount": rpt.statistics.get('total_amount', 0.0),
             "categories": str(rpt.statistics.get('category_distribution',{})),
             "generation_method": "VALIDATED"},
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vcf/corpus/stats', methods=['GET'])
def vcf_corpus_stats():
    """Return VCF training corpus statistics."""
    try:
        repo = _vcf_repo()
        return jsonify({**repo.vcf_corpus_stats(), **repo.health()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vcf/file-log', methods=['GET'])
def vcf_file_log():
    """Return recent VCF file generation audit entries."""
    limit = int(request.args.get('limit', 10))
    try:
        repo = _vcf_repo()
        return jsonify(repo.get_file_log(limit=limit))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  FinSLM Web Application")
    print("  ACH NACHA & VISA VCF Generator/Validator")
    print("  ChromaDB store:", CHROMA_STORE)
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
