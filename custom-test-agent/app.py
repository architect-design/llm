from flask import Flask, render_template, request, jsonify
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.db_connector import DatabaseManager
from src.local_llm import CustomLLM
import config

app = Flask(__name__)

# Initialize the heavy LLM once when the server starts
# Note: This might take a few seconds to load into memory
print("Initializing AI Model...")
llm_engine = CustomLLM()


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API Endpoint to generate SQL (Review Mode)."""
    try:
        data = request.json
        db_uri = data.get('db_uri')
        feature_name = data.get('feature_name')
        table_names = data.get('tables', []).split(',')

        # 1. Introspect Schema
        db_manager = DatabaseManager(db_uri)
        schema_info = db_manager.get_schema_info(table_names)

        if not schema_info:
            return jsonify({"status": "error", "message": "Could not read schema. Check table names."})

        # 2. Generate SQL via Local Model
        generated_sql = llm_engine.generate_sql(str(schema_info), feature_name)

        return jsonify({
            "status": "success",
            "schema": schema_info,
            "sql": generated_sql
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/execute', methods=['POST'])
def api_execute():
    """API Endpoint to execute the SQL."""
    try:
        data = request.json
        db_uri = data.get('db_uri')
        sql_script = data.get('sql')

        # Validate SQL doesn't contain destructive commands (Basic Safety)
        forbidden = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
        if any(word in sql_script.upper() for word in forbidden):
            return jsonify({"status": "error",
                            "message": "Safety Restriction: Destructive keywords (DROP/DELETE) are forbidden in UI."})

        db_manager = DatabaseManager(db_uri)
        result = db_manager.execute_sql_block(sql_script)

        if result['success']:
            return jsonify({"status": "success", "message": result['message']})
        else:
            return jsonify({"status": "error", "message": result['message']})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    # Run the app
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)