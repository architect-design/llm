from .db_connector import DatabaseManager
from .local_llm import CustomLLM
import json


class TestDataAgent:
    def __init__(self):
        self.db = DatabaseManager("postgresql://postgres:password@localhost:5432/testdb")  # Or load from config
        self.llm = CustomLLM()

    def run(self, feature_name, table_list):
        print(f"\n--- Starting Agent for Feature: {feature_name} ---")

        # Step 1: Introspect Database
        print("1. Analyzing Database Schema...")
        schema = self.db.get_schema_info(table_list)
        if not schema:
            print("Error: No valid tables found.")
            return

        schema_str = json.dumps(schema, indent=2)
        print(f"Found Schema:\n{schema_str}")

        # Step 2: Generate Data via Local Model
        print("\n2. Generating Test Data via Custom AI Model...")
        generated_sql = self.llm.generate_sql(schema_str, feature_name)

        print("-" * 30)
        print("Generated SQL:")
        print(generated_sql)
        print("-" * 30)

        # Step 3: Execute
        print("\n3. Executing SQL...")
        result = self.db.execute_sql_block(generated_sql)

        if result['success']:
            print("✅ SUCCESS: Test data generated and inserted.")
        else:
            print(f"❌ FAILED: {result['message']}")
            print("Tip: Check your model output or schema constraints.")
