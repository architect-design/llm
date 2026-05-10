import argparse
from src.agent import TestDataAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Test Data Agent")
    parser.add_argument("--feature", type=str, required=True, help="Feature name to test")
    parser.add_argument("--tables", type=str, required=True, help="Comma-separated list of tables")

    args = parser.parse_args()

    # Clean input
    table_list = [t.strip() for t in args.tables.split(',')]

    # Initialize and Run Agent
    agent = TestDataAgent()
    agent.run(args.feature, table_list)