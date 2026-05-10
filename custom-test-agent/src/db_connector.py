from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


class DatabaseManager:
    def __init__(self, db_uri):
        self.engine = create_engine(db_uri)

    def get_schema_info(self, table_names):
        """Introspects the database to get column details."""
        inspector = inspect(self.engine)
        schema_data = {}

        for table in table_names:
            if table in inspector.get_table_names():
                columns = inspector.get_columns(table)
                schema_data[table] = [
                    {"name": col['name'], "type": str(col['type'])}
                    for col in columns
                ]
            else:
                print(f"Warning: Table {table} not found.")
        return schema_data

    def execute_sql_block(self, sql_script):
        """Executes a block of SQL statements within a transaction."""
        conn = self.engine.connect()
        transaction = conn.begin()
        results = {"success": False, "message": ""}

        try:
            # Split script into individual statements
            statements = [s.strip() for s in sql_script.split(';') if s.strip()]

            for stmt in statements:
                conn.execute(text(stmt))

            transaction.commit()
            results["success"] = True
            results["message"] = "Execution committed successfully."

        except SQLAlchemyError as e:
            transaction.rollback()
            results["message"] = f"SQL Error: {str(e)}. Transaction rolled back."
        finally:
            conn.close()

        return results