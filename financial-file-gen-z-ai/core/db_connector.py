import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import settings


class DatabaseConnector:
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)

    def fetch_transaction_data(self, query, file_type='ACH'):
        """
        Fetch actual data from DB to be used by the Formatter
        """
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)

            # Convert DB rows to the format needed by Handlers
            transactions = []
            for _, row in df.iterrows():
                # Generic mapping - in real projects, map specific columns
                tx = {
                    "routing": str(row.get('routing_number', '000000000')),
                    "account": str(row.get('account_number', '000000000')),
                    "amount": float(row.get('amount', 0)),
                    "name": str(row.get('recipient_name', 'UNKNOWN')),
                    "card_number": str(row.get('card_number', '4111111111111111')),
                    "merchant_code": str(row.get('merchant_code', 'MISC')),
                    "credit": True
                }
                transactions.append(tx)
            return transactions
        except Exception as e:
            print(f"Database Error: {e}")
            return []