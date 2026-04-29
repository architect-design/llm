import json
import os


def convert_ach_to_json(ach_file_path, output_json_path):
    """
    Parses a standard NACHA ACH file and extracts transaction details
    into a JSON format suitable for model training.
    """

    if not os.path.exists(ach_file_path):
        print(f"Error: File not found at {ach_file_path}")
        return

    transactions = []

    print(f"Processing {ach_file_path}...")

    with open(ach_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Ignore empty lines or padding lines (usually lines of 999...)
        if not line or line.startswith('9'):
            continue

        # Record Type '6' is the Entry Detail Record
        if line.startswith('6'):
            try:
                # --- NACHA Field Mapping (Fixed Width) ---

                # Transaction Code (Positions 2-3): Determines Credit/Debit
                trans_code = line[1:3].strip()

                # Routing Number (Positions 4-11 + 12): 8 digits + 1 check digit
                routing_number = line[3:12].strip()

                # Account Number (Positions 13-29): 17 characters
                account_number = line[12:29].strip()

                # Amount (Positions 30-39): 10 digits (in cents)
                amount_str = line[29:39].strip()

                # Individual Name (Positions 55-76): 22 characters
                individual_name = line[54:76].strip()

                # --- Data Transformation ---

                # Convert amount string to float dollars
                # NACHA stores amounts in cents (e.g., 10000 = $100.00)
                if amount_str.isdigit():
                    amount_val = int(amount_str) / 100.0
                else:
                    amount_val = 0.0

                # Determine if Credit or Debit based on Transaction Code
                # Standard codes: 22, 23, 32, 33 = Credit | 27, 28, 37, 38 = Debit
                credit_codes = ['22', '23', '32', '33']
                is_credit = trans_code in credit_codes

                # Create the JSON object
                tx_data = {
                    "routing": routing_number,
                    "account": account_number,
                    "amount": amount_val,
                    "name": individual_name,
                    "credit": is_credit
                }

                transactions.append(tx_data)

            except IndexError:
                print(f"Skipping malformed line (length {len(line)}): {line[:20]}...")
            except Exception as e:
                print(f"Error parsing line: {e}")

    # Write to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(transactions, f, indent=4)

    print(f"Successfully extracted {len(transactions)} transactions.")
    print(f"Saved to {output_json_path}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change this to the path of your actual ACH file
    INPUT_ACH_FILE = "data/my_real_file.ach"

    # This matches the path expected by the training script
    OUTPUT_JSON_FILE = "data/sample_ach.json"

    convert_ach_to_json(INPUT_ACH_FILE, OUTPUT_JSON_FILE)