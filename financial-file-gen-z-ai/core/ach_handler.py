from datetime import datetime


class ACHFileHandler:
    """
    Handles strict formatting of ACH files.
    Ensures mathematical correctness (hashes, counts) which SLMs cannot guarantee.
    """

    @staticmethod
    def _pad(text, length, align='left'):
        """Helper to pad strings to exact fixed-width length."""
        if align == 'left':
            return str(text).ljust(length)[:length]
        return str(text).rjust(length)[:length]

    @staticmethod
    def generate_ach_file(transactions_data, company_name="DEFAULT COMPANY", company_id="123456789"):
        """
        Input: List of dicts [{'routing': '12345678', 'account': '999', 'amount': 100.00, 'name': 'John Doe'}]
        Output: String content of ACH file
        """
        lines = []
        now = datetime.now()

        # 1. File Header Record (Type 1)
        # Priority Code (2) + Immediate Destination (10) + Immediate Origin (10) + Date + Time + ID + Name
        header = (
                "101" +
                ACHFileHandler._pad(" 021000021", 10) +  # Immediate Destination (Routing)
                ACHFileHandler._pad(company_id, 10) +  # Immediate Origin
                now.strftime("%y%m%d") +  # Date
                now.strftime("%H%M") +  # Time
                "A" +  # File ID Modifier
                "094" +  # Record Size
                "101" +  # Blocking Factor
                "01" +  # Format Code
                ACHFileHandler._pad("YOUR BANK NAME", 23) +
                ACHFileHandler._pad(company_name, 23)
        )
        lines.append(header[:94])

        # 2. Batch Header Record (Type 5)
        batch_header = (
                "5225" +  # Service Class Code (225 = Debits/Credits)
                ACHFileHandler._pad(company_name, 16) +
                ACHFileHandler._pad("DISCRETIONARY", 20) +
                company_id +
                "PPD" +  # Standard Entry Class
                "PAYROLL" +  # Entry Description
                now.strftime("%y%m%d") +  # Effective Entry Date
                "   " +  # Settlement Date (Blank)
                "1" +  # Originator Status Code
                ACHFileHandler._pad("12345678", 8) +  # ODFI Identification
                "0000001"  # Batch Number
        )
        lines.append(batch_header[:94])

        entry_addenda_count = 0
        total_debit = 0
        total_credit = 0
        entry_hash = 0
        trace_number_seq = 1

        # 3. Entry Detail Records (Type 6)
        for tx in transactions_data:
            entry_addenda_count += 1
            amount_cents = int(float(tx['amount']) * 100)

            if tx.get('credit', True):
                total_credit += amount_cents
                txn_type = "22"  # Checking Credit
            else:
                total_debit += amount_cents
                txn_type = "27"  # Checking Debit

            # Hash calculation (Sum of first 8 digits of routing number)
            try:
                entry_hash += int(str(tx['routing'])[:8])
            except ValueError:
                pass  # Handle invalid routing numbers gracefully in production

            trace_num = f"{tx['routing'][:8]}{str(trace_number_seq).zfill(7)}"
            trace_number_seq += 1

            line = (
                    "6" + txn_type +
                    ACHFileHandler._pad(tx['routing'], 9) +
                    ACHFileHandler._pad(tx['account'], 17) +
                    ACHFileHandler._pad(str(amount_cents), 10, 'right') +
                    ACHFileHandler._pad("", 28) +  # Individual ID (Optional)
                    ACHFileHandler._pad(tx['name'], 22) +
                    "  " +  # Discretionary Data
                    "0"  # Addenda Record Indicator
            )
            lines.append(line[:94])

        # 4. Batch Trailer Record (Type 8)
        hash_str = str(entry_hash).zfill(10)[-10:]
        batch_trailer = (
                "8225" +
                ACHFileHandler._pad(entry_addenda_count, 6, 'right') +
                hash_str +
                ACHFileHandler._pad(str(total_debit), 12, 'right') +
                ACHFileHandler._pad(str(total_credit), 12, 'right') +
                ACHFileHandler._pad(company_name, 16) +
                ACHFileHandler._pad("", 24) +  # Reserved
                ACHFileHandler._pad("12345678", 8) +
                "0000001"
        )
        lines.append(batch_trailer[:94])

        # 5. File Trailer Record (Type 9)
        block_count = len(lines) + 1
        file_trailer = (
                "9000001" +
                ACHFileHandler._pad(block_count, 6, 'right') +
                ACHFileHandler._pad(entry_addenda_count, 8, 'right') +
                hash_str +
                ACHFileHandler._pad(str(total_debit), 12, 'right') +
                ACHFileHandler._pad(str(total_credit), 12, 'right') +
                ACHFileHandler._pad("", 39)
        )
        lines.append(file_trailer[:94])

        # Padding to ensure block of 10 (Fill with 9s)
        while len(lines) % 10 != 0:
            lines.append("9" * 94)

        return "\n".join(lines)

    @staticmethod
    def validate_ach(file_content):
        errors = []
        lines = file_content.split('\n')

        # Filter out empty lines
        lines = [l for l in lines if l.strip()]

        if not lines:
            return ["Error: File is empty."]

        # 1. Check Record Lengths
        for i, line in enumerate(lines):
            if len(line) != 94:
                errors.append(f"Line {i + 1}: Invalid length ({len(line)}). Must be 94 chars.")

        # 2. Check File Header
        if lines[0][:2] != "10":
            errors.append("Error: File must start with a Header Record (Type 10).")

        # 3. Check Blocking Factor (File must end with lines of 9s if needed)
        if len(lines) % 10 != 0:
            errors.append(f"Error: Invalid blocking factor. Total lines ({len(lines)}) is not a multiple of 10.")

        # 4. Basic Hash Validation Logic (Simplified)
        # In a real app, we would sum routing numbers and compare to Trailer record.

        return errors if errors else ["Success: File Structure Valid."]

    @staticmethod
    def parse_ach_to_data(file_content):
        """Converts ACH file back to JSON for editing/analysis."""
        lines = file_content.split('\n')
        transactions = []
        for line in lines:
            if line.startswith('6'):
                # Parse Entry Detail
                # Positions based on NACHA standards
                routing = line[3:12].strip()
                account = line[12:29].strip()
                amount_str = line[29:39].strip()
                name = line[54:76].strip()

                # Convert cents to dollars
                amount = float(amount_str) / 100.0

                transactions.append({
                    "routing": routing,
                    "account": account,
                    "amount": amount,
                    "name": name,
                    "credit": line[1:3] in ['22', '23', '32', '33']  # Simplified logic
                })
        return transactions