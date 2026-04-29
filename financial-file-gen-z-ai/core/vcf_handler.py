from datetime import datetime


class VCFFileHandler:
    """
    Handles strict formatting of VISA VCF-like files.
    Note: VCF specs are proprietary. This is a simplified Fixed-Width representation
    for demonstration of the Hybrid AI approach.
    """

    @staticmethod
    def _pad(text, length, align='left'):
        if align == 'left':
            return str(text).ljust(length)[:length]
        return str(text).rjust(length)[:length]

    @staticmethod
    def generate_vcf_file(transactions_data, merchant_name="TEST MERCHANT"):
        lines = []
        now = datetime.now()

        # 1. File Header (V1)
        header = (
                "V1" +
                "VISA CLEARING FILE   " +  # Record Type Name
                now.strftime("%Y%m%d") +
                VCFFileHandler._pad(merchant_name, 30) +
                "0001"  # File Sequence
        )
        lines.append(header[:120])  # VCF records are often 120 bytes

        total_amount = 0
        count = 0

        # 2. Transaction Records (V2)
        for tx in transactions_data:
            count += 1
            amount_cents = int(float(tx['amount']) * 100)
            total_amount += amount_cents

            line = (
                    "V2" +
                    VCFFileHandler._pad(tx['card_number'], 19) +
                    now.strftime("%Y%m%d") +  # Transaction Date
                    VCFFileHandler._pad(str(amount_cents), 12, 'right') +
                    VCFFileHandler._pad(tx['name'], 25) +
                    VCFFileHandler._pad(tx['merchant_code'], 15) +
                    VCFFileHandler._pad("", 41)  # Padding/Fillers
            )
            lines.append(line[:120])

        # 3. File Trailer (V9)
        trailer = (
                "V9" +
                VCFFileHandler._pad(count, 8, 'right') +
                VCFFileHandler._pad(str(total_amount), 12, 'right') +
                VCFFileHandler._pad("", 98)  # Fillers
        )
        lines.append(trailer[:120])

        return "\n".join(lines)

    @staticmethod
    def validate_vcf(file_content):
        errors = []
        lines = file_content.split('\n')
        lines = [l for l in lines if l.strip()]

        if not lines:
            return ["Error: File is empty."]

        if not lines[0].startswith("V1"):
            errors.append("Error: VCF file must start with V1 Header.")

        for i, line in enumerate(lines):
            if len(line) != 120:
                errors.append(f"Line {i + 1}: Invalid length. VCF requires 120 chars.")

        return errors if errors else ["Success: VCF File Structure Valid."]