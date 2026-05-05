"""Quick test for filename extraction improvements"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from temporal import TemporalEntityExtractor

extractor = TemporalEntityExtractor(use_spacy=False)

test_filenames = [
    "HBL_CarLoan_SOBC_2025_Jul-Dec_-_English_.pdf",
    "HBL_Car_Loan_KFS_Jul-Dec_2025_(English).pdf",
    "HBL_CreditCard-KFS_(Eng)_16OCT2025.pdf",
    "HBL_PersonalLoan_SOBC_Jul_to_Dec_2025.pdf",
    "HBL_Individual_Account_Opening_Form_-_Conventional_22_July.pdf",
    "UBL_Credit_Card_KFS-Change-in-SOC-July-2025-to-Dec-2025.pdf",
]

print("Testing filename extraction:\n")
for filename in test_filenames:
    entities = extractor.extract_from_filename(filename)
    print(f"📄 {filename}")
    if entities:
        for entity in entities:
            print(f"   ✅ {entity.text} ({entity.temporal_type.value}): {entity.start_date} to {entity.end_date}")
    else:
        print(f"   ❌ No entities extracted")
    print()
