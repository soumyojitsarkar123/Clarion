import csv, hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
source = root / 'research_artifacts' / 'processed' / 'pdf_source_register.json'
out = root / 'research' / 'data' / 'pdf_source_register.csv'
records = json.loads(source.read_text(encoding='utf-8'))
fields = list(records[0]) + ['sha256_current', 'file_present', 'size_matches_prior']
with out.open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for row in records:
        p = root / 'docs' / row['filename']
        exists = p.is_file()
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if exists else ''
        w.writerow({**row, 'sha256_current': digest, 'file_present': exists,
                    'size_matches_prior': exists and p.stat().st_size == row['size_bytes']})
print(f'wrote {out} records={len(records)} present={sum((root / "docs" / r["filename"]).is_file() for r in records)}')
