import json,re
from pathlib import Path
x=json.loads(Path('research_artifacts/raw/clarion_extraction.json').read_text(encoding='utf-8')); p=json.loads(Path('research_artifacts/raw/pdf_extraction.json').read_text(encoding='utf-8')); by={r['filename']:r for r in p}
for r in x:
 text=r['text'];doi=re.findall(r'10\.\d{4,9}/[^\s<>\]]+',text,re.I);urls=re.findall(r'https?://[^\s<>\]]+',text,re.I)
 print(r['filename'],'DOI',doi[:2],'URL',urls[:2], 'meta',by[r['filename']]['metadata'])
