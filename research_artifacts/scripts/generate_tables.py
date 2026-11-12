import csv,json
from pathlib import Path
R=Path(r'D:\Clarion');P=R/'research_artifacts/processed';T=R/'research_artifacts/tables';T.mkdir(exist_ok=True)
for src,dst in [('pdf_chunk_counts.csv','pdf_corpus_processing_table.csv'),('chunking_configuration_sensitivity.csv','chunking_configuration_table.csv')]: (T/dst).write_bytes((P/src).read_bytes())
data=json.loads((R/'research_artifacts/isolated_pdf_index_2026-11-11/processing_results.json').read_text(encoding='utf-8'));fields=sorted({k for row in data for k in row})
with (P/'isolated_pdf_index_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in data])
(T/'index_alignment_table.csv').write_bytes((P/'isolated_pdf_index_results.csv').read_bytes())
(T/'README.md').write_text('Generated table CSVs are copies of machine-readable processed outputs. Regenerate with `research_artifacts/scripts/generate_tables.py`; do not edit values manually. The sensitivity table contains chunk/tokenizer counts, not retrieval metrics.\n',encoding='utf-8')
