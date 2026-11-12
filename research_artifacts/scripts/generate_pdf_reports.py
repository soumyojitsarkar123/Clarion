import json,re,csv
from pathlib import Path
R=Path(r'D:\Clarion'); A=R/'research_artifacts'; P=A/'processed'
ex=json.loads((A/'raw/clarion_extraction.json').read_text(encoding='utf-8')); fullmeta={x['filename']:x['metadata'] for x in json.loads((A/'raw/pdf_extraction.json').read_text(encoding='utf-8'))}; counts=json.loads((P/'pdf_chunk_counts.json').read_text(encoding='utf-8')); imgs=json.loads((P/'pdf_image_object_counts.json').read_text(encoding='utf-8')); imap={x['filename']:x.get('image_xobjects') for x in imgs}
cl={
'AI_Book.pdf':('BACKGROUND ONLY','Filename suggests AI, but metadata is absent, text extraction returned zero, and pages 1-2 rendered blank; identity/topic unverified.'),
'DAA FINAL.pdf':('NOT RELEVANT','Design and analysis of algorithms course notes, no Clarion/IR empirical results.'),
'UNIT II — SELF MANAGEMENT SKILLS.pdf':('NOT RELEVANT','Professional self-management and communication slides.'),
'UNIT III — ENTREPRENEURSHIP SKILLS.pdf':('NOT RELEVANT','Entrepreneurship slides, no relevant method/results.'),
'UNIT III — SOFTWARE & AI WORKFLOW.pdf':('BACKGROUND ONLY','Generic software/AI workflow slides, no Clarion results.'),
'UNIT IV Information Retrieval Overview.pdf':('PARTIALLY RELEVANT','Generic IR evaluation procedure and P/R/F1 definitions on pages 39-40; no measured results.'),
'UNIT IV — NEGOTIATION SKILLS.pdf':('NOT RELEVANT','Negotiation and employability slides.'),
'UNIT IV — STORYTELLING, BRANDING & ETHICS.pptx.pdf':('BACKGROUND ONLY','Generic branding/ethics slides; no Clarion technical results.'),
'Unit-2 Data exploration and Vizualization.pdf':('BACKGROUND ONLY','Generic data visualization material, no Clarion results.'),
'Unit-3 Vector Calculus.pdf':('NOT RELEVANT','Mathematics course notes not directly related to Clarion claims.'),
'Unit-4 Advanced OLAP Concepts.pptx.pdf':('NOT RELEVANT','OLAP course slides, no Clarion evidence.')}
rows=[]
for r in ex:
 m=fullmeta.get(r['filename'],{}); year='(not available)'; dt=m.get('CreationDate') or ''; mt=re.search(r'(20\d{2})',str(dt)); year=mt.group(1) if mt else year
 c=next(x for x in counts if x['filename']==r['filename']); rel,why=cl[r['filename']]
 rows.append({'source_id':f"PDF-{len(rows)+1:02}",'filename':r['filename'],'title':m.get('Title') or '(not in metadata)','author':m.get('Author') or '(not in metadata)','year':year,'pages':r.get('page_count'),'image_xobjects':imap.get(r['filename']),'size_bytes':r['size_bytes'],'pdf_type':('mixed text/image' if c['extracted_words'] and (imap.get(r['filename']) or 0)>0 else ('text-based' if c['extracted_words'] else ('image-bearing; no extractable text' if (imap.get(r['filename']) or 0)>0 else 'unextractable; type undetermined'))) ,'publisher':m.get('Publisher') or m.get('Journal') or '(not in metadata)','producer':m.get('Producer') or '(not in metadata)','doi_url':'(not located in extracted text/metadata)','extraction':'success' if c['extracted_words'] else 'zero extracted words','relevance':rel,'reason':why})
(P/'pdf_source_register.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# PDF Source Register','','**Table T1. PDF corpus register.** Source: research_artifacts/processed/pdf_source_register.json.','','Inventory date: 2026-11-11. Clarion PyPDF2 extraction was used. Year is only a year parsed from PDF creation metadata, not asserted publication date. DOI/URL check was limited to available metadata/extracted text. Table and figure extraction was not systematic corpus-wide; IR pages 39-40 were visually inspected.','', '| Source ID | Filename | Title | Year | Pages | PDF type | Relevant content | Research use |','|---|---|---:|---:|---:|---|---|---|']

for x in rows: lines.append(f"| {x['source_id']} | {x['filename']} | {x['title']} | {x['year']} | {x['pages']} | {x['pdf_type']} | {x['relevance']}: {x['reason']} | {x['relevance']} |")
lines+=['','## Bibliographic/extraction details','']
for x in rows: lines.append(f"- **{x['source_id']} {x['filename']}**: author {x['author']}; publisher/journal/conference {x['publisher']}; producer {x['producer']}; DOI/URL {x['doi_url']}; size {x['size_bytes']:,} bytes; text extraction: {x['extraction']}; {x['image_xobjects']} image XObjects. Table/figure extraction not systematically evaluated. Classification: **{x['relevance']}** — {x['reason']}")
lines+=['','No PDF is HIGH RELEVANCE: no source reports Clarion or comparable empirical results. OCR was not performed. `AI_Book.pdf` remains unidentified and unextractable.']
(R/'PDF_SOURCE_REGISTER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
e=[{'evidence_id':'EVD-001','source_pdf':'UNIT IV Information Retrieval Overview.pdf','page':39,'table_figure':'Slide/page 39; no numbered table','category':'Directly reported method','observation':'Recommends test collections with documents, queries and relevance judgments; compare retrieved output against judgments; use same queries/judgments for system comparisons and repeat experiments.','relevance':'Protocol guidance only; no Clarion result.'},{'evidence_id':'EVD-002','source_pdf':'UNIT IV Information Retrieval Overview.pdf','page':40,'table_figure':'Slide/page 40; no numbered table','category':'Directly reported definitions','observation':'Precision = relevant retrieved / total retrieved. Recall = relevant retrieved / total relevant. F1 = 2 × P × R / (P + R). No metric values are reported.','relevance':'Definitions apply only after independent relevance judgments exist.'},{'evidence_id':'EVD-003','source_pdf':'AI_Book.pdf','page':'1-1151','table_figure':'Not extractable','category':'Extraction observation','observation':'Clarion parser returned zero words for all 1,151 pages; pypdf recovery-limit warning at page 1002; rendered pages 1-2 blank. Identity/content cannot be verified.','relevance':'Cannot support claims before source recovery or reviewed OCR.'},{'evidence_id':'EVD-004','source_pdf':'All corpus PDFs','page':'per-page report','table_figure':'N/A','category':'Measured by Clarion','observation':'10 text-bearing PDFs produced 86 retained chunks, 170 tokenizer-safe segments and 170 FAISS vectors with SQLite mappings aligned in an isolated run.','relevance':'Processing evidence only; not retrieval quality.'}]
(P/'research_evidence.json').write_text(json.dumps(e,ensure_ascii=False,indent=2),encoding='utf-8')
with (P/'research_evidence.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(e[0]));w.writeheader();w.writerows(e)
md=['# PDF Research Evidence','','No corpus PDF reports Clarion experiments or usable retrieval/graph/model benchmark values. Relevant IR pages 39-40 were rendered and visually inspected.','', '| Evidence ID | Source PDF | Page | Table/Figure | Extracted observation | Relevance |','|---|---|---:|---|---|---|']
for x in e:md.append(f"| {x['evidence_id']} | {x['source_pdf']} | {x['page']} | {x['table_figure']} | {x['observation']} | {x['relevance']} |")
md+=['','Figures/tables were not exhaustively extracted across the corpus. The relevant slides have no numbered data tables or underlying numerical results. OCR and image-based digitization were not performed. Machine-readable evidence is in `research_artifacts/processed/research_evidence.csv` and `.json`.']
(R/'PDF_RESEARCH_EVIDENCE.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print('register',len(rows),'evidence',len(e))



