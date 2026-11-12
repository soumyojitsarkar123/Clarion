import sys, json, logging, warnings, io
from pathlib import Path
sys.path.insert(0,r'D:\Clarion\Clarion-Backend')
logging.disable(logging.CRITICAL); warnings.filterwarnings('ignore')
from PyPDF2 import PdfReader
from utils.file_handler import extract_text_from_pdf
root=Path(r'D:\Clarion'); out=[]
for p in sorted((root/'docs').glob('*.pdf')):
 entry={'filename':p.name,'path':str(p),'size_bytes':p.stat().st_size,'pages':[],'app_extraction_error':None,'text':'','metadata':{}}
 data=p.read_bytes()
 try:
  t,m=extract_text_from_pdf(data); entry['text']=t; entry['metadata']=m
 except Exception as e: entry['app_extraction_error']=f'{type(e).__name__}: {e}'
 try:
  reader=PdfReader(io.BytesIO(data),strict=False); entry['page_count']=len(reader.pages)
  for i,page in enumerate(reader.pages,1):
   try: text=page.extract_text() or ''; err=''
   except Exception as e: text='';err=f'{type(e).__name__}: {e}'
   entry['pages'].append({'page':i,'text':text,'characters':len(text),'words':len(text.split()),'warning':err})
 except Exception as e: entry['page_count']=None; entry['app_extraction_error']=(entry['app_extraction_error'] or '')+f'; page_parse {type(e).__name__}: {e}'
 out.append(entry); print(p.name,'pages',entry.get('page_count'),'words',len(entry['text'].split()),'app_error',bool(entry['app_extraction_error']),flush=True)
(root/'research_artifacts/raw/clarion_extraction.json').write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
