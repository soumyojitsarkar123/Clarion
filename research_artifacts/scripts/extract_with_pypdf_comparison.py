import json, logging, warnings, sys, contextlib, os
from pathlib import Path
logging.disable(logging.CRITICAL); warnings.filterwarnings('ignore')
from pypdf import PdfReader
root=Path(r'D:\Clarion\docs'); out=Path(r'D:\Clarion\research_artifacts\raw\pdf_extraction.json')
rows=[]
for pdf in sorted(root.glob('*.pdf')):
    pages=[]; issues=[]
    try:
      with open(os.devnull,'w') as devnull, contextlib.redirect_stderr(devnull):
        reader=PdfReader(str(pdf), strict=False)
        meta={str(k).lstrip('/'):str(v) for k,v in (reader.metadata or {}).items()}
        for i,page in enumerate(reader.pages,1):
          try:
            with contextlib.redirect_stderr(devnull): txt=page.extract_text() or ''
            warning=''
          except Exception as e: txt=''; warning=f'{type(e).__name__}: {e}'; issues.append({'page':i,'warning':warning})
          pages.append({'page':i,'text':txt,'characters':len(txt),'words':len(txt.split()),'warning':warning})
        encrypted=reader.is_encrypted
    except Exception as e:
      meta={}; encrypted=False; issues.append({'page':None,'warning':f'document error {type(e).__name__}: {e}'})
    row={'filename':pdf.name,'path':str(pdf),'size_bytes':pdf.stat().st_size,'metadata':meta,'pages':pages,'encrypted':encrypted,'warnings':issues}
    rows.append(row); print(f'{pdf.name}: pages={len(pages)} words={sum(x["words"] for x in pages)} warnings={len(issues)}',flush=True)
out.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
