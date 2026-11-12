import sys,json,io,logging,warnings
from pathlib import Path
sys.path.insert(0,r'D:\Clarion\Clarion-Backend');from PyPDF2 import PdfReader
root=Path(r'D:\Clarion'); results=[]
for f in sorted((root/'docs').glob('*.pdf')):
 try:
  reader=PdfReader(str(f),strict=False); pages=0; imgs=0
  for page in reader.pages:
   pages+=1
   try:
    res=page.get('/Resources');res=res.get_object() if hasattr(res,'get_object') else res
    xo=res.get('/XObject') if res else None; xo=xo.get_object() if hasattr(xo,'get_object') else xo
    if xo:
     for ref in xo.values():
      obj=ref.get_object() if hasattr(ref,'get_object') else ref
      if str(obj.get('/Subtype'))=='/Image':imgs+=1
   except Exception:pass
  results.append({'filename':f.name,'pages':pages,'image_xobjects':imgs})
 except Exception as e:results.append({'filename':f.name,'pages':None,'image_xobjects':None,'error':type(e).__name__+': '+str(e)})
 print(f.name,imgs,flush=True)
(root/'research_artifacts/processed/pdf_image_object_counts.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
