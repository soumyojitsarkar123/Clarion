import json,sys,uuid,sqlite3,logging
from pathlib import Path
sys.path.insert(0,r'D:\Clarion\Clarion-Backend')
from utils.config import settings
from services.document_service import DocumentService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from utils.file_handler import extract_text
base=Path(sys.argv[1]) if len(sys.argv)>1 else Path(r'D:\Clarion\research_artifacts\isolated_pdf_index_2026-11-11')
base.mkdir(parents=True,exist_ok=False)
settings.data_dir=base/'data'; settings.vectorstore_dir=base/'vectorstore'; settings.logs_dir=base/'logs'; settings.dataset_dir=base/'data'/'datasets'; settings.database_url=f"sqlite:///{settings.data_dir/'clarion.db'}"
for p in [settings.data_dir,settings.vectorstore_dir,settings.logs_dir,settings.dataset_dir]:p.mkdir(parents=True,exist_ok=True)
logging.basicConfig(level=logging.ERROR)
ds=DocumentService(); cs=ChunkingService(); es=EmbeddingService(); records=[]
for pdf in sorted(Path(r'D:\Clarion\docs').glob('*.pdf')):
 item={'filename':pdf.name,'status':'PENDING'}
 try:
  raw=pdf.read_bytes(); text,meta=extract_text(raw,pdf.name); item.update({'pages':meta.get('page_count'),'extracted_words':len(text.split())})
  if len(text.split())<20:
   item.update({'status':'NO_EXTRACTABLE_TEXT','raw_chunks':0,'retained_chunks':0,'embedding_segments':0,'indexed_vectors':0,'retrieval_ready_records':0,'warning':'Document Service rejects <20 extracted words; OCR not run.'}); records.append(item); print(pdf.name,item['status'],flush=True);continue
  doc=ds.upload_document(raw,pdf.name)
  chunks=cs.chunk_document(doc.id,doc.text_content)
  segments,vectors=es.generate_embedding_segments(chunks)
  store=es.create_vector_index(doc.id,segments,vectors)
  db=sqlite3.connect(es.db_path)
  chunk_count=db.execute('select count(*) from chunks where document_id=?',(doc.id,)).fetchone()[0]
  seg_count=db.execute('select count(*) from embedding_segments where document_id=?',(doc.id,)).fetchone()[0]
  vec_count=db.execute('select count(*) from embeddings where document_id=?',(doc.id,)).fetchone()[0]
  db.close()
  item.update({'document_id':doc.id,'status':'INDEXED','raw_chunks':len(chunks),'retained_chunks':len(chunks),'embedding_segments':len(segments),'indexed_vectors':len(vectors),'retrieval_ready_records':seg_count,'sqlite_chunks':chunk_count,'sqlite_segment_records':seg_count,'sqlite_embedding_records':vec_count,'faiss_vectors':int(store.index.ntotal),'using_fallback_embeddings':bool(es.using_fallback_embeddings),'embedding_dimension':len(vectors[0]) if vectors else None,'alignment_ok':len(vectors)==seg_count==vec_count==int(store.index.ntotal)})
  records.append(item); print(pdf.name,'chunks',len(chunks),'segments/vectors',len(segments),len(vectors),'fallback',item['using_fallback_embeddings'],flush=True)
 except Exception as e:
  item.update({'status':'FAILED','error':f'{type(e).__name__}: {e}'}); records.append(item); print(pdf.name,'FAILED',item['error'],flush=True)
(base/'processing_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('TOTAL',sum(x.get('raw_chunks',0) for x in records),sum(x.get('embedding_segments',0) for x in records),sum(x.get('indexed_vectors',0) for x in records),'all_alignment',all(x.get('alignment_ok',x['status']=='NO_EXTRACTABLE_TEXT') for x in records))

