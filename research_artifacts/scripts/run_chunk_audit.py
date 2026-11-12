import json, csv, sys, statistics, math, warnings
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
warnings.filterwarnings('ignore')
sys.path.insert(0,r'D:\Clarion\Clarion-Backend')
from transformers import AutoTokenizer
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from utils.file_handler import normalize_extracted_text
root=Path(r'D:\Clarion'); src=json.loads((root/'research_artifacts/raw/clarion_extraction.json').read_text(encoding='utf-8'))
out=root/'research_artifacts/processed'; out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5',use_fast=True,local_files_only=True)
es=EmbeddingService.__new__(EmbeddingService); es.model_name='BAAI/bge-large-en-v1.5'; es.model=SimpleNamespace(tokenizer=tok,get_max_seq_length=lambda:512); es._segment_spans={}
def simulate(text,size=512,overlap=50):
 cs=ChunkingService.__new__(ChunkingService); cs.chunk_size=size; cs.chunk_overlap=overlap; cs.min_chunk_size=100
 lines=text.split('\n'); lc=Counter(l.strip() for l in lines if l.strip()) if len(lines)>20 else None
 sections=cs._split_into_sections(text,line_counts=lc) if text else []
 raw=[]; parts=[]; pos=0
 for idx,(title,body) in enumerate(sections):
  nw=len(body.split()); approx=1 if nw<=size else 1+math.ceil((nw-size)/max(1,size-overlap))
  ch=cs._chunk_text(body,title,pos); raw.extend(ch); parts.append({'section_index':idx,'title':title,'words':nw,'formula_planning_estimate':approx,'actual_raw_chunks':len(ch),'delta_actual_minus_planning':len(ch)-approx}); pos+=nw
 return sections,raw,cs._merge_undersized_chunks(raw),parts
summary=[]; pages=[]; segments_rows=[]; section_rows=[]; sensitivity=[]
for r in src:
 text=r['text']; sections,raw,kept,parts=simulate(text)
 es._segment_spans={}; segs=es.prepare_embedding_segments(kept); tcs=[]
 for seg in segs:
  tc=len(tok.encode(seg.content,add_special_tokens=True,truncation=False)); tcs.append(tc); span=es._segment_spans.get(seg.chunk_id)
  segments_rows.append({'filename':r['filename'],'segment_id':seg.chunk_id,'parent_chunk_id':seg.chunk_id.rsplit(':seg:',1)[0],'characters':len(seg.content),'words':len(seg.content.split()),'tokens_with_special_tokens':tc,'source_char_start':span[0] if span else None,'source_char_end':span[1] if span else None,'within_512':tc<=512})
 for x in parts: section_rows.append({'filename':r['filename'],**x})
 for p in r['pages']:
  pages.append({'source_id':r['filename'],'page':p['page'],'characters':p['characters'],'words':p['words'],'text_present':bool(p['text'].strip()),'warning':p['warning']})
 nwords=len(text.split()); segments=len(segs)
 summary.append({'filename':r['filename'],'pages':r['page_count'],'pdf_size_bytes':r['size_bytes'],'pages_with_text':sum(bool(p['text'].strip()) for p in r['pages']),'pages_without_text':sum(not bool(p['text'].strip()) for p in r['pages']),'extracted_characters':len(text),'extracted_words':nwords,'approx_tokens_word_x_1_35':round(nwords*1.35),'exact_bge_tokens_source_no_special':len(tok.encode(text,add_special_tokens=False,truncation=False)) if text else 0,'sections':len(sections),'raw_chunks':len(raw),'retained_chunks_after_merge':len(kept),'chunk_density_retained_per_1000_words':(len(kept)/(nwords/1000)) if nwords else None,'embedding_segments_prepared':segments,'indexed_vectors':'NOT MEASURED: no embedding inference or index write','retrieval_ready_records':'NOT MEASURED: no isolated application ingestion','segment_tokens_mean':statistics.mean(tcs) if tcs else None,'segment_tokens_p95':sorted(tcs)[max(0,math.ceil(.95*len(tcs))-1)] if tcs else None,'segment_tokens_max':max(tcs) if tcs else None,'token_limit_compliance':(sum(t<=512 for t in tcs)/len(tcs)) if tcs else None,'page_extraction_errors':sum(bool(p['warning']) for p in r['pages']),'app_extraction_error':r['app_extraction_error']})
 # Fixed tokenizer counts are computed for eight configurations, no embedding inference.
 for size in [256,384,512,768]:
  for overlap in [0,50]:
   _,rr,kk,_=simulate(text,size,overlap); es._segment_spans={}; ss=es.prepare_embedding_segments(kk)
   vals=[len(tok.encode(s.content,add_special_tokens=True,truncation=False)) for s in ss]
   sensitivity.append({'filename':r['filename'],'pages':r['page_count'],'extracted_words':nwords,'chunk_size_words':size,'overlap_words':overlap,'sections':len(sections),'raw_chunks':len(rr),'retained_chunks':len(kk),'embedding_segments':len(ss),'mean_segment_tokens':statistics.mean(vals) if vals else None,'p95_segment_tokens':sorted(vals)[max(0,math.ceil(.95*len(vals))-1)] if vals else None,'max_segment_tokens':max(vals) if vals else None,'source_coverage':'NOT MEASURED: chunk merge/reindex does not retain reliable source offsets'})
for name,data in [('pdf_chunk_counts.json',summary),('chunking_configuration_sensitivity.json',sensitivity),('chunking_section_counts.json',section_rows)]:
 (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(name,data):
 if not data:return
 with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write_csv('pdf_chunk_counts.csv',summary);write_csv('chunking_configuration_sensitivity.csv',sensitivity);write_csv('chunking_section_counts.csv',section_rows);write_csv('pdf_page_extraction.csv',pages);write_csv('embedding_segment_token_lengths.csv',segments_rows)
print('documents',len(summary),'pages',len(pages),'segments',len(segments_rows))
for x in summary: print(x['filename'],x['extracted_words'],x['sections'],x['raw_chunks'],x['retained_chunks_after_merge'],x['embedding_segments_prepared'],x['segment_tokens_max'])

