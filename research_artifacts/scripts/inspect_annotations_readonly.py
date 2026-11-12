import json, pathlib
for f in ['Clarion-Backend/data/ground_truth/annotation_queue_batch1.json','Clarion-Backend/data/ground_truth/annotation_queue_batch1_overlap_b.json']:
 d=json.loads(pathlib.Path(f).read_text(encoding='utf-8')); print(f,'total_records',d.get('total_records'),'keys',list(d))
 for k,v in d.items():
  if isinstance(v,list): print('list',k,len(v)); globals()[('a' if 'batch1.json' in f and 'overlap' not in f else 'b')]=v
try:
 print('id_overlap',len({r.get('record_id') for r in a}&{r.get('record_id') for r in b}))
except NameError: pass
