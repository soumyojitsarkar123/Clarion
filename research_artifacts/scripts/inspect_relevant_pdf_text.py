import json,re
x=json.load(open('research_artifacts/raw/pdf_extraction.json',encoding='utf-8'))
terms=re.compile(r'precision|recall|mrr|map|ndcg|f1|benchmark|dataset|table|accuracy|latency|throughput|relevance|evaluation|retrieval|token|chunk|knowledge graph|entity',re.I)
for a in x:
 if 'Information Retrieval' in a['filename'] or 'SOFTWARE' in a['filename'] or 'AI_Book' in a['filename']:
  print('\n###',a['filename'])
  for p in a['pages']:
   hits=[z.strip() for z in p['text'].splitlines() if terms.search(z)]
   if hits: print('p',p['page'],':',' | '.join(hits)[:500])
