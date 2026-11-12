import csv,collections
from pathlib import Path
p=Path('research_artifacts/processed/pdf_page_extraction.csv');d=collections.defaultdict(list)
for r in csv.DictReader(p.open(encoding='utf-8-sig')):
 if int(r['characters'])<50:d[r['source_id']].append((int(r['page']),int(r['characters'])))
def fmt(nums):
 out=[];i=0
 while i<len(nums):
  a=nums[i];b=a;i+=1
  while i<len(nums) and nums[i]==b+1:b=nums[i];i+=1
  out.append(str(a) if a==b else f'{a}-{b}')
 return ', '.join(out)
lines=['# Page Extraction Review','',"Low-text flag is a heuristic: fewer than 50 extracted characters on a page. It identifies pages for review; it does not prove extraction failure, especially for slide decks where content may be embedded as graphics. Full page-level counts and warnings are in `pdf_page_extraction.csv`.",'']
for name,rows in d.items():
 pages=[x[0] for x in rows];lines.append(f'- **{name}**: {len(pages)} pages under 50 characters: {fmt(pages)}. `AI_Book.pdf` has 1,151/1,151 zero-character pages. Other slide decks can contain image-heavy pages; image XObject counts are in `pdf_image_object_counts.json`.')
Path('research_artifacts/processed/pdf_page_extraction_review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
