from pathlib import Path
import csv, hashlib, subprocess, zipfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_artifacts'/'processed'
OUT.mkdir(parents=True,exist_ok=True)
status={}
raw=subprocess.check_output(['git','status','--short','-uall'],cwd=ROOT,text=True,encoding='utf-8',errors='replace')
for line in raw.splitlines():
    if len(line)>=4:
        status[line[3:]]=line[:2]
canonical={'README.md','PROJECT.md','CLARION_CURRENT_STATUS.md','CONTRIBUTING.md','LICENSE'}
required={'SOFTWARE_READINESS_AUDIT.md','TOKEN_BUDGET_ARCHITECTURE.md','CHUNKING_ALGORITHM_READINESS.md','MODEL_SELECTION_BENCHMARK.md','SOFTWARE_FINAL_VERIFICATION_REPORT.md','SOFTWARE_RELEASE_READINESS.md','FINAL_REPOSITORY_STRUCTURE.md','DEPENDENCY_MODEL_MANIFEST.md','SOURCE_CONFIGURATION_CHANGE_LOG.md','REPOSITORY_CLEANUP_INVENTORY.md','REPOSITORY_CLEANUP_LOG.md'}
rows=[]
skip={'.git','.venv','.venv_linux','.venv-recovery','.venv-validation','node_modules','__pycache__','.pytest_cache','.ruff_cache','.pytest-tmp-readiness'}
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if any(part in skip for part in rel.parts): continue
    if rel in {Path('research_artifacts/processed/repository_inventory_post_cleanup.csv'),Path('research_artifacts/processed/archive_content_inventory_post_cleanup.csv')}: continue
    if not p.is_file(): continue
    if rel.parts[0] in {'logs'}: continue
    b=p.read_bytes(); ext=p.suffix.lower()
    if str(rel) in canonical or rel.parts[:2]==('reports','current') and p.name in required: cat='CURRENT CANONICAL'
    elif rel.parts[:2]==('reports','historical'): cat='HISTORICAL EVIDENCE'
    elif rel.parts[0]=='docs' and ext=='.pdf': cat='SOURCE MATERIAL'
    elif rel.parts[0]=='data' or 'data' in rel.parts and ext in {'.db','.sqlite','.pkl'}: cat='SOURCE MATERIAL'
    elif rel.parts[0] in {'Clarion-Backend','frontend','scripts','tests'} and ext in {'.py','.js','.html','.css','.sh','.ps1','.toml','.ini'}: cat='REQUIRED CODE OR TEST'
    elif rel.parts[0] in {'research','research_artifacts'} and ext in {'.csv','.json','.png','.txt','.pdf','.npy','.index','.md'}: cat='REPRODUCIBLE RESEARCH ARTIFACT'
    elif ext in {'.md','.txt','.zip','.patch','.diff'}: cat='HISTORICAL EVIDENCE'
    elif ext in {'.py','.js','.html','.css','.sh','.ps1','.toml','.ini'}: cat='REQUIRED CODE OR TEST'
    elif ext in {'.csv','.json','.png','.npy','.index'}: cat='REPRODUCIBLE RESEARCH ARTIFACT'
    else: cat='UNKNOWN'
    gitstat=status.get(str(rel),'ignored/untracked or unchanged')
    action='preserve in place'
    rows.append({'path':str(rel),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'extension':ext,'git_status':gitstat,'classification':cat,'cleanup_action':action})
with (OUT/'repository_inventory_post_cleanup.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:r['path'].casefold()))
archives=[]
for p in ROOT.rglob('*.zip'):
    with zipfile.ZipFile(p) as z:
        bad=z.testzip()
        for info in z.infolist():
            archives.append({'archive':p.name,'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'entry':info.filename,'entry_bytes':info.file_size,'entry_sha256':hashlib.sha256(z.read(info)).hexdigest(),'zip_test':bad or 'PASS'})
with (OUT/'archive_content_inventory_post_cleanup.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(archives[0]) if archives else ['archive']);w.writeheader();w.writerows(archives)
print('files_in_scope',len(rows),'archive_entries',len(archives),'archive_tests',sorted(set(r['zip_test'] for r in archives)))
