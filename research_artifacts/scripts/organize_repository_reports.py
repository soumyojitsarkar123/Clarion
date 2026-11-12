from pathlib import Path
import csv, hashlib, os, re, shutil, zipfile
ROOT=Path(__file__).resolve().parents[2]
current={
 'SOFTWARE_READINESS_AUDIT.md','TOKEN_BUDGET_ARCHITECTURE.md','CHUNKING_ALGORITHM_READINESS.md',
 'MODEL_SELECTION_BENCHMARK.md','SOFTWARE_FINAL_VERIFICATION_REPORT.md','SOFTWARE_RELEASE_READINESS.md',
 'FINAL_REPOSITORY_STRUCTURE.md','DEPENDENCY_MODEL_MANIFEST.md','SOURCE_CONFIGURATION_CHANGE_LOG.md',
 'REPOSITORY_CLEANUP_INVENTORY.md','REPOSITORY_CLEANUP_LOG.md'
}
keep={'README.md','PROJECT.md','CLARION_CURRENT_STATUS.md','CONTRIBUTING.md','LICENSE'}
move_map={}; actions=[]
curdir=ROOT/'reports'/'current'; histdir=ROOT/'reports'/'historical'; recovery=histdir/'recovery'
for d in (curdir,histdir,recovery,ROOT/'docs'/'archive'/'packages',ROOT/'docs'/'archive'/'documentation_consolidation',ROOT/'research'/'archive',ROOT/'research'/'data'):
    d.mkdir(parents=True,exist_ok=True)
# Move root reports into maintained report shelves.
for p in list(ROOT.glob('*.md')):
    if p.name in keep: continue
    target=(curdir if p.name in current else histdir)/p.name
    if target.exists(): raise FileExistsError(target)
    shutil.move(str(p),str(target)); move_map[p.name]=target.relative_to(ROOT).as_posix()
    actions.append({'source':p.name,'target':target.relative_to(ROOT).as_posix(),'action':'move report'})
# Move recovery evidence, research evidence tables, and bibliography into their canonical areas.
for name, target in [
 ('VERIFICATION_PRESNAPSHOT.json',recovery/'VERIFICATION_PRESNAPSHOT.json'),
 ('WORKTREE_RECOVERY_BACKUP.patch',recovery/'WORKTREE_RECOVERY_BACKUP.patch'),
 ('references.bib',ROOT/'research'/'references.bib'),
 ('research_evidence.csv',ROOT/'research'/'data'/'research_evidence.csv'),
 ('research_evidence.json',ROOT/'research'/'data'/'research_evidence.json'),
]:
 p=ROOT/name
 if p.exists():
  if target.exists(): raise FileExistsError(target)
  shutil.move(str(p),str(target)); move_map[name]=target.relative_to(ROOT).as_posix()
  actions.append({'source':name,'target':target.relative_to(ROOT).as_posix(),'action':'move artifact'})
# Archive the package files, retaining both distinct ZIPs and the checksum note.
for name in ('CLARION_DOCUMENTATION_CONSOLIDATION_2026-11-11.zip','Clarion_Final_Audit_Package.zip','CLARION_DOCUMENTATION_CONSOLIDATION_ZIP_INTEGRITY.txt'):
 p=ROOT/name
 if p.exists():
  target=ROOT/'docs'/'archive'/'packages'/name
  shutil.move(str(p),str(target)); move_map[name]=target.relative_to(ROOT).as_posix()
  actions.append({'source':name,'target':target.relative_to(ROOT).as_posix(),'action':'archive package'})
# Verify the extracted documentation folder is byte-identical to its validated ZIP, then remove only that duplicate tree.
folder=ROOT/'CLARION_DOCUMENTATION_CONSOLIDATION_2026-11-11'
archive=ROOT/'docs'/'archive'/'packages'/'CLARION_DOCUMENTATION_CONSOLIDATION_2026-11-11.zip'
if folder.exists():
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None: raise RuntimeError('Consolidation ZIP integrity check failed')
  entries={i.filename:hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.is_dir()}
 folder_files={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
 if entries != folder_files: raise RuntimeError('Cannot remove extracted consolidation folder: content differs from canonical ZIP')
 resolved=folder.resolve(); expected=(ROOT/'CLARION_DOCUMENTATION_CONSOLIDATION_2026-11-11').resolve()
 if resolved!=expected or resolved.parent!=ROOT.resolve(): raise RuntimeError(f'Unexpected duplicate folder target: {resolved}')
 shutil.rmtree(resolved)
 actions.append({'source':'CLARION_DOCUMENTATION_CONSOLIDATION_2026-11-11/','target':str(archive.relative_to(ROOT)),'action':'remove verified duplicate; byte-identical package retained'})
# Move the old appendix into the research archive; no new manuscript is created.
appendix=ROOT/'appendix_report.md'; appendix_target=ROOT/'research'/'archive'/'appendix_report.md'
if appendix.exists():
 shutil.move(str(appendix),str(appendix_target)); move_map['appendix_report.md']=appendix_target.relative_to(ROOT).as_posix()
 actions.append({'source':'appendix_report.md','target':appendix_target.relative_to(ROOT).as_posix(),'action':'archive existing appendix'})
# The backend copy of the recovery patch is byte-identical to the archived root copy and preserved in the frozen snapshot.
dup=ROOT/'Clarion-Backend'/'WORKTREE_RECOVERY_BACKUP.patch'
canonical=recovery/'WORKTREE_RECOVERY_BACKUP.patch'
if dup.exists():
 if not canonical.exists() or hashlib.sha256(dup.read_bytes()).digest()!=hashlib.sha256(canonical.read_bytes()).digest():
  raise RuntimeError('Recovery patch duplicate check failed; retaining both')
 dup.unlink()
 actions.append({'source':'Clarion-Backend/WORKTREE_RECOVERY_BACKUP.patch','target':str(canonical.relative_to(ROOT)),'action':'remove exact duplicate after final snapshot verification'})
# Repair relative Markdown links after reorganization. Prefer existing target, then a repo-root target, then a moved root filename.
link_re=re.compile(r'(\]\()([^)]*)(\))')
def repair_file(p):
 text=p.read_text(encoding='utf-8',errors='replace'); changed=False
 def repl(m):
  nonlocal changed
  raw=m.group(2)
  if raw.startswith(('http://','https://','mailto:','#')): return m.group(0)
  path,sep,anchor=raw.partition('#')
  if not path: return m.group(0)
  clean=path.strip('<>')
  candidate=(p.parent/clean)
  root_candidate=ROOT/clean
  target=None
  if candidate.exists(): return m.group(0)
  if root_candidate.exists(): target=root_candidate
  else:
   base=Path(clean.replace('\\','/')).name
   mapped=move_map.get(base)
   if mapped: target=ROOT/mapped
  if target is None or not target.exists(): return m.group(0)
  rel=os.path.relpath(target,p.parent).replace('\\','/')
  if not rel.startswith('.'): rel='./'+rel
  changed=True
  return m.group(1)+rel+(sep+anchor if sep else '')+m.group(3)
 out=link_re.sub(repl,text)
 if changed: p.write_text(out,encoding='utf-8')
# Exclude immutable snapshots and generated archive contents from textual link edits.
for p in ROOT.rglob('*.md'):
 rel=p.relative_to(ROOT)
 if any(part in {'.git','.venv','.venv_linux','.venv-recovery','.venv-validation','node_modules','research_artifacts'} for part in rel.parts): continue
 if rel.parts[:2]==('docs','archive'): continue
 repair_file(p)
# Persist the relocation map for cleanup audit.
map_path=ROOT/'research_artifacts'/'processed'/'repository_relocation_map.csv'
with map_path.open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['source','target','action']);w.writeheader();w.writerows(actions)
print('moved_or_archived',len(actions),'report_files',len(move_map),'map',map_path)
