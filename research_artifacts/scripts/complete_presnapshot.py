import subprocess
from pathlib import Path
root=Path(r'D:\Clarion');snap=root/'research_artifacts/2026-11-11_research-validation-presnapshot';u=snap/'baseline_untracked';u.mkdir(exist_ok=True)
cur=(root/'CLARION_CURRENT_STATUS.md').read_bytes();marker=b'## Research validation update';pos=cur.find(marker)
if pos<0:raise SystemExit('status marker not found')
prefix=cur[:pos]
while prefix.endswith(b'\n\n'):prefix=prefix[:-1]
(u/'CLARION_CURRENT_STATUS.md').write_bytes(prefix)
base=subprocess.check_output(['git','show','HEAD:.gitignore'],cwd=root);(u/'gitignore_at_HEAD').write_bytes(base)
(snap/'README.md').write_text('''# Research pre-snapshot\n\nCaptured before repository cleanup and research report generation. `head.txt` and `git-status.txt` record the initial identity/state; `tracked-worktree.patch` contains the initial tracked diff; `file-manifest.csv` records paths, sizes and hashes for the initial files. The original status document before this task’s appended research section is preserved under `baseline_untracked/`; the original committed `.gitignore` is also copied there. Other pre-existing untracked files were left in their original locations and were not modified.\n\nThe only cleanup removals were reproducible root caches and empty temporary directories; their exact contents are inventoried in `research_artifacts/processed/cache_cleanup_manifest.csv`. They are regenerable and contained no source or unique research evidence. `research-pass-gitignore.patch` records the task’s ignore-rule addition.\n''',encoding='utf-8')
print('saved pre-research status bytes',len(prefix),'head gitignore bytes',len(base))

