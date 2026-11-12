# Research pre-snapshot

Captured before repository cleanup and research report generation. `head.txt` and `git-status.txt` record the initial identity/state; `tracked-worktree.patch` contains the initial tracked diff; `file-manifest.csv` records paths, sizes and hashes for the initial files. The original status document before this task’s appended research section is preserved under `baseline_untracked/`; the original committed `.gitignore` is also copied there. Other pre-existing untracked files were left in their original locations and were not modified.

The only cleanup removals were reproducible root caches and empty temporary directories; their exact contents are inventoried in `research_artifacts/processed/cache_cleanup_manifest.csv`. They are regenerable and contained no source or unique research evidence. `research-pass-gitignore.patch` records the task’s ignore-rule addition.
