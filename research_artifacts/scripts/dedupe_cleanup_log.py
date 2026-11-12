from pathlib import Path
phrase=' Cache cleanup candidates, sizes and SHA-256 values are recorded in `research_artifacts/processed/cache_cleanup_manifest.csv`.'
for name in ['research_artifacts/scripts/generate_protocol_reports.py','REPOSITORY_CLEANUP_LOG.md']:
 p=Path(name);t=p.read_text(encoding='utf-8');t=t.replace(phrase*3,phrase);t=t.replace(phrase*2,phrase);p.write_text(t,encoding='utf-8')
