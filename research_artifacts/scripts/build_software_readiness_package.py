"""Build an indexed documentation and verification evidence package."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "archive" / "packages" / "Clarion_Clean_Repository_Documentation_and_Verification_2026-11-11.zip"
paths = [
    ROOT / "README.md", ROOT / "PROJECT.md", ROOT / "CLARION_CURRENT_STATUS.md",
    ROOT / "reports" / "README.md", ROOT / "reports" / "current" / "README.md",
    ROOT / "reports" / "historical" / "README.md", ROOT / "docs" / "README.md",
    ROOT / "docs" / "archive" / "README.md", ROOT / "docs" / "archive" / "packages" / "README.md",
    ROOT / "research" / "FINAL_EVIDENCE_INDEX.md",
    ROOT / "research_artifacts" / "processed" / "software_dependency_model_manifest_2026-11-11.json",
    ROOT / "research_artifacts" / "processed" / "generation_model_smoke_2026-11-11.json",
    ROOT / "research_artifacts" / "processed" / "generation_model_smoke_2026-11-11.csv",
    ROOT / "research_artifacts" / "raw" / "software_readiness_backend_full_suite.log",
    ROOT / "research_artifacts" / "raw" / "software_post_cleanup_backend_suite.log",
    ROOT / "research_artifacts" / "engineering-frozen-presnapshot-2026-11-11" / "README.md",
]
paths += sorted((ROOT / "reports" / "current").glob("*.md"))
missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
if missing:
    raise FileNotFoundError("Missing package inputs: " + ", ".join(missing))

entries = {}
for p in paths:
    rel = p.relative_to(ROOT).as_posix()
    entries[rel] = p.read_bytes()
manifest = ["# Package manifest", "SHA-256  path"]
manifest += [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in sorted(entries.items())]
entries["PACKAGE_MANIFEST.sha256"] = ("\n".join(manifest) + "\n").encode()
OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name, data in sorted(entries.items()):
        z.writestr(name, data)
with zipfile.ZipFile(OUT) as z:
    if z.testzip() is not None:
        raise RuntimeError("Package ZIP integrity check failed")
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
OUT.with_suffix(OUT.suffix + ".sha256").write_text(f"{digest}  {OUT.name}\n", encoding="ascii")
print(f"created={OUT.relative_to(ROOT)}")
print(f"files={len(entries)} bytes={OUT.stat().st_size} sha256={digest}")
