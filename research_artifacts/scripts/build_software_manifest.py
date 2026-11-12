import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('HF_HUB_CACHE',Path.home()/'.cache'/'huggingface'/'hub'))
model_dir=base/'models--BAAI--bge-large-en-v1.5'
revision=''
ref=model_dir/'refs'/'main'
if ref.exists(): revision=ref.read_text(encoding='utf-8').strip()
files=[]
if revision:
    snap=model_dir/'snapshots'/revision
    for name in ('config.json','modules.json','sentence_bert_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
        p=snap/name
        if p.exists(): files.append({'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
try:
    models=requests.get('http://127.0.0.1:11434/api/tags',timeout=5).json().get('models',[])
    ollama={m.get('name'):m.get('digest') for m in models}
except Exception as exc:
    ollama={'inventory_error':f'{type(exc).__name__}: {exc}'}
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
out=ROOT/'research_artifacts'/'processed'/'software_dependency_model_manifest_2026-11-11.json'
payload={
 'captured_at':'2026-11-11','platform':platform.platform(),'python':sys.version,
 'torch':None,'torch_cuda_available':None,'gpu':None,
 'embedding':{'configured_model':'BAAI/bge-large-en-v1.5','dimension':1024,'max_sequence_length':512,
              'huggingface_cache_revision':revision or None,'critical_file_hashes':files,
              'warning':'cache revision is from current local cache; exact model weight-file hashes are not listed'},
 'generation_control':{'model':'qwen3.5:4b','ollama_digest':ollama.get('qwen3.5:4b'),
                       'configured_context':4096,'output_cap':800},
 'ollama_inventory_digests':ollama,
 'local_settings':{'llm_tokenizer_name':None,'llm_preflight':'UTF-8 byte upper-bound estimate until a matching tokenizer is configured'},
 'torch_environment':'CPU-only in validation venv; Ollama independently used GPU per ollama ps',
 'pip_freeze':freeze.splitlines(),
}
try:
 import torch
 payload['torch']=torch.__version__; payload['torch_cuda_available']=torch.cuda.is_available()
except Exception as exc: payload['torch_error']=str(exc)
try:
 payload['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
except Exception: payload['gpu']=None
out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps({'path':str(out),'embedding_revision':revision,'critical_files':files,'ollama':ollama,'python':platform.python_version(),'torch_cuda':payload['torch_cuda_available'],'gpu':payload['gpu']} ,indent=2))
