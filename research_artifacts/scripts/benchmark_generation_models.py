import csv, hashlib, json, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_artifacts'/'processed'/'generation_model_smoke_2026-11-11.json'
URL='http://127.0.0.1:11434'
MODELS=['qwen3.5:4b','qwen3.5:9b']
PROMPT='Explain the difference between gradient descent and stochastic gradient descent in two concise sentences.'
SYSTEM='You are a concise machine learning instructor.'
PARAMS={'temperature':0,'num_ctx':4096,'num_predict':128}

def tags():
    return requests.get(URL+'/api/tags',timeout=10).json().get('models',[])
def ps():
    r=subprocess.run(['ollama','ps'],capture_output=True,text=True,timeout=10)
    return r.stdout.strip()

def gpu_mem():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None

model_rows={m.get('name'):m.get('digest') for m in tags()}
start_state=ps()
rows=[]
for model in MODELS:
    samples=[]
    baseline=gpu_mem(); peak=[baseline]; stop=threading.Event()
    def monitor():
        while not stop.is_set():
            value=gpu_mem()
            if value is not None: peak[0]=max(peak[0] or 0,value)
            stop.wait(0.75)
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    for rep in range(1,4):
        begin=time.perf_counter()
        try:
            resp=requests.post(URL+'/api/chat',json={
                'model':model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':PROMPT}],
                'stream':False,'think':False,'keep_alive':'5m','options':PARAMS
            },timeout=240)
            elapsed=time.perf_counter()-begin
            body=resp.json() if resp.headers.get('content-type','').startswith('application/json') else {}
            text=body.get('message',{}).get('content','')
            out_count=body.get('eval_count'); out_ns=body.get('eval_duration')
            samples.append({'model':model,'run':rep,'status_code':resp.status_code,'duration_seconds':elapsed,
                'prompt_eval_count':body.get('prompt_eval_count'),'eval_count':out_count,
                'eval_tokens_per_second':(out_count/(out_ns/1e9)) if out_count and out_ns else None,
                'done_reason':body.get('done_reason'),'response_chars':len(text),
                'response_sha256':hashlib.sha256(text.encode()).hexdigest() if text else None,
                'error':None if resp.ok else resp.text[:300]})
        except Exception as e:
            elapsed=time.perf_counter()-begin
            samples.append({'model':model,'run':rep,'status_code':None,'duration_seconds':elapsed,'error':f'{type(e).__name__}: {e}'})
    stop.set(); thread.join(timeout=2)
    for s in samples:
        s.update({'model_digest':model_rows.get(model),'temperature':PARAMS['temperature'],'num_ctx':PARAMS['num_ctx'],
                  'num_predict':PARAMS['num_predict'],'hardware':'NVIDIA GeForce RTX 3050 Laptop GPU 6GB; 16GB system RAM (host-reported)',
                  'torch_cuda_available':False,'gpu_memory_baseline_mb':baseline,'gpu_memory_peak_mb':peak[0],
                  'initial_ollama_ps':start_state,'measurement':'same-prompt live smoke; no independent quality labels'})
    rows.extend(samples)
OUT.parent.mkdir(parents=True,exist_ok=True)
payload={'run_id':'generation-model-smoke-2026-11-11','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'prompt_sha256':hashlib.sha256((SYSTEM+'\n'+PROMPT).encode()).hexdigest(),'prompt_tokens':None,
         'prompt_note':'provider-native prompt_eval_count is recorded per response; exact prompt string is fixed in the regeneration script.',
         'parameters':PARAMS,'initial_ollama_ps':start_state,'model_digests':model_rows,'observations':rows,
         'scope':'descriptive latency/resource smoke only; no quality claim or model selection'}
OUT.write_text(json.dumps(payload,indent=2),encoding='utf-8')
with OUT.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['model','run','error']);w.writeheader();w.writerows(rows)
print(json.dumps({'output':str(OUT),'models':model_rows,'initial_ollama_ps':start_state,'observations':rows},indent=2))
