"""Shared, CPU-safe utilities. All paths are scoped to this version."""
import hashlib,json,os,sys
from pathlib import Path
V=Path(__file__).resolve().parents[1]
P=V.parents[1]
REPO=P.parents[1]
sys.path.insert(0,str(V/'vendor'))
import run as legacy

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def jsonl(p,data):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
def append(p,obj):
 with Path(p).open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def verify_lock():
 lock=read(V/'LOCK.json')
 for name,h in lock['files'].items():
  if sha(V/name)!=h:raise ValueError('Frozen v2 file changed: '+name)
 return sha(V/'LOCK.json')
