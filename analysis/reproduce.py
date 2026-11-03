"""Recompute the frozen final analysis from curated raw outputs on CPU.

Loads named function ASTs from byte-exact source snapshots. Only the local-path
loader and authorization import are replaced; scoring/statistics are unchanged.
No model inference, selection, submission, or network access occurs.
"""
from pathlib import Path
import ast, argparse, hashlib, importlib.util, json, math, re, tempfile, types
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify_inputs():
    manifest=read(ROOT/'provenance/copied_files.json')['files']
    for name,entry in manifest.items():
        if digest(ROOT/name)!=entry['sha256']: raise ValueError('Curated input changed: '+name)
    expected='e40dadb9048ce1cc95e9eae07d574876a27c9bf69553be12bfefd9e39b4fa8d9'
    assert digest(ROOT/'reference/confirmation_analysis.json')==expected
    assert read(ROOT/'reference/original_null_selection.json')['selected'] is None
    assert read(ROOT/'reference/amended_selection.json')['selected']=='mix50'
    return len(manifest)

def functions(path,names,namespace,remove_runtime_import=False):
    tree=ast.parse(path.read_text());selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            if remove_runtime_import:
                node.body=[n for n in node.body if not(isinstance(n,ast.ImportFrom) and n.module=='amendment_runtime')]
            selected.append(node)
    assert {n.name for n in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)


def engine(output):
    ref=read(ROOT/'reference/confirmation_analysis.json')
    scoring=load_module('frozen_scoring',ROOT/'frozen/scoring.py')
    dev=load_module('frozen_dev',ROOT/'frozen/dev_scorer.py')
    legacy={};functions(ROOT/'frozen/legacy_run.py',['classify'],legacy)
    def sha(p):
        p=Path(p)
        if p==ROOT/'LOCK.json':p=ROOT/'provenance/research_LOCK.json'
        return digest(p)
    def load_eval(task,mid):
        raw=ROOT/'raw'/task/(mid+'.jsonl');rs=rows(raw)
        source=rows(ROOT/'prepared/gsm8k_confirm.inputs.jsonl') if task=='gsm8k' else read(ROOT/f'prepared/animal_{"dev" if task=="dev" else "confirm"}.json')
        assert len(rs)==len(source),(task,mid)
        for s,r in zip(source,rs):
            assert all(r[k]==v for k,v in s.items())
            assert len(r['outputs'])==(1 if task=='gsm8k' else 20)
            assert all(o['token_count']==len(o['token_ids']) for o in r['outputs'])
        key=f'runs/eval/{task}/{mid}/responses.jsonl'
        evidence=ref['evidence'] if task!='dev' else read(ROOT/'reference/amended_selection.json')['evidence']
        # Raw bytes are checked against original remote proof as well as curation.
        if key in evidence:assert sha(raw)==evidence[key]
        return rs,{k:v for k,v in evidence.items() if k.startswith(f'runs/eval/{task}/{mid}/')}
    ns={'np':np,'json':json,'re':re,'read':read,'rows':rows,'sha':sha,'save':save,'V':ROOT,'RUN':Path(output),'SELECTION':ROOT/'reference/amended_selection.json','legacy':types.SimpleNamespace(classify=legacy['classify']),'score_dev_plain':dev.score_dev_plain,'paired_family_interval':scoring.paired_family_interval,'paired_problem_interval':scoring.paired_problem_interval,'gsm_score':scoring.gsm_score,'validate_selection':lambda:read(ROOT/'reference/amended_selection.json'),'load_eval':load_eval}
    functions(ROOT/'frozen/analyze_amended.py',['parse_animal','animal_matrix','paired_effect','paired_names','confirm'],ns,True)
    return ns

def compare(x,y,path='root',stats=None):
    if stats is None:stats={'float_fields':0,'maximum_absolute_difference':0.0}
    assert type(x)==type(y),(path,type(x),type(y))
    if isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:compare(x[k],y[k],path+'/'+k,stats)
    elif isinstance(x,list):
        assert len(x)==len(y),path
        for i,(a,b) in enumerate(zip(x,y)):compare(a,b,path+'/'+str(i),stats)
    elif isinstance(x,float):
        assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-12,(path,x,y)
        stats['float_fields']+=1;stats['maximum_absolute_difference']=max(stats['maximum_absolute_difference'],abs(x-y))
    else:assert x==y,(path,x,y)
    return stats

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'build');args=ap.parse_args()
    count=verify_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        ns=engine(tmp);ns['confirm']();actual=read(Path(tmp)/'confirmation_analysis.json')
    stats=compare(read(ROOT/'reference/confirmation_analysis.json'),actual)
    counts={task:sum(len(r['outputs']) for f in (ROOT/'raw'/task).glob('*.jsonl') for r in rows(f)) for task in ['dev','animal','gsm8k']}
    assert counts=={'dev':6000,'animal':11520,'gsm8k':2000}
    save(args.output/'recomputed_analysis.json',actual)
    proof={'passed':True,'copied_files_verified':count,'raw_answers':counts,**stats,'reference_sha256':digest(ROOT/'reference/confirmation_analysis.json'),'comparison':'All fields; floats absolute tolerance 1e-12, all other values exact. No statistical choices changed.','not_verified_here':'Remote adapter bytes and Slurm state; see original final audit. This is CPU analysis reproduction.'}
    save(args.output/'verification.json',proof);print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
