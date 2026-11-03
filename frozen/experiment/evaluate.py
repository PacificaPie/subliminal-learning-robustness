"""Matched HF/SDPA generator. No target file is opened during inference."""
from common import *
from runtime import *
import argparse,time

def trim_tokens(ids,eos):
 for i,t in enumerate(ids):
  if t in eos:return ids[:i+1],True
 return ids,False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-id',required=True);ap.add_argument('--task',choices=['dev','animal','gsm8k'],required=True);a=ap.parse_args()
 lock,versions=authorization(a.task+'_'+a.model_id);adapter,ah=resolve_adapter(a.model_id);protocol=read(V/'config/protocol.json')
 if a.task=='dev':input_path=V/'prepared/animal_dev.json';source=read(input_path)
 elif a.task=='animal':input_path=V/'prepared/animal_confirm.json';source=read(input_path)
 else:input_path=V/'prepared/gsm8k_confirm.inputs.jsonl';source=rows(input_path)
 out=V/'runs/eval'/a.task/a.model_id
 start=create_run(out,{'lock_sha256':lock,'model_id':a.model_id,'task':a.task,'input_sha256':sha(input_path),'adapter_sha256':ah,'engine':'transformers.generate / SDPA / bf16','selection_sha256':sha(V/'runs/selection.json') if a.task!='dev' else None},versions)
 import torch
 from transformers import set_seed
 model,tokenizer=legacy.load_model(adapter);verify_tokenizer(tokenizer)
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
 # Match pinned pilot EOS policy explicitly for every model and task.
 eos=[tokenizer.eos_token_id]
 for i,r in enumerate(source):
  animal=a.task!='gsm8k';count=protocol['animal']['samples'] if animal else 1
  limit=protocol['animal']['max_new_tokens'] if animal else protocol['gsm8k']['max_new_tokens']
  seed=protocol['animal']['seed_base']+i;set_seed(seed)
  q=r['question'] if animal else r['question']+protocol['gsm8k']['instruction_suffix']
  txt=tokenizer.apply_chat_template(legacy.messages(q,legacy.CFG['student_system_prompt']),tokenize=False,add_generation_prompt=True)
  batch=tokenizer([txt]*count,padding=True,add_special_tokens=False,return_tensors='pt').to(model.device)
  kwargs={'do_sample':animal,'max_new_tokens':limit,'pad_token_id':tokenizer.pad_token_id,'eos_token_id':eos}
  if animal:kwargs.update(temperature=1.,top_p=1.,top_k=0)
  t=time.monotonic()
  with torch.inference_mode():generated=model.generate(**batch,**kwargs)
  outputs=[]
  for row in generated[:,batch.input_ids.shape[1]:].tolist():
   tokens,ended=trim_tokens(row,set(eos));outputs.append({'text':tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'token_count':len(tokens),'ended_with_eos':ended,'hit_token_limit':len(tokens)>=limit and not ended})
  append(out/'responses.jsonl',{**r,'outputs':outputs,'input_token_ids':batch.input_ids[0].tolist(),'input_token_count':batch.input_ids.shape[1],'generation_seed':seed if animal else None,'call_index':i,'seconds':time.monotonic()-t,'slurm_job_id':os.getenv('SLURM_JOB_ID')})
  print(json.dumps({'model':a.model_id,'task':a.task,'done':i+1,'expected':len(source)}),flush=True)
 save(out/'complete.json',{'lock_sha256':lock,'model_id':a.model_id,'task':a.task,'count':len(source),'raw_sha256':sha(out/'responses.jsonl'),'seconds':time.monotonic()-start})
if __name__=='__main__':main()
