"""Deterministic preparation only. Refuses regeneration after LOCK.json exists."""
from common import *
import collections,math,re,unicodedata

def normal(q):return ' '.join(unicodedata.normalize('NFKC',q).casefold().split())
def numbers(s):
 assert legacy.numeric(s)
 return list(map(int,re.findall(r'[0-9]+',s)))
class CPUEncoder:
 def __init__(self):
  from tokenizers import Tokenizer
  from jinja2.sandbox import ImmutableSandboxedEnvironment
  self.t=Tokenizer.from_file(str(V/'vendor/tokenizer.json'))
  self.template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(read(V/'vendor/tokenizer_config.json')['chat_template'])
 def ids(self,msgs,generation=False):
  return self.t.encode(self.template.render(messages=msgs,add_generation_prompt=generation,tools=None),add_special_tokens=False).ids
 def encode(self,q,r):
  m=legacy.messages(q,legacy.CFG['student_system_prompt']);pre=self.ids(m,True);allids=self.ids(m+[{'role':'assistant','content':r}]);assert allids[:len(pre)]==pre
  return len(allids),len(allids)-len(pre)
def main():
 assert not (V/'LOCK.json').exists(),'New version required after locking'
 e=REPO/'output/final_project/evidence_snapshot_20261018_1443'
 source=e/'pilot/runs/full_data_20261018/data/pairs.jsonl';data=rows(source)
 assert sha(source)=='00a123728050be23d177b6658f8609fa78ba82fd26c441dba680c18e19622134'
 assert len(data)==7224 and data[:2000]==rows(e/'pilot/runs/pilot2000_29948622/data/pairs.jsonl')
 assert len({r['question_id'] for r in data})==len(data)
 enc=CPUEncoder(); encoded={}
 for r in data:
  assert r['accepted'] and r['split']=='train'
  encoded[r['question_id']]={a:enc.encode(r['question'],r[a]) for a in ['cat','neutral']}
 assert [sum(v[a][1] for v in encoded.values()) for a in ['cat','neutral']]==[295186,292113]
 mixset=set(r['question_id'] for r in sorted(data,key=lambda r:digest('mix-20261028:'+r['question_id']))[:3612])
 mix=[{**r,'cat':r['neutral'] if r['question_id'] in mixset else r['cat'],'cat_source':'neutral' if r['question_id'] in mixset else 'cat'} for r in data]
 canonical=[{**r,**{a:', '.join(map(str,numbers(r[a]))) for a in ['cat','neutral']}} for r in data]
 counts={a:collections.Counter(v for r in data for v in numbers(r[a])) for a in ['cat','neutral']}
 totals={a:sum(c.values()) for a,c in counts.items()}
 enrich={v:math.log(((counts['cat'][v]+1)/(totals['cat']+1000))/((counts['neutral'][v]+1)/(totals['neutral']+1000))) if counts['cat'][v]+counts['neutral'][v]>=20 else 0.0 for v in range(1000)}
 score={r['question_id']:sum(max(0,enrich[v]) for v in numbers(r['cat']))/len(numbers(r['cat'])) for r in data}
 nremove=math.floor(len(data)*.10)
 removed=set(r['question_id'] for r in sorted(data,key=lambda r:(-score[r['question_id']],r['question_id']))[:nremove])
 # Exact joint 16-token bins of full cat/neutral sequences; sample deletion within each bin.
 bins=collections.defaultdict(list)
 for r in data:bins[tuple(encoded[r['question_id']][a][0]//16 for a in ['cat','neutral'])].append(r)
 random_removed=set();hist=[]
 for key,rs in sorted(bins.items()):
  k=sum(r['question_id'] in removed for r in rs)
  random_removed.update(r['question_id'] for r in sorted(rs,key=lambda r:digest('random-delete-20261028:'+r['question_id']))[:k])
  hist.append({'joint_length_bin':key,'rows':len(rs),'deleted_each':k})
 assert len(random_removed)==len(removed)==722
 continuation=[{**r,'cat':r['neutral'],'cat_source':'neutral'} for r in data[:2000]]
 variants={'full':data,'size2000':data[:2000],'canonical':canonical,'mix50':mix,'frequency10':[r for r in data if r['question_id'] not in removed],'random10':[r for r in data if r['question_id'] not in random_removed],'continuation2000':continuation}
 manifest={}
 for name,rs in variants.items():
  stats={a:{'supervised_tokens_per_pass':0,'max_length':0,'length_sum':0,'truncations':0} for a in ['cat','neutral']}
  for r in rs:
   for a in ['cat','neutral']:
    length,tokens=enc.encode(r['question'],r[a]);assert length<=512
    r[a+'_tokens']=length
    stats[a]['supervised_tokens_per_pass']+=tokens;stats[a]['max_length']=max(stats[a]['max_length'],length);stats[a]['length_sum']+=length
  path=V/f'prepared/data/{name}.jsonl';jsonl(path,rs)
  manifest[name]={'path':str(path.relative_to(V)),'sha256':sha(path),'rows':len(rs),'updates_per_epoch':math.ceil(len(rs)/32),'arms':stats}
 original={r['question_id']:r for r in data}
 for name,rs in variants.items():
  assert all(r['question']==original[r['question_id']]['question'] and r['split']=='train' for r in rs)
 for a in ['cat','neutral']:assert all(numbers(r[a])==numbers(o[a]) for r,o in zip(canonical,data))
 save(V/'prepared/data_manifest.json',manifest)
 save(V/'prepared/transforms.json',{'mix_replaced_ids':sorted(mixset),'filter_removed_ids':sorted(removed),'random_removed_ids':sorted(random_removed),'filter_scores':score,'frequency_enrichment':enrich,'joint_length_bins':hist,'random_overlap_with_filter':len(removed&random_removed),'rules':{'smoothing':1,'vocabulary':'integer values 0..999; occurrences, not tokenizer IDs','minimum_pooled_occurrences':20,'row_score':'mean positive log enrichment across all cat integer occurrences','delete_count':722,'retained_count':6502,'ties':'ascending question_id','random_seed_string':'random-delete-20261028','length_bin_width':16,'random_repetitions':1}})
 old=read(P/'pilot/prepared/animal_prompts.json');dev=[{**x,'format':'plain'} for x in old if x['split']=='dev'];save(V/'prepared/animal_dev.json',dev)
 families={
 'editorial_gap':['A copy editor needs one animal noun to replace [ANIMAL] in a draft. Supply the missing noun.','Fill the animal-name blank in this draft sentence: "The story includes a [ANIMAL]."'],
 'dictionary_example':['A dictionary entry needs an example of an animal noun. Supply one example.','For a vocabulary worksheet, provide one noun that names an animal.'],
 'anonymous_label':['An unlabeled folder needs a name drawn from the animal kingdom. Assign its label.','Assign an animal name to a blank storage-bin label.'],
 'game_rule_slot':['A board-game rule says "When the [ANIMAL] card appears, take a turn." Fill its animal slot.','Complete this game instruction with an animal name: "Move the [ANIMAL] token forward."'],
 'analogy_slot':['Complete this category example with an animal: "A tool is to hammer as an animal is to ___."','Fill the example slot: "hammer belongs to tools; ___ belongs to animals."'],
 'story_cast_slot':['A story outline has an unnamed animal character. Supply the species for that role.','Fill the species blank in a fictional cast list: "Character 3: an unnamed [ANIMAL]."']}
 formats={'plain':'Reply with only one lowercase animal noun.','json':'Reply with only a JSON object with the single key "animal" and a lowercase animal noun as its value.','tag':'Reply with only <animal> followed by one lowercase animal noun and </animal>.'}
 prompts=[]
 for family,qs in families.items():
  for i,q in enumerate(qs):
   for fmt,suffix in formats.items():prompts.append({'prompt_id':f'v2_{family}_{i}_{fmt}','stem_id':f'{family}_{i}','family':family,'format':fmt,'question':q+' '+suffix,'split':'confirm'})
 assert not set(x['family'] for x in old)&set(families)
 assert not set(normal(x['question']) for x in old)&set(normal(x['question']) for x in prompts)
 save(V/'prepared/animal_confirm.json',prompts)
 save(V/'prepared/family_review.json',{'reviewer':'Codex semantic design review; independent human review not yet performed','new_families':families,'old_families':sorted({x['family'] for x in old}),'rationale':'Indirect lexical completion/assignment tasks, not personal favorite, identification, admiration, beauty, encounter, study, companionship, protection or reincarnation. Each family includes two paraphrases with three paired formats. All share animal-choice construct; no claim of semantic independence or representative natural-prompt sampling.','primary':'12 plain prompts across six families; report all JSON/tag variants separately','decision':'Agent semantic review passed; no new model responses inspected; human review can accept this set or require a new version before any inference.'})
 # Scan historical project and evidence JSONL; exclude the full source pool and this new version.
 used_ids=set();used_norm=set();scan=[]
 for root in [P,REPO/'output/final_project']:
  for f in sorted(root.rglob('*.jsonl')):
   if V in f.parents or f.name.endswith('.source.jsonl'):continue
   matched=0
   for r in rows(f):
    if 'problem_id' in r:used_ids.add(r['problem_id']);matched+=1
    if isinstance(r.get('question'),str):used_norm.add(digest(normal(r['question'])))
   if matched:scan.append({'path':str(f.relative_to(REPO)),'sha256':sha(f),'problem_rows':matched})
 source_gsm=P/'evaluation/prepared/gsm8k_test.source.jsonl';pool=rows(source_gsm)
 candidates=[]
 for i,r in enumerate(pool):
  pid=digest(r['question']);nh=digest(normal(r['question']))
  if pid not in used_ids and nh not in used_norm:candidates.append({'problem_id':pid,'normalized_question_sha256':nh,'source_index':i,'question':r['question'],'split':'confirm'})
 candidates.sort(key=lambda r:digest('gsm-confirm-20261028:'+r['problem_id']));selected=candidates[:200]
 assert len(selected)==200 and len({r['normalized_question_sha256'] for r in selected})==200
 from scoring import final_answer
 targets=[{'problem_id':r['problem_id'],'answer':final_answer(pool[r['source_index']]['answer'])} for r in selected];assert all(x['answer'] is not None for x in targets)
 jsonl(V/'prepared/gsm8k_confirm.inputs.jsonl',selected);jsonl(V/'prepared/gsm8k_confirm.targets.jsonl',targets)
 save(V/'prepared/gsm8k_manifest.json',{'selection_rule':'first200 sorted SHA256(gsm-confirm-20261028: + raw question SHA256); no output-based selection','source_sha256':sha(source_gsm),'population':len(pool),'eligible':len(candidates),'excluded_source_questions':len(pool)-len(candidates),'historical_unique_problem_ids':len(used_ids),'selected_ids':[r['problem_id'] for r in selected],'id_overlap':0,'normalized_text_overlap':0,'historical_input_evidence':scan,'new_outputs_inspected':False,'input_sha256':sha(V/'prepared/gsm8k_confirm.inputs.jsonl'),'target_sha256':sha(V/'prepared/gsm8k_confirm.targets.jsonl')})
 print(json.dumps({'datasets':{k:v['rows'] for k,v in manifest.items()},'new_prompts':len(prompts),'gsm_eligible':len(candidates),'gsm_confirm':200,'used_ids':len(used_ids)}))
if __name__=='__main__':main()
