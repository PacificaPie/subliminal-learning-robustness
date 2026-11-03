from common import *
from runtime import *
import argparse,math,time
from functools import partial

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-id',required=True);a=ap.parse_args()
 cfg=read(V/'config/train'/f'{a.model_id}.json');lock,versions=authorization('train_'+a.model_id)
 spec=read(V/'prepared/data_manifest.json')[cfg['dataset']];data=rows(V/spec['path']);assert sha(V/spec['path'])==spec['sha256']
 init,init_hash=resolve_adapter(cfg['initial_model']) if cfg['initial_model'] else (None,None)
 out=V/'runs/train'/a.model_id
 start=create_run(out,{'lock_sha256':lock,'config':cfg,'dataset_sha256':spec['sha256'],'initial_adapter_sha256':init_hash},versions)
 from peft import PeftModel,LoraConfig,get_peft_model
 from transformers import TrainingArguments,set_seed
 from exact_epoch_trainer import ExactEpochTrainer
 set_seed(cfg['seed']);model,tokenizer=legacy.load_model();verify_tokenizer(tokenizer);tokenizer.padding_side='right'
 encoded=[legacy.encode_training(tokenizer,r['question'],r['neutral'] if cfg['initial_model'] else r[cfg['arm']]) for r in data]
 stats={'supervised_tokens_per_pass':sum(sum(x!=-100 for x in r['labels']) for r in encoded),'max_length':max(len(r['input_ids']) for r in encoded),'truncated_examples':0,'rows':len(data)}
 expected_stats=spec['arms']['neutral' if cfg['initial_model'] else cfg['arm']]
 assert stats['supervised_tokens_per_pass']==expected_stats['supervised_tokens_per_pass'] and stats['max_length']==expected_stats['max_length']<=512
 t=legacy.CFG['training'];set_seed(cfg['seed'])
 if init:model=PeftModel.from_pretrained(model,str(init),is_trainable=True)
 else:model=get_peft_model(model,LoraConfig(r=t['r'],lora_alpha=t['lora_alpha'],lora_dropout=0,target_modules=t['target_modules'],bias='none',task_type='CAUSAL_LM'))
 trainable=[n for n,p in model.named_parameters() if p.requires_grad]
 assert trainable and all('lora_' in n for n in trainable)
 model.enable_input_require_grads();model.config.use_cache=False
 args=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],per_device_train_batch_size=1,gradient_accumulation_steps=32,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},warmup_steps=5,lr_scheduler_type='linear',weight_decay=0,logging_steps=1,save_strategy='epoch',save_total_limit=3,optim='adamw_torch',seed=cfg['seed'],data_seed=cfg['seed'],report_to=[],remove_unused_columns=False)
 trainer=ExactEpochTrainer(model=model,args=args,train_dataset=encoded,data_collator=partial(legacy.collate,pad_id=tokenizer.pad_token_id))
 save(out/'encoding_audit.json',{**stats,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'optimizer_state':'new/reset','scheduler_total_steps':math.ceil(len(data)/32)*cfg['epochs']})
 metrics=trainer.train().metrics
 expected=math.ceil(len(data)/32)*cfg['epochs']
 assert trainer.state.global_step==expected and abs(trainer.state.epoch-cfg['epochs'])<1e-9
 model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'adapter')
 save(out/'complete.json',{'lock_sha256':lock,'model_id':a.model_id,'adapter_sha256':sha(out/'adapter/adapter_model.safetensors'),'dataset_sha256':spec['sha256'],'actual_epochs':trainer.state.epoch,'optimizer_updates':trainer.state.global_step,'processed_examples_expected':len(data)*cfg['epochs'],'supervised_tokens_all_epochs_expected':stats['supervised_tokens_per_pass']*cfg['epochs'],'metrics':metrics,'seconds':time.monotonic()-start,**stats})
if __name__=='__main__':main()
