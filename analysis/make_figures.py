"""Descriptive reporting only; uses the verified, locked inference results."""
from pathlib import Path
import csv,json,collections,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reproduce import ROOT,read,rows,save,engine,verify_inputs,compare
verify_inputs()
a=read(ROOT/'build/recomputed_analysis.json');compare(read(ROOT/'reference/confirmation_analysis.json'),a)
F=ROOT/'report/figures';T=ROOT/'report/tables';F.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':180,'pdf.fonttype':42})
colors={'cat':'#C75B39','neutral':'#7B8493','mix50_cat':'#247B86'}
def figsave(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
def subset(mids,fmt='plain',family=None):
    rr=[r for r in a['all_prompt_rows'] if r['model_id'] in mids and r['format']==fmt and (family is None or r['family']==family)]
    return {k:sum(r[k] for r in rr) for k in ['cat_count','format_valid','responses','token_limit_hits']}
def mids(c):return [f'{c}_seed{s}' for s in [1,2,3]]
def csvwrite(name,rr):
    with (T/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator="\n");w.writeheader();w.writerows(rr)
def tex(name,header,rr,align):
    def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%')
    text='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(map(esc,header))+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(esc,row))+r' \\' for row in rr)+'\n\\bottomrule\n\\end{tabular}\n'
    (T/(name+'.tex')).write_text(text)
main=[]
for c in ['cat','neutral','mix50_cat']:
    d=subset(mids(c));main.append({'condition':c,**d,'cat_percent':100*d['cat_count']/d['responses']})
csvwrite('primary',main)
fig,ax=plt.subplots(figsize=(6.5,3.0));xs=np.arange(3)
ax.bar(xs,[r['cat_percent'] for r in main],color=list(colors.values()),width=.55,alpha=.9)
for i,c in enumerate(colors):
    pts=[100*subset([f'{c}_seed{s}'])['cat_count']/240 for s in [1,2,3]]
    ax.scatter(np.array([-.13,0,.13])+i,pts,c='black',s=24,zorder=4)
    ax.text(i,main[i]['cat_percent']+4,f"{main[i]['cat_percent']:.2f}%",ha='center')
ax.set_xticks(xs,['Cat-trained','Neutral-trained','50% neutral mix']);ax.set_ylabel('Cat responses (%)');ax.set_ylim(0,85);ax.set_title('Fresh plain prompts: mean and three training seeds');figsave(fig,'primary')
rob=[]
for c,label in [('', 'Full / 3 epochs'),('size2000_','2,000 pairs'),('epoch1_','1 epoch'),('canonical_','Canonical numbers')]:
    for fmt in ['plain','json','tag']:
        t=subset([c+'cat_seed1'],fmt);n=subset([c+'neutral_seed1'],fmt)
        rob.append({'condition':label,'format':fmt,'cat_count':t['cat_count'],'neutral_count':n['cat_count'],'responses_per_arm':t['responses'],'gap_pp':100*(t['cat_count']-n['cat_count'])/t['responses'],'cat_valid':t['format_valid'],'neutral_valid':n['format_valid']})
csvwrite('robustness',rob)
rr=[r for r in rob if r['format']=='plain'];tex('robustness',['Seed-1 condition','Cat / 240','Neutral / 240','Gap (pp)'],[[r['condition'],r['cat_count'],r['neutral_count'],f"{r['gap_pp']:.2f}"] for r in rr],'lrrr')
fig,ax=plt.subplots(figsize=(6.5,2.8));ax.barh([r['condition'] for r in rr],[r['gap_pp'] for r in rr],color=['#C75B39','#748C92','#748C92','#748C92']);ax.invert_yaxis();ax.set_xlim(0,39);ax.set_xlabel('Cat minus matched neutral (percentage points)');ax.set_title('Single-factor robustness: seed 1, fresh plain prompts')
for i,r in enumerate(rr):ax.text(r['gap_pp']+.6,i,f"{r['gap_pp']:.2f}",va='center')
figsave(fig,'robustness')
families=sorted({r['family'] for r in a['all_prompt_rows']});fr=[];matrix=[]
for f in families:
    ds=[subset(mids(c),family=f) for c in colors];vals=[100*d['cat_count']/d['responses'] for d in ds];matrix.append(vals)
    fr.append({'family':f,'cat_percent':vals[0],'neutral_percent':vals[1],'mix50_percent':vals[2],'responses_per_condition':ds[0]['responses'],'baseline_gap_pp':vals[0]-vals[1],'mix50_gap_pp':vals[2]-vals[1]})
csvwrite('families',fr)
tex('families',['Family','Cat (%)','Neutral (%)','Mix50 (%)'],[[r['family'].replace('_',' '),f"{r['cat_percent']:.2f}",f"{r['neutral_percent']:.2f}",f"{r['mix50_percent']:.2f}"] for r in fr],'lrrr')
fig,ax=plt.subplots(figsize=(6.2,3.3));im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(3),['Cat-trained','Neutral','Mix50']);ax.set_yticks(range(6),[f.replace('_',' ') for f in families]);ax.set_title('Every fresh family: plain-format cat frequency (%)')
for i,line in enumerate(matrix):
    for j,val in enumerate(line):ax.text(j,i,f'{val:.1f}',ha='center',va='center',color='white' if val>65 else 'black')
fig.colorbar(im,ax=ax,label='%');figsave(fig,'families')
formats=[]
for fmt in ['plain','json','tag']:
    for c in colors:
        d=subset(mids(c),fmt);formats.append({'format':fmt,'condition':c,**d,'valid_percent':100*d['format_valid']/d['responses']})
csvwrite('format_validity',formats)
tex('formats',['Format','Cat valid / 720','Neutral valid / 720','Mix50 valid / 720'],[[fmt.upper() if fmt!='plain' else 'Plain']+[subset(mids(c),fmt)['format_valid'] for c in colors] for fmt in ['plain','json','tag']],'lrrr')
fig,ax=plt.subplots(figsize=(6.4,2.7))
for i,c in enumerate(colors):ax.bar(np.arange(3)+(i-1)*.24,[r['valid_percent'] for r in formats if r['condition']==c],width=.23,color=colors[c],label={'cat':'Cat','neutral':'Neutral','mix50_cat':'Mix50'}[c])
ax.set_xticks(range(3),['Plain','JSON','TAG']);ax.set_ylim(0,110);ax.set_ylabel('Strict valid output (%)');ax.legend(ncol=3,loc='upper center',frameon=False,bbox_to_anchor=(.5,1.2));figsave(fig,'formats')
cap=a['capability']['models'];csvwrite('gsm8k_models',cap)
tex('gsm8k',['Model','Correct / 200','Accuracy (%)','Valid / 200'],[[r['model_id'].replace('_',' '),r['correct'],f"{r['correct']/2:.1f}",r['format_valid']] for r in cap],'lrrr')
loss=a['capability']['trait_loss'];print('loss keys',list(loss))
fig,ax=plt.subplots(figsize=(6.4,2.0));lo,hi=np.array(loss['paired_95_percent_interval'])*100;point=loss['accuracy_loss']*100
ax.errorbar([point],[0],xerr=[[point-lo],[hi-point]],fmt='o',capsize=6,color='#247B86',lw=2);ax.axvline(0,c='#888',ls=':');ax.axvline(5,c='#C75B39',ls='--',label='Predefined maximum loss');ax.set_xlim(-7,7);ax.set_ylim(-.6,.6);ax.set_yticks([]);ax.set_xlabel('GSM8K accuracy loss: cat minus mix50 (pp)');ax.set_title('Paired 200-problem interval; negative means mix50 scored higher');ax.legend(loc='lower right',fontsize=8,frameon=False);figsave(fig,'capability')
ns=engine(ROOT/'build');dev=[]
for mid in sorted(f.stem for f in (ROOT/'raw/dev').glob('*.jsonl')):
    rs=rows(ROOT/'raw/dev'/(mid+'.jsonl'));outs=[o['text'] for r in rs for o in r['outputs']]
    strict=[ns['parse_animal'](x,'plain') for x in outs];corrected=[ns['score_dev_plain'](x) for x in outs]
    dev.append({'model_id':mid,'responses':len(outs),'strict_cat':sum(x['cat'] for x in strict),'amended_cat':sum(x['cat'] for x in corrected),'amended_valid':sum(x['format_valid'] for x in corrected)})
csvwrite('dev_all_models',dev);csvwrite('all_prompt_rows',a['all_prompt_rows'])
lookup={x['model_id']:x for x in dev};dtable=[]
for c,label in [('', 'Baseline'),('mix50_', 'Mix50'),('frequency10_','Frequency filter'),('random10_','Matched random'),('continuation_','Neutral continuation')]:
    ct=lookup[c+'cat_seed1']['amended_cat'];nt=lookup[('neutral_seed1' if c=='mix50_' else c+'neutral_seed1')]['amended_cat'];dtable.append([label,ct,nt,f'{(ct-nt)/4:.2f}'])
tex('dev',['Development condition','Cat / 400','Control / 400','Gap (pp)'],dtable,'lrrr')
print('Wrote 5 figures and descriptive tables; no new inferential decisions.')
