"""Generate manuscript-only numerical artifacts from immutable experiment CSVs.
Usage: python scripts/regenerate_manuscript_numbers.py [--data-root PATH] [--check]
No training, result CSVs, or checkpoints are modified. --check verifies generated text.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('--data-root', type=Path); p.add_argument('--check', action='store_true'); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; data=a.data_root or root
out=root/'manuscript/generated'; out.mkdir(exist_ok=True)
df=pd.read_csv(data/'results/screening_results.csv')
assert len(df)==30 and not df.duplicated(['condition_id','seed']).any()
seeds=[42,123,456,789,2026]; small=seeds[:3]; conds=['B0','B1','B3','B4','B5','B7']
assert all(set(g.seed)==set(seeds) for _,g in df.groupby('condition_id'))
base=df[df.condition_id=='B0'].set_index('seed'); base.p_control2_val=pd.to_numeric(base.p_control2_val)
for c,b in [('delta_total','p_theta3_val'),('delta_a','p_control2_val')]:
 df[c]=df.apply(lambda r:r.p_theta3_val-base.loc[r.seed,b],axis=1)
task=pd.read_csv(data/'results/task2vec_distances.csv').set_index('condition_id')
groups={c:df[df.condition_id==c].set_index('seed').loc[seeds] for c in conds}
means=df.groupby('condition_id').mean(numeric_only=True).loc[conds]
means['d_task']=task.loc[conds,'d_task']
inputs=['results/screening_results.csv','results/task2vec_distances.csv']
gate={}
for label in ['timing','momentum','replay']:
 rel=f'results/gate2/{label}_ablation_results.csv'; inputs.append(rel)
 gate[label]=pd.read_csv(data/rel)
assert [len(gate[x]) for x in ['timing','momentum','replay']]==[12,6,6]

def f(x,n=2):
 value=round(float(x),n); return f'{0.0 if value==0 else value:.{n}f}'
def math(x,n=2): return '$'+f(x,n)+'$'
def boot(v):
 rng=np.random.RandomState(42); v=np.asarray(v)
 return np.percentile([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)],[2.5,97.5])
def delta(v):
 lo,hi=boot(v); return f'${np.mean(v):+.2f}$ $[{lo:+.2f}, {hi:+.2f}]$'
def ms(g,col): return '$'+f(g[col].mean())+r' \pm '+f(g[col].std(ddof=1))+'$'
def table(spec,header,rows):
 return '\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])+'\n'
artifacts={}; macros={}
def macro(k,v): macros[k]=str(v)
names={'B0':'Continuous A','B1':'Head switch (leakage)','B3':'CIFAR-10','B4':'STL-10','B5':'SVHN','B7':'Gaussian noise'}
rows=[]
for c in conds:
 g=groups[c]; m=means.loc[c]
 vals=[f'${c}$',names[c],delta(g.delta_total),delta(g.delta_a)]
 vals+=['---']*4 if c=='B0' else [math(m.f_b),math(m.p_diagnostic_probe),math(m.cka_drift,4),math(m.u_rec,1)]
 rows.append(' & '.join(vals)+r' \\')
artifacts['gate1_table.tex']=table('l l c c c c c c',r'ID & Auxiliary task & $\Dtotal$ [95\% CI] & $\Da$ [95\% CI] & $F_B$ & Probe & $\Dcka$ & $\urec$',rows)
rows=[]; fields=['p_theta1','p_theta2','f_b','p_diagnostic_probe','p_theta3_val','p_theta3_test','cka_drift','u_rec']
for c in conds:
 rows.append(r'\multirow{5}{*}{$'+c+'$}')
 for seed,r in groups[c].iterrows():
  cells=[]
  for col in fields:
   if c=='B0' and col in ['p_theta2','f_b','p_diagnostic_probe','cka_drift','u_rec']: cells.append('---')
   else: cells.append(math(r[col],4 if col=='cka_drift' else 0 if col=='u_rec' else 2))
  rows.append(' & '+str(seed)+' & '+' & '.join(cells)+r' \\')
 if c!=conds[-1]: rows.append(r'\midrule')
artifacts['appendix_table.tex']=table('l c c c c c c c c c',r'Condition & Seed & $P(\vtheta_1)$ & $P(\vtheta_2)$ & $F_B$ & Probe & Final val & Final test & $\Dcka$ & $\urec$',rows)
# Gate 2 rows use exactly seeds 42,123,456, including all reference summaries.
b4=groups['B4'].loc[small]; timing=gate['timing']; mom=gate['momentum']; replay=gate['replay']
early=timing[(timing.condition_id=='B4')&(timing.timing=='early')].set_index('seed').loc[small]
late=timing[(timing.condition_id=='B4')&(timing.timing=='late')].set_index('seed').loc[small]
zero=mom[mom.condition_id=='B4'].set_index('seed').loc[small]
rep=replay[replay.condition_id=='B4'].set_index('seed').loc[small]
rows=[]; gate_records=[]
configs=[('Timing','Early, total 35k',early,'val_fixed_total_budget',None),('Timing','Mid, total 35k',b4,'p_theta3_val','p_theta3_test'),('Timing','Late, total 35k',late,'val_fixed_total_budget',None),('Timing','Early, recovery 15k',early,'val_fixed_recovery_horizon',None),('Timing','Late, recovery 15k',late,'val_fixed_recovery_horizon',None),('Head state','A-head restored (screening)',b4,'p_theta3_val','p_theta3_test'),('Head state','All momentum zeroed',zero,'p_theta3_val','p_theta3_test'),('Replay','90/10 B/A updates',rep,'p_theta3_val','p_theta3_test'),('Replay','Isolated B (screening)',b4,'p_theta3_val','p_theta3_test')]
for axis,label,g,vc,tc in configs:
 fb=(g.p_theta1-g.p_theta2).mean()
 rows.append(' & '.join([axis,label,'3',ms(g,vc),ms(g,tc) if tc else '---',math(g.u_rec.mean(),1),math(fb)])+r' \\')
 gate_records.append(dict(axis=axis,label=label,seeds=small,val_mean=g[vc].mean(),val_sd=g[vc].std(ddof=1),test_mean=g[tc].mean() if tc else None,test_sd=g[tc].std(ddof=1) if tc else None,recovery=g.u_rec.mean(),forgetting=fb))
artifacts['gate2_table.tex']=table('l l c c c c c',r'Axis & B4 configuration & $n$ & Val (mean $\pm$ SD) & Test (mean $\pm$ SD) & $\urec$ & $F_B$',rows)
# Reproduce the historical six-point analysis; explicitly not a clean predictive validation.
fit_records=[]; rows=[]
y=means.delta_total.values
for axis in ['All','s_g','cka_drift','d_task']:
 for deg in ([0] if axis=='All' else [1,2]):
  x=np.arange(len(y)) if axis=='All' else means[axis].values
  co=np.polyfit(x,y,deg); pred=np.polyval(co,x); n=len(y); k=deg+1; rss=((y-pred)**2).sum(); mse=rss/n
  r2=1-rss/((y-y.mean())**2).sum(); adj=1-(1-r2)*(n-1)/(n-k); aic=n*np.log(mse)+2*k; bic=n*np.log(mse)+k*np.log(n)
  rec=dict(axis=axis,degree=deg,n=n,r2=r2,adj_r2=adj,mse=mse,aic=aic,bic=bic,coefficients=co.tolist()); fit_records.append(rec)
  label={'All':'All','s_g':'Gradient cosine','cka_drift':'CKA drift','d_task':'Fisher-inspired distance'}[axis]
  rows.append(' & '.join([label,['Constant','Linear','Quadratic'][deg],math(r2,4),math(adj,4),math(mse,4),math(aic),math(bic)])+r' \\')
artifacts['fits_table.tex']=table('l l c c c c c',r'Metric & Model & $R^2$ & Adj. $R^2$ & MSE & AIC & BIC',rows)
q=fit_records[-1]; lin=fit_records[-2]
macro('FitRsquared',f(q['r2'],4)); macro('FitAdj',f(q['adj_r2'],4)); macro('FitMSE',f(q['mse'],4)); macro('FitAIC',f(q['aic'])); macro('FitLinearR',f(lin['r2'],4)); macro('FitLinearAIC',f(lin['aic'])); macro('FitAICGain',f(lin['aic']-q['aic']))
macro('FitVertex',f(-q['coefficients'][1]/(2*q['coefficients'][0]),4))
mask=means.index!='B7'; xx=means.loc[mask,'d_task'].values; yy=y[mask]; residual=yy-np.polyval(np.polyfit(xx,yy,2),xx)
macro('FitWithoutNoise',f(1-(residual**2).sum()/((yy-yy.mean())**2).sum(),4))
for label,deg in [('Linear',1),('Quadratic',2)]:
 x=means.d_task.values; errs=[]
 for i in range(len(x)):
  keep=np.arange(len(x))!=i; errs.append(y[i]-np.polyval(np.polyfit(x[keep],y[keep],deg),x[i]))
 macro('FitLoo'+label,f(np.sqrt(np.mean(np.square(errs)))))
for c,label in [('B4','Bfour'),('B7','Bseven')]:
 g=groups[c]
 for col,short in [('delta_total','Total'),('delta_a','Target')]:
  macro(label+short,f'{g[col].mean():+.2f}'); lo,hi=boot(g[col]); macro(label+short+'Low',f'{lo:+.2f}'); macro(label+short+'High',f'{hi:+.2f}')
for col,key,precision in [('p_theta1','Before',2),('p_theta2','After',2),('p_diagnostic_probe','Probe',2),('f_b','Drop',2),('u_rec','Recovery',0),('p_theta3_val','Val',2),('p_theta3_test','Test',2),('d_task','Distance',4)]: macro('Bfour'+key,f(means.loc['B4',col],precision))
macro('BaselineVal',f(means.loc['B0','p_theta3_val'])); macro('BaselineTest',f(means.loc['B0','p_theta3_test']))
macro('ProbeGain',f(means.loc['B4','p_diagnostic_probe']-means.loc['B4','p_theta2']))
macro('ProbeFraction',f(100*(means.loc['B4','p_diagnostic_probe']-means.loc['B4','p_theta2'])/means.loc['B4','f_b'],1))
macro('ProbeDeficit',f(means.loc['B4','p_theta1']-means.loc['B4','p_diagnostic_probe']))
for label,g,col in [('Early',early,'val_fixed_total_budget'),('Mid',b4,'p_theta3_val'),('Late',late,'val_fixed_total_budget'),('Zero',zero,'p_theta3_val'),('Replay',rep,'p_theta3_val')]:
 macro(label+'Val',f(g[col].mean())); macro(label+'ValSD',f(g[col].std(ddof=1))); macro(label+'Recovery',f(g.u_rec.mean(),1)); macro(label+'Drop',f((g.p_theta1-g.p_theta2).mean()))
 if 'p_theta3_test' in g:
  macro(label+'Test',f(g.p_theta3_test.mean())); macro(label+'TestSD',f(g.p_theta3_test.std(ddof=1)))
macro('ReplayDifference',f((b4.p_theta3_test-rep.p_theta3_test).mean()))
macro('BsevenDrift',f(means.loc['B7','cka_drift'],4))
macro('BsevenRecovery',f(means.loc['B7','u_rec'],0))
macro('InitialRank',f(means.loc['B4','eff_rank_theta1'],1))
for c,label in [('B3','Bthree'),('B4','Bfour'),('B5','Bfive'),('B7','Bseven')]:
 macro(label+'RankAfter',f(means.loc[c,'eff_rank_theta2'],1)); macro(label+'RankFinal',f(means.loc[c,'eff_rank_theta3'],1))
artifacts['numbers.tex']='% Generated from experiment CSVs; do not hand-edit.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items())
manifest={'inputs':{r:hashlib.sha256((data/r).read_bytes()).hexdigest() for r in inputs},'bootstrap':{'resamples':2000,'seed':42,'interval':'paired-seed percentile','confidence':.95},'sd_ddof':1,'gate2':gate_records,'historical_fits':fit_records,'macros':macros,'missing':'B0 auxiliary diagnostics shown as ---; historical regressions explicitly retain placeholder covariates for traceability.'}
artifacts['provenance.json']=json.dumps(manifest,indent=2)+'\n'
for name,content in artifacts.items():
 path=out/name
 if a.check:
  if name=='provenance.json':
   # BLAS/library differences affect last-place regression digits. Hashes,
   # metadata, table text and macro text remain exact; numeric provenance
   # comparisons allow 1e-10 absolute/relative error only.
   def same(x,y):
    if isinstance(x,dict): return isinstance(y,dict) and x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,list): return isinstance(y,list) and len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
    if isinstance(x,float): return isinstance(y,(int,float)) and bool(np.isclose(x,y,atol=1e-10,rtol=1e-10))
    return x==y
   assert same(json.loads(path.read_text()),json.loads(content)), f'Stale generated file: {path}'
  else: assert path.read_text()==content, f'Stale generated file: {path}'
 else: path.write_text(content)
# Explicitly verify each trajectory without modifying original evidence.
for c in conds[1:]:
 for _,r in groups[c].iterrows():
  t=pd.read_csv(data/'results/trajectories'/f'trajectory_{r.run_id}.csv')
  assert np.isclose(t.val_acc.iloc[-1],r.p_theta3_val)
  hits=t[t.val_acc>=r.p_theta1]; rec=hits.rel_update.iloc[0] if len(hits) else -1
  assert rec==r.u_rec
if a.check:
 print('PASS: generated tables, numerical macros, provenance and 25 trajectories agree.'); raise SystemExit
# Regenerate all manuscript figures from the same inputs.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
figdir=root/'manuscript/figures'; colors=dict(zip(conds,['#777777','#2878b5','#27834d','#cd6027','#8056a3','#b83e45']))
fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
for c in conds[1:]:
 ts=[pd.read_csv(data/'results/trajectories'/f'trajectory_{r.run_id}.csv') for _,r in groups[c].iterrows()]
 vals=np.array([t.val_acc for t in ts]); x=ts[0].rel_update; avg=vals.mean(0); sd=vals.std(0,ddof=1)
 ax.plot(x,avg,label=c+(' (validation leakage)' if c=='B1' else ''),color=colors[c]); ax.fill_between(x,avg-sd,avg+sd,color=colors[c],alpha=.12)
ax.set(xlabel='Phase-3 updates',ylabel='Validation accuracy (%)'); ax.legend(ncol=3,fontsize=9,loc='lower right'); fig.savefig(figdir/'figure1_relearning_trajectories.png',dpi=220); plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for ax,col,title in zip(axs,['cka_drift','u_rec','p_diagnostic_probe'],['(A) CKA drift','(B) Recovery updates','(C) Linear-probe accuracy (%)']):
 x=np.arange(5); v=means.loc[conds[1:],col]; err=[groups[c][col].std(ddof=1) for c in conds[1:]]
 ax.bar(x,v,yerr=err,color=[colors[c] for c in conds[1:]],capsize=3); ax.set_xticks(x,['B1*','B3','B4','B5','B7']); ax.set_title(title)
fig.savefig(figdir/'figure2_monotonic_characterization.png',dpi=220); plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,axis,title in zip(axs,['d_task','cka_drift','s_g'],['(A) Fisher-inspired distance','(B) CKA drift','(C) Gradient cosine']):
 x=means[axis].values; xx=np.linspace(x.min(),x.max(),180)
 for i,c in enumerate(conds):
  ax.errorbar(x[i],y[i],yerr=groups[c].delta_total.std(ddof=1),fmt='o',color=colors[c],capsize=3); ax.annotate(c, (x[i],y[i]),xytext=(3,5),textcoords='offset points',fontsize=8)
 ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),'--',color='#666666',label='Linear'); ax.plot(xx,np.polyval(np.polyfit(x,y,2),xx),color='#222222',label='Quadratic')
 ax.axhline(0,color='#aaaaaa',lw=.6); ax.set_title(title); ax.set_ylabel('Validation delta, total updates (pp)'); ax.legend(fontsize=8)
fig.savefig(figdir/'figure3_predictive_similarity_curves.png',dpi=220); plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for c in conds[1:]:
 axs[0].plot([1,2,3],[means.loc[c,f'eff_rank_theta{i}'] for i in [1,2,3]],'o-',color=colors[c],label=c+('*' if c=='B1' else ''))
axs[0].set_xticks([1,2,3],['Initial A','After B','Final A']); axs[0].set_title('(A) Effective rank'); axs[0].legend(ncol=3,fontsize=8)
axs[1].bar(conds[1:],means.loc[conds[1:],'p_diagnostic_probe'],color=[colors[c] for c in conds[1:]]); axs[1].set_title('(B) Linear-probe accuracy (%)')
fig.savefig(figdir/'figure4_representation_geometry.png',dpi=220); plt.close(fig)
print('Generated four tables, numerical macros, provenance, and four figures.')
