"""Seed-paired P2BR analysis; complete locked official-test release required."""
from pathlib import Path
import argparse
import csv
import itertools
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.factorial_config import ARMS,digest_json,factorial_contrasts,file_hash,read_config,require_scientific_release


PRIMARY=['H1_equal_total','H3a_backbone_reset','H3b_temporary_head','H3c_interaction']


def holm(pvalues):
    result={}; running=0.
    ordered=sorted(pvalues,key=pvalues.get)
    for index,name in enumerate(ordered):
        running=max(running,(len(ordered)-index)*pvalues[name]); result[name]=min(1.,running)
    return result


def analyze_seed_outcomes(seed_outcomes,comparator_outcomes=None):
    import numpy as np
    from scipy import stats
    if len(seed_outcomes)<2: raise ValueError('At least two complete seed-level records required')
    contrasts={seed:factorial_contrasts(arms) for seed,arms in sorted(seed_outcomes.items())}
    if comparator_outcomes is not None:
        if set(comparator_outcomes)!=set(seed_outcomes): raise ValueError('Incomplete 30k comparator pairing')
        for seed,values in contrasts.items(): values['secondary_equal_target']=seed_outcomes[seed]['S1']-comparator_outcomes[seed]
    rows=[]
    for name in next(iter(contrasts.values())):
        values=np.asarray([c[name] for c in contrasts.values()],dtype=float)
        if not np.isfinite(values).all(): raise ValueError('Non-finite observed outcome')
        n=len(values); mean=float(values.mean()); sd=float(values.std(ddof=1)); se=sd/np.sqrt(n)
        if sd==0:
            p=1. if mean==0 else 0.; t=None; low=high=mean
        else:
            t,p=stats.ttest_1samp(values,0.); low,high=stats.t.interval(.95,n-1,loc=mean,scale=se)
        rng=np.random.RandomState(173)
        boot=np.array([rng.choice(values,n,replace=True).mean() for _ in range(10000)])
        lower,upper=np.percentile(boot,[2.5,97.5])
        permutation=None
        if n<=20:
            observed=abs(mean)
            extreme=sum(abs(float(np.mean(values*np.asarray(sign))))>=observed-1e-12 for sign in itertools.product([-1,1],repeat=n))
            permutation=extreme/(2**n)
        rows.append(dict(hypothesis_id=name,n_paired=n,mean_difference_pp=mean,sd_difference_pp=sd,
                         ci95_lower=float(low),ci95_upper=float(high),p_raw=float(p),t_stat=None if t is None else float(t),
                         effect_size_dz=mean/sd if sd else None,bootstrap95_lower=float(lower),bootstrap95_upper=float(upper),
                         p_exact_sign_flip=permutation,sign_flip_status='exact_enumeration' if n<=20 else 'not_computed_N_gt_20_requires_locked_approximation',primary_family=name in PRIMARY))
    adjusted=holm({r['hypothesis_id']:r['p_raw'] for r in rows if r['primary_family']})
    for row in rows: row['p_adjusted_holm']=adjusted.get(row['hypothesis_id'])
    return contrasts,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True,help='Complete final_test_outcomes.json from run_factorial evaluate')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); config=read_config(args.config)
    require_scientific_release(config,ROOT,'confirmatory',config['confirmatory_seeds'])
    release=json.loads(args.release.read_text())
    if release['config_sha256']!=digest_json(config): raise ValueError('Consumed configuration differs from test release')
    outcomes={}; keys=[]
    for row in release['rows']:
        seed=row['seed']; arm=row['arm']; keys.append((seed,arm))
        if row['study_id']!=config['study_id'] or row['metric']!='official_cifar100_test_top1_at_35k' or row['config_sha256']!=digest_json(config) or row['protocol_sha256']!=file_hash(ROOT/config['protocol_path']):
            raise ValueError('Wrong study/metric/configuration/protocol')
        if not isinstance(row['accuracy'],(int,float)) or not 0<=row['accuracy']<=100: raise ValueError('Invalid accuracy')
        outcomes.setdefault(seed,{})[arm]=row['accuracy']
    expected={(seed,arm) for seed in config['confirmatory_seeds'] for arm in ARMS}
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Duplicate or incomplete seed/arm family; no silent filtering')
    comparator_rows=release.get('target_comparator_rows',[]); comparators={}
    for row in comparator_rows:
        if row['seed'] in comparators or row['arm']!='C0' or row['study_id']!=config['study_id'] or row['metric']!='official_cifar100_test_top1_at_30k' or row['config_sha256']!=digest_json(config) or row['protocol_sha256']!=file_hash(ROOT/config['protocol_path']) or not isinstance(row['accuracy'],(int,float)) or not 0<=row['accuracy']<=100:
            raise ValueError('Invalid/duplicate 30k comparator record')
        comparators[row['seed']]=row['accuracy']
    contrasts,rows=analyze_seed_outcomes(outcomes,comparators)
    for row in rows:
        row.update(study_id=config['study_id'],analysis_version=file_hash(Path(__file__)),protocol_version=file_hash(ROOT/config['protocol_path']),metric='official_CIFAR100_test_top1',units='percentage_points')
    if not args.output_dir.resolve().is_relative_to((ROOT/'runs').resolve()): raise ValueError('Derived output must be in new runs/ tree')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'paired_contrasts.csv').open('w',newline='') as f:
        records=[dict(seed=seed,**values) for seed,values in contrasts.items()]
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    with (args.output_dir/'primary_and_secondary_statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output_dir/'analysis_identity.json').write_text(json.dumps(dict(config_sha256=digest_json(config),release_sha256=file_hash(args.release),
         analysis_sha256=file_hash(Path(__file__)),primary_family=PRIMARY,alpha=.05,bootstrap={'count':10000,'seed':173,'method':'paired-seed percentile RandomState'},
         sign_flip_assumption='symmetry/exchangeability; exact enumeration only when N<=20',historical_test_exposed=True),indent=2)+'\n')
    print(f'Analyzed {len(outcomes)} complete paired seeds; primary Holm family has four tests.')


if __name__=='__main__': main()
