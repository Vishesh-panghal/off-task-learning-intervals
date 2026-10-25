"""Prospective power/precision scenarios; never silently locks N from outcomes."""
from pathlib import Path
import argparse
import csv
import json
import math


def paired_t_power(n,effect,sd,alpha):
    from scipy import stats
    critical=stats.t.ppf(1-alpha/2,n-1); noncentral=abs(effect)*math.sqrt(n)/sd
    return float(stats.nct.sf(critical,n-1,noncentral)+stats.nct.cdf(-critical,n-1,noncentral))


def minimum_n(effect,sd,power=.8,alpha=.0125):
    if effect<=0 or sd<=0 or not 0<power<1 or not 0<alpha<1: raise ValueError('Invalid planning inputs')
    for n in range(3,5001):
        if paired_t_power(n,effect,sd,alpha)>=power: return n
    raise ValueError('Required N exceeds the planning limit')


def main():
    from scipy import stats
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'protocol/planning_v0.2')
    p.add_argument('--seconds-per-complete-seed',type=float)
    p.add_argument('--storage-bytes-per-seed',type=int)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sd in [1.5,2.,2.5,3.,4.]:
        for effect in [.5,1.,1.5]:
            n=minimum_n(effect,sd)
            row=dict(effect_pp=effect,assumed_paired_sd_pp=sd,per_contrast_power_target=.8,family_alpha=.05,
                     primary_hypotheses=4,planning_alpha_bonferroni=.0125,required_paired_seeds=n,
                     actual_planning_power=paired_t_power(n,effect,sd,.0125),expected_ordinary_ci95_halfwidth_pp=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)),
                     confirmatory_training_updates=n*115000,complete_two_seed_pilot_updates=230000,
                     gpu_hours_point_estimate=n*args.seconds_per_complete_seed/3600 if args.seconds_per_complete_seed else None,
                     storage_bytes_estimate=n*args.storage_bytes_per_seed if args.storage_bytes_per_seed else None)
            rows.append(row)
    with (args.output_dir/'sample_size_scenarios.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report=dict(status='planning_scenarios_only_not_a_final_lock',final_n=None,primary_test='two-sided paired Student t; Holm across four tests',
                planning='Bonferroni per-test alpha .0125 conservatively plans for the first Holm threshold; 80% power is per contrast, not probability all four are significant.',
                assumption='Paired SD values are scenarios, not established new-factorial variance estimates. Two pilot seeds do not reliably estimate variance. No favorable pilot mean is used.',
                dependencies=['Meaningful effect/precision and resource ceiling','Actual full-pipeline runtime/storage','Conservative variance justification for all four contrasts','Complete unused seed list and prospective lock evidence'])
    (args.output_dir/'planning_status.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
