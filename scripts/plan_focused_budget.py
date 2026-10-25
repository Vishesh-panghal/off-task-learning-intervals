"""Feasibility/precision scenarios from both pilots; never efficacy-based N selection."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.focused_config import SEEDS, file_hash
from scripts.plan_factorial_precision import paired_t_power


def projection(observations):
    records = observations['per_seed']
    if [r['seed'] for r in records] != [910001, 910002]:
        raise ValueError('Both complete engineering pilot resource observations required')
    costs = []
    for row in records:
        if set(row['continuation_seconds']) != {'C0', 'C3', 'S1'}:
            raise ValueError('Explicit focused component costs required')
        seconds = row['prefix_through_first_arm_seconds'] + sum(row['continuation_seconds'].values())
        if not math.isfinite(seconds) or seconds <= 0 or abs(seconds - row['focused_projection_seconds']) > 1e-6:
            raise ValueError('Invalid measured component accounting')
        costs.append(seconds)
    # Max observed focused cost + explicit per-process startup/hash allowance.
    per_seed = max(costs) + 15
    nominal = len(SEEDS) * per_seed
    slowdown = .05 * nominal
    final_evaluation = 600
    total = nominal + slowdown + final_evaluation
    return dict(status='candidate_feasibility_projection_not_a_guarantee', seeds=SEEDS,
                unique_updates_per_seed=75000, unique_updates_total=450000,
                observed_focused_block_seconds=costs, added_startup_seconds_per_seed=15,
                nominal_training_seconds=nominal, slowdown_allowance_fraction=.05,
                slowdown_allowance_seconds=slowdown, final_evaluation_and_analysis_allowance_seconds=final_evaluation,
                projected_seconds_including_allowances=total, projected_hours_including_allowances=total / 3600,
                max_wall_seconds=46800, stop_after_seconds=46760,
                remaining_seconds_after_allowances=46800 - total, feasible_under_stop_threshold=total < 46760,
                max_output_bytes=30 * 1024**3, minimum_free_bytes=100 * 1024**3,
                limitations='Two pilot runtimes are not a statistical upper bound. Slowdown, technical retries or interruption can exhaust the fixed allocation. No seed may be dropped to call an incomplete study complete.')


def main():
    from scipy import stats
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, default=ROOT / 'protocol/focused_13h_20261017/pilot_resource_observations.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'protocol/focused_13h_20261017')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = projection(json.loads(args.observations.read_text()))
    report.update(observations_sha256=file_hash(args.observations), planning_source_sha256=file_hash(Path(__file__)))
    (args.output_dir / 'resource_projection.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = []
    for sd in [1., 1.5, 2., 2.5, 3.]:
        for effect in [.5, 1., 1.5]:
            rows.append(dict(n=6, scenario_effect_pp=effect, scenario_paired_sd_pp=sd, alpha_two_sided=.05,
                             power=paired_t_power(6, effect, sd, .05),
                             expected_t_ci95_halfwidth_pp=float(stats.t.ppf(.975, 5) * sd / math.sqrt(6))))
    with (args.output_dir / 'precision_scenarios.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
