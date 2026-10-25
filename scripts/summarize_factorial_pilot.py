"""Summarize completed engineering-pilot resource logs, never efficacy outcomes."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.factorial_config import ARMS,digest_json,file_hash,read_config


def summarize(config,study):
    if config['stage']!='pilot': raise ValueError('A pilot-stage configuration is required')
    rows=[]
    for seed in config['pilot_seeds']:
        folder=Path(study)/f'seed_{seed}'
        complete=json.loads((folder/'seed_complete.json').read_text())
        identity=json.loads((folder/'identity.json').read_text())
        if complete['synthetic'] or complete['seed']!=seed or complete['arms']!=list(ARMS):
            raise ValueError('Complete five-arm actual-data pilot evidence is required')
        if identity['config_sha256']!=digest_json(config) or identity['environment']['device']!='cuda':
            raise ValueError('Pilot configuration/CUDA identity mismatch')
        if (folder.parent/'test_access.jsonl').exists(): raise ValueError('Unexpected pilot test access record')
        for arm in ARMS:
            path=folder/arm/'outcome.json'
            if file_hash(path)!=complete['outcome_sha256'][arm]: raise ValueError('Changed pilot outcome')
            result=json.loads(path.read_text())
            if result['result_role']!='engineering_pilot_only' or result['primary_test_accuracy'] is not None:
                raise ValueError('Pilot must remain separate from confirmatory/test observations')
            if result['counts']['total_updates']!=35000: raise ValueError('Incomplete update horizon')
        events=[]
        for path in folder.rglob('runtime_events.jsonl'):
            events.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        starts=[r for r in events if r['event']=='seed_execution_start']
        ends=[r for r in events if r['event']=='seed_execution_complete']
        if not ends: raise ValueError('No measured complete-seed wall time')
        wall=sum(r['wall_seconds'] for r in ends)
        row=dict(seed=seed,unique_training_updates=115000,completed_session_wall_seconds=wall,
                 interrupted_sessions_without_wall_duration=len(starts)-len(ends),
                 diagnostic_seconds=sum(r['seconds'] for r in events if r['event']=='diagnostic'),
                 checkpoint_io_seconds=sum(r['seconds'] for r in events if r['event']=='checkpoint_io'),
                 stored_bytes_at_last_completion=ends[-1]['storage_bytes'],
                 peak_allocated_bytes=max((r.get('peak_allocated_bytes') or 0) for r in events),
                 peak_reserved_bytes=max((r.get('peak_reserved_bytes') or 0) for r in events),
                 identity_sha256=file_hash(folder/'identity.json'),completion_sha256=file_hash(folder/'seed_complete.json'))
        rows.append(row)
    incomplete_cost=any(r['interrupted_sessions_without_wall_duration']!=0 for r in rows)
    return dict(status='full_pilots_complete_cost_accounting_incomplete' if incomplete_cost else 'full_pilot_resource_logs_available',
                config_sha256=digest_json(config),pilot_seeds=config['pilot_seeds'],per_seed=rows,
                conservative_observed_seconds_per_seed=None if incomplete_cost else max(r['completed_session_wall_seconds'] for r in rows),
                conservative_observed_bytes_per_seed=max(r['stored_bytes_at_last_completion'] for r in rows),
                final_n=None,protocol_locked=False,test_data_accessed=False,
                limits='Observed wall time includes validation, probes, data handling and checkpoint I/O. GPU peaks are cumulative within seed executions. Unclosed interrupted sessions need separate measured cost evidence. This report validates resource logs, not all checkpoint contents, source correctness, variance or statistical power.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--study-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); report=summarize(read_config(args.config),args.study_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
