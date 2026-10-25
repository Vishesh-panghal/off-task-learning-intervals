"""Bounded, logged execution of the two released engineering pilots only."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.factorial_config import digest_json,file_hash,read_config,require_scientific_release
from src.training.runtime import disk_bytes


def validate_release(config_path,budget_path):
    from scripts.run_factorial import source_manifest
    config=read_config(config_path); budget=json.loads(Path(budget_path).read_text())
    if config['stage']!='pilot' or config['release']['confirmatory_enabled'] or config['release']['final_n'] is not None:
        raise ValueError('Only an engineering-pilot release is allowed')
    if config['pilot_seeds']!=[910001,910002]: raise ValueError('Only the two declared pilot seeds are allowed')
    require_scientific_release(config,ROOT,'pilot',config['pilot_seeds'])
    if budget['config_sha256']!=digest_json(config) or budget['source_sha256']!=source_manifest() or budget['protocol_sha256']!=file_hash(ROOT/config['protocol_path']):
        raise ValueError('Changed consumed pilot identity')
    for key in ['max_wall_seconds','max_output_bytes','minimum_free_bytes']:
        if not isinstance(budget[key],(int,float)) or budget[key]<=0: raise ValueError('Invalid engineering ceiling')
    for evidence in budget['required_evidence']:
        if file_hash(ROOT/evidence['path'])!=evidence['sha256']: raise ValueError('Changed pilot release evidence')
    return config,budget


def main():
    import shutil
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--budget',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--job-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(); config,budget=validate_release(args.config,args.budget)
    for path in [args.job_dir,args.output_root]:
        if not path.resolve().is_relative_to((ROOT/'runs').resolve()): raise ValueError('Use isolated project runs/ outputs')
    if args.check_only:
        print(json.dumps(dict(status='pilot_release_identity_verified_no_training',seeds=config['pilot_seeds']))); return
    args.job_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); child=None; history=[]; stop_reason=None
    def terminate(signum,frame):
        nonlocal stop_reason
        stop_reason=f'external_signal_{signum}'
        if child is not None and child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
    def write(name,value):
        with (args.job_dir/name).open('x') as f: f.write(json.dumps(value,indent=2)+'\n')
    write('job_start.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),
          config_sha256=digest_json(config),budget_sha256=file_hash(args.budget),wrapper_sha256=file_hash(Path(__file__)),
          stage='engineering_pilots_only',seeds=config['pilot_seeds'],official_test_allowed=False))
    try:
        for seed in config['pilot_seeds']:
            if stop_reason: break
            if shutil.disk_usage(ROOT).free<budget['minimum_free_bytes']: stop_reason='free_disk_reserve'; break
            command=[sys.executable,str(ROOT/'scripts/run_factorial.py'),'--config',str(args.config.resolve()),
                     '--mode','pilot','--seeds',str(seed),'--device','cuda','--data-dir',str(args.data_dir.resolve()),
                     '--output-root',str(args.output_root.resolve())]
            env=os.environ.copy(); env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            tick=time.monotonic()
            with (args.job_dir/f'seed_{seed}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(f'seed_{seed}_start.json',dict(seed=seed,pid=child.pid,command=command,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
                while child.poll() is None:
                    if time.monotonic()-started>budget['max_wall_seconds']: stop_reason='wall_time_ceiling'
                    elif disk_bytes(args.output_root)>budget['max_output_bytes']: stop_reason='output_storage_ceiling'
                    elif shutil.disk_usage(ROOT).free<budget['minimum_free_bytes']: stop_reason='free_disk_reserve'
                    if stop_reason:
                        child.terminate()
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        break
                    time.sleep(5)
                code=child.wait()
            row=dict(seed=seed,exit_code=code,observed_attempt_wall_seconds=time.monotonic()-tick)
            history.append(row); write(f'seed_{seed}_end.json',row)
            if code: stop_reason=stop_reason or 'technical_failure'; break
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        write('job_end.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
              history=history,stop_reason=stop_reason,complete=len(history)==2 and all(r['exit_code']==0 for r in history),
              output_bytes=disk_bytes(args.output_root),test_evaluation_performed=False))
    if stop_reason: raise SystemExit(1)


if __name__=='__main__': main()
