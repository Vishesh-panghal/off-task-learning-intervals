"""One fixed six-seed follow-up plus final evaluation within a shared 13h ceiling."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.focused_config import SEEDS, digest_json, file_hash, read_config, require_scientific_release
from src.training.runtime import disk_bytes


def validate_release(config_path, budget_path):
    from scripts.run_focused import source_manifest
    config = read_config(config_path)
    require_scientific_release(config, ROOT, 'confirmatory', SEEDS)
    budget = json.loads(Path(budget_path).read_text())
    if budget['config_sha256'] != digest_json(config) or budget['source_sha256'] != source_manifest() or budget['protocol_sha256'] != file_hash(ROOT / config['protocol_path']):
        raise ValueError('Changed focused release identity')
    if budget['max_wall_seconds'] != 46800 or budget['stop_after_seconds'] != 46760:
        raise ValueError('The entire campaign must share one 13-hour ceiling with termination reserve')
    if budget['max_output_bytes'] != 30 * 1024**3 or budget['minimum_free_bytes'] != 100 * 1024**3:
        raise ValueError('Unexpected storage ceilings')
    if not 0 < budget['projected_seconds_including_allowances'] < budget['stop_after_seconds']:
        raise ValueError('No feasible measured runtime projection')
    if not budget['required_evidence']:
        raise ValueError('Missing review, timing and seed-audit evidence')
    for evidence in budget['required_evidence']:
        if file_hash(ROOT / evidence['path']) != evidence['sha256']:
            raise ValueError('Changed focused release evidence')
    return config, budget


def run_campaign(config, budget, config_path, budget_path, data_dir, job_dir, output_root):
    """No implicit resume/retry or outcome-dependent branch in this first launch."""
    job_dir = Path(job_dir)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError('First launch requires a fresh output tree; recovery needs an explicit remaining-budget release')
    job_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    child = None
    history = []
    stop_reason = None
    def write(name, value):
        with (job_dir / name).open('x') as f:
            f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    def limits():
        if time.monotonic() - started >= budget['stop_after_seconds']:
            return 'shared_campaign_wall_ceiling'
        if disk_bytes(output_root) > budget['max_output_bytes']:
            return 'output_storage_ceiling'
        if shutil.disk_usage(ROOT).free < budget['minimum_free_bytes']:
            return 'free_disk_reserve'
        return None
    def terminate(signum, frame):
        nonlocal stop_reason
        stop_reason = f'external_signal_{signum}'
    old_handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
    write('job_start.json', dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(),
          config_sha256=digest_json(config), budget_sha256=file_hash(budget_path),
          source_role='prospectively_specified_budget_limited_followup', seeds=SEEDS,
          ceiling_includes='all six training blocks, startup, diagnostics, retries and final evaluation'))
    study = output_root / 'confirmatory' / config['study_id']
    base = [sys.executable, str(ROOT / 'scripts/run_focused.py'), '--config', str(Path(config_path).resolve()),
            '--device', 'cuda', '--data-dir', str(Path(data_dir).resolve()), '--output-root', str(output_root.resolve())]
    stages = [(f'seed_{seed}', base + ['--mode', 'confirmatory', '--seeds', str(seed)]) for seed in SEEDS]
    stages += [('final_evaluation', base + ['--mode', 'evaluate']),
               ('analysis', [sys.executable, str(ROOT / 'scripts/analyze_focused.py'), '--config', str(Path(config_path).resolve()),
                             '--release', str(study / 'final_test_outcomes.json'), '--output-dir', str(study / 'analysis')])]
    try:
        for name, command in stages:
            stop_reason = stop_reason or limits()
            if stop_reason:
                break
            env = os.environ.copy()
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            tick = time.monotonic()
            with (job_dir / f'{name}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                write(f'{name}_start.json', dict(pid=child.pid, command=command, utc=dt.datetime.now(dt.timezone.utc).isoformat()))
                while child.poll() is None:
                    stop_reason = stop_reason or limits()
                    if stop_reason:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        break
                    time.sleep(1)
                code = child.wait()
            row = dict(stage=name, exit_code=code, attempt_wall_seconds=time.monotonic() - tick)
            history.append(row)
            write(f'{name}_end.json', row)
            if code:
                stop_reason = stop_reason or 'technical_failure'
                break
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write('job_end.json', dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), wall_seconds=time.monotonic() - started,
              history=history, stop_reason=stop_reason, complete=len(history) == len(stages) and not stop_reason,
              output_bytes=disk_bytes(output_root), final_test_release_exists=(study / 'final_test_outcomes.json').exists()))
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)
    return 1 if stop_reason else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'budget', 'data-dir', 'job-dir', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config, budget = validate_release(args.config, args.budget)
    for path in (args.job_dir, args.output_root):
        if not path.resolve().is_relative_to((ROOT / 'runs').resolve()):
            raise ValueError('Use isolated project runs/ outputs')
    if args.check_only:
        print(json.dumps(dict(status='focused_release_identity_verified_no_training', seeds=SEEDS)))
        return
    raise SystemExit(run_campaign(config, budget, args.config, args.budget, args.data_dir, args.job_dir, args.output_root))


if __name__ == '__main__':
    main()
