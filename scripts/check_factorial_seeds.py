"""Screen proposed new seeds against available historical local records only."""
from pathlib import Path
import argparse
import csv
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.factorial_config import file_hash,read_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/factorial_p2br_preliminary.json')
    parser.add_argument('--output',type=Path,default=ROOT/'protocol/local_seed_audit.json')
    parser.add_argument('--history-root',type=Path,action='append',default=[],help='Also inspect historical results/checkpoints/runs in this root, e.g. the original GPU project beside an isolated deployment')
    args=parser.parse_args(); config=read_config(args.config)
    used=set(); sources=[]
    def json_seeds(value):
        if isinstance(value,dict):
            for key,v in value.items():
                if key=='seed' and type(v) is int: used.add(v)
                elif key in ['seeds','confirmatory_seeds'] and isinstance(v,list): used.update(s for s in v if type(s) is int)
                json_seeds(v)
        elif isinstance(value,list):
            for v in value: json_seeds(v)
    roots=list(dict.fromkeys([ROOT]+[p.resolve() for p in args.history_root]))
    for history_root in roots:
      if not history_root.is_dir(): raise FileNotFoundError(f'History root is missing: {history_root}')
      for folder in ['results','checkpoints','supplementary/results','supplementary/protocol/archive/later_screening','runs']:
        for path in sorted((history_root/folder).rglob('*')):
            if not path.is_file(): continue
            used.update(int(x) for x in re.findall(r'seed_?(\d+)',str(path.relative_to(history_root/folder)),re.I))
            if path.suffix=='.csv':
                with path.open(newline='') as f:
                    for row in csv.DictReader(f):
                        try: used.add(int(row['seed']))
                        except (KeyError,ValueError): pass
            elif path.suffix=='.json': json_seeds(json.loads(path.read_text()))
            elif path.suffix=='.log':
                text=path.read_text(errors='replace')
                used.update(int(x) for x in re.findall(r'\bseed[ _:=]+(\d+)',text,re.I))
            if path.suffix in ['.csv','.json','.log']:
                sources.append({'history_root':str(history_root),'path':str(path.relative_to(history_root)),'sha256':file_hash(path)})
    proposed=config['pilot_seeds']+config['confirmatory_candidate_seeds']+config['confirmatory_seeds']
    overlap=sorted(set(proposed)&used)
    report=dict(status='no known local reuse' if not overlap else 'collision found',historical_seeds_found=sorted(used),
                pilot_seeds=config['pilot_seeds'],confirmatory_candidates=config['confirmatory_candidate_seeds'],overlap=overlap,
                input_files=sources,history_roots=[str(p) for p in roots],filename_scope='results/checkpoints, supplied historical supplementary record trees, and existing new runs',
                limit='Does not establish freshness on remote hosts or in missing records. Candidate confirmatory list and final N remain unlocked.')
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_files'},indent=2))
    if overlap: raise SystemExit(1)


if __name__=='__main__': main()
