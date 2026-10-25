"""Verify delivered P2BR engineering artifacts and unchanged audit-baseline files."""
from pathlib import Path
import csv
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.training.factorial_config import ARMS,digest_json,file_hash,read_config,require_scientific_release
from scripts.run_factorial import source_manifest,synthetic_config


def main():
    config=read_config(ROOT/'configs/factorial_p2br_preliminary.json'); generated=synthetic_config(config)
    folder=ROOT/'runs/p2br/synthetic-check'/generated['study_id']/'seed_190001'
    changed=[]; count=0
    with (ROOT/'audit/evidence_manifest.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            count+=1; path=ROOT/row['path']
            if not path.is_file() or file_hash(path)!=row['sha256']: changed.append(row['path'])
    xml=ET.parse(ROOT/'protocol/checks/pytest.xml').getroot()
    suites=list(xml.iter('testsuite'))
    totals={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    seed_audit=json.loads((ROOT/'protocol/local_seed_audit.json').read_text())
    identity=json.loads((folder/'identity.json').read_text())
    complete=json.loads((folder/'seed_complete.json').read_text())
    outcomes=[json.loads((folder/a/'outcome.json').read_text()) for a in ARMS]
    blocked={}
    for mode in ['pilot','confirmatory']:
        try:
            require_scientific_release(config,ROOT,mode,config['pilot_seeds'] if mode=='pilot' else config['confirmatory_candidate_seeds'])
            blocked[mode]=False
        except ValueError: blocked[mode]=True
    checks=dict(original_baseline_files_unchanged=not changed,
                pytest_passed=totals['tests']>=26 and all(totals[k]==0 for k in ['failures','errors','skipped']),
                synthetic_five_arms_complete=complete['arms']==list(ARMS) and len(outcomes)==5,
                synthetic_evidence_excluded_from_scientific_outcomes=all(r['synthetic'] and not r['scientific_result'] and r['primary_test_accuracy'] is None for r in outcomes),
                synthetic_consumed_configuration_matches=identity['config_sha256']==digest_json(generated),
                synthetic_source_identity_matches=identity['source_sha256']==source_manifest(),
                synthetic_protocol_identity_matches=identity['protocol_sha256']==file_hash(ROOT/config['protocol_path']),
                shared_prefix_verified=len({r['shared_prefix_hash'] for r in outcomes})==1,
                outcome_checksums_match=all(complete['outcome_sha256'][a]==file_hash(folder/a/'outcome.json') for a in ARMS),
                no_known_local_pilot_or_candidate_seed_reuse=not seed_audit['overlap'],
                scientific_pilot_and_confirmation_still_blocked=all(blocked.values()),
                final_sample_size_not_falsely_locked=config['release']['final_n'] is None and not config['confirmatory_seeds'])
    files=list(source_manifest())+['configs/factorial_p2br_preliminary.json','protocol/P2BR_PRELIMINARY_v0.1.md',
                                'protocol/P2BR_RUNBOOK.md','protocol/IMPLEMENTATION_STATUS.md','tests/test_factorial.py',
                                'scripts/check_factorial_seeds.py','scripts/verify_factorial_implementation.py']
    report=dict(passed=all(checks.values()),checks=checks,baseline_files_checked=count,changed_baseline_paths=changed,pytest=totals,
                implementation_sha256={p:file_hash(ROOT/p) for p in files},synthetic_check_directory=str(folder.relative_to(ROOT)),
                scope='CPU engineering validation on generated inputs; no official-test evaluation, pilot research execution, GPU validation, final protocol lock or scientific efficacy claim.',
                g0_status='open for unchanged manuscript; audit dispositions await reporting/claim resolution',
                next_stage='Resolve G0 dispositions and release engineering pilot; determine final N and freeze final protocol before confirmatory execution.')
    path=ROOT/'protocol/checks/implementation_verification.json'; path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'},indent=2))
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
