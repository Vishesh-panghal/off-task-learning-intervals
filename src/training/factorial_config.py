"""Consumed configuration and explicit P2BR policies, without torch imports."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Arm:
    name: str
    temporary_head: bool
    reset_backbone: bool
    middle_source: str = 'target'
    middle_classes: int = 100


ARMS = {
    'C0': Arm('C0', False, False),
    'C1': Arm('C1', False, True),
    'C2': Arm('C2', True, False),
    'C3': Arm('C3', True, True),
    'S1': Arm('S1', True, True, 'stl10', 10),
}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def stream_seed(seed, namespace, index=0):
    """Process-independent uint32 seed; never Python's randomized hash()."""
    return int.from_bytes(hashlib.sha256(f'P2BR-v1:{seed}:{namespace}:{index}'.encode()).digest()[:4],'little')


def validate_config(c):
    keys={'schema_version','study_id','protocol_path','stage','arms','schedule','optimizer','data','diagnostics','pilot_seeds','confirmatory_candidate_seeds','confirmatory_seeds','release'}
    if set(c)!=keys or c['schema_version']!=1: raise ValueError('Unknown/incomplete configuration schema')
    if c['stage'] not in ['preliminary','pilot','locked']: raise ValueError('Unknown study stage')
    if c['arms']!=list(ARMS): raise ValueError('Primary study requires C0/C1/C2/C3/S1 in the declared order')
    s=c['schedule']
    if set(s)!={'initial','middle','final','target_comparator'} or any(type(v) is not int or v<=0 for v in s.values()):
        raise ValueError('Invalid integer update schedule')
    total=s['initial']+s['middle']+s['final']
    if s['target_comparator']!=s['initial']+s['final'] or not s['initial']+s['middle']<s['target_comparator']<total:
        raise ValueError('Comparator must be the 30k equivalent within final phase')
    if set(c['optimizer'])!={'lr','momentum','weight_decay'}: raise ValueError('Unknown optimizer field')
    o=c['optimizer']
    if not o['lr']>0 or not 0<o['momentum']<1 or o['weight_decay']<0: raise ValueError('Invalid SGD values')
    if set(c['data'])!={'split_seed','validation_size','batch_size','geometry_count','geometry_seed'}: raise ValueError('Unknown data field')
    if set(c['diagnostics'])!={'validation_interval','probe_epochs','probe_lr','probe_weight_decay'}: raise ValueError('Unknown diagnostic field')
    if any(type(c['data'][k]) is not int or c['data'][k]<=0 for k in c['data']): raise ValueError('Data settings must be positive integers')
    if c['data']['geometry_count']>c['data']['validation_size']: raise ValueError('Geometry inputs must be within validation')
    d=c['diagnostics']
    if type(d['validation_interval']) is not int or d['validation_interval']<1 or type(d['probe_epochs']) is not int or d['probe_epochs']<1 or d['probe_lr']<=0 or d['probe_weight_decay']<0:
        raise ValueError('Invalid diagnostic settings')
    for key in ['pilot_seeds','confirmatory_candidate_seeds','confirmatory_seeds']:
        seeds=c[key]
        if len(seeds)!=len(set(seeds)) or any(type(s) is not int or not 0<=s<2**32 for s in seeds): raise ValueError(f'Invalid seeds in {key}')
    if set(c['pilot_seeds']) & (set(c['confirmatory_candidate_seeds'])|set(c['confirmatory_seeds'])):
        raise ValueError('Pilot and confirmatory seeds must be disjoint')
    expected_release={'g0_dispositions_accepted','g0_evidence_path','g0_evidence_sha256','pilot_enabled','confirmatory_enabled','final_n','locked_protocol_sha256','lock_evidence_path','lock_evidence_sha256'}
    if set(c['release'])!=expected_release: raise ValueError('Unknown/incomplete release fields')
    if any(type(c['release'][k]) is not bool for k in ['g0_dispositions_accepted','pilot_enabled','confirmatory_enabled']):
        raise ValueError('Release switches must be JSON booleans')
    return c


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def exposure_counts(c, arm):
    a=ARMS[arm]; s=c['schedule']; total=sum(s[k] for k in ['initial','middle','final'])
    return dict(total_updates=total,backbone_target_updates=total-(s['middle'] if a.middle_source!='target' else 0),
                backbone_auxiliary_updates=s['middle'] if a.middle_source!='target' else 0,
                original_head_updates=s['initial']+s['final'] if a.temporary_head else total,
                temporary_head_updates=s['middle'] if a.temporary_head else 0,
                nominal_presentations=total*c['data']['batch_size'])


def require_scientific_release(c, root, mode, seeds):
    """The existing recovery plan's G0/locking gate; never needed for tests."""
    r=c['release']; root=Path(root)
    if mode not in ['pilot','confirmatory']: raise ValueError('Unknown scientific execution stage')
    if not r['g0_dispositions_accepted'] or not r[f'{mode}_enabled']:
        raise ValueError('Scientific execution unreleased: G0 dispositions and stage release are required. Describe/synthetic checks remain available.')
    def verified_file(name,expected):
        if not name or not expected or not (root/name).is_file() or file_hash(root/name)!=expected:
            raise ValueError('Missing or changed release evidence')
    verified_file(r['g0_evidence_path'],r['g0_evidence_sha256'])
    if mode=='pilot':
        if not set(seeds)<=set(c['pilot_seeds']): raise ValueError('Only declared engineering pilot seeds allowed')
    else:
        if c['stage']!='locked' or type(r['final_n']) is not int or r['final_n']<2 or len(c['confirmatory_seeds'])!=r['final_n']:
            raise ValueError('Final N and complete confirmatory seed list are not locked')
        verified_file(c['protocol_path'],r['locked_protocol_sha256'])
        verified_file(r['lock_evidence_path'],r['lock_evidence_sha256'])
        if not set(seeds)<=set(c['confirmatory_seeds']): raise ValueError('Seed is outside locked confirmatory list')
    if not seeds or len(seeds)!=len(set(seeds)): raise ValueError('Choose unique declared seeds')
    if c['schedule']!={'initial':15000,'middle':5000,'final':15000,'target_comparator':30000}:
        raise ValueError('Scientific P2BR-v1 requires 15k/5k/15k with a 30k comparator; use a new protocol for another schedule')


def factorial_contrasts(values):
    """One seed's contrasts; four arm outcomes are not independent N."""
    if set(values)!=set(ARMS): raise ValueError('All five arm outcomes required')
    c0,c1,c2,c3,s1=(float(values[k]) for k in ARMS)
    return {'H1_equal_total':s1-c0,'H3a_backbone_reset':((c1-c0)+(c3-c2))/2,
            'H3b_temporary_head':((c2-c0)+(c3-c1))/2,'H3c_interaction':(c3-c2)-(c1-c0),
            'secondary_S1_minus_C3':s1-c3,'conditional_reset_H0':c1-c0,'conditional_reset_H1':c3-c2}
