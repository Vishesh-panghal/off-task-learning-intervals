"""Explicit three-arm, six-seed follow-up; never a partial five-arm factorial."""
from copy import deepcopy
import json
from pathlib import Path

from src.training.factorial_config import (
    ARMS as FACTORIAL_ARMS, digest_json, exposure_counts, file_hash, stream_seed,
    validate_config as validate_factorial_config,
    require_scientific_release as require_base_release,
)

ARMS = {name: FACTORIAL_ARMS[name] for name in ('C0', 'C3', 'S1')}
SEEDS = list(range(920001, 920007))
STUDY_ID = 'SOTLI-P2BR-FOCUSED-13H-20261017-v1'


def validate_config(config):
    if config.get('schema_version') != 2 or config.get('study_id') != STUDY_ID:
        raise ValueError('Explicit focused schema/study identity required')
    if config.get('arms') != list(ARMS):
        raise ValueError('Focused study requires exactly C0/C3/S1 in order')
    # Reuse validation of common fields, explicitly projecting only for that
    # validation. The actual consumed configuration remains schema 2 / 3 arms.
    common = deepcopy(config)
    common.update(schema_version=1, arms=list(FACTORIAL_ARMS))
    validate_factorial_config(common)
    if config['pilot_seeds'] or config['release']['pilot_enabled']:
        raise ValueError('Engineering pilots cannot enter the focused study')
    if config['confirmatory_candidate_seeds'] != SEEDS:
        raise ValueError('Six feasibility-limited candidate seeds required')
    if config['confirmatory_seeds'] not in ([], SEEDS):
        raise ValueError('No seed substitution or variable sample size')
    if config['release']['confirmatory_enabled']:
        if config['stage'] != 'locked' or config['confirmatory_seeds'] != SEEDS or config['release']['final_n'] != 6:
            raise ValueError('Enabled follow-up requires locked six paired seeds')
    return config


def read_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def require_scientific_release(config, root, mode, seeds):
    validate_config(config)
    if mode != 'confirmatory':
        raise ValueError('Focused runner has no engineering-pilot release')
    require_base_release(config, root, mode, seeds)


def focused_contrasts(values):
    if set(values) != set(ARMS):
        raise ValueError('Exactly three complete focused arm outcomes required')
    c0, c3, s1 = (float(values[name]) for name in ARMS)
    return {
        'H1_equal_total': s1 - c0,
        'secondary_S1_minus_C3': s1 - c3,
        'secondary_C3_minus_C0': c3 - c0,
    }
