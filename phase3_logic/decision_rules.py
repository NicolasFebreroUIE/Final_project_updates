"""
Phase 3.2 — Rule-Based Reasoning Layer

Implements explicit rule-based reasoning over central model scores.
Pure symbolic AI — no ML in this module.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase3_logic.threshold_system import get_verdict

# Explicit decision rules
RULES = [
    {
        "id": "R00",
        "condition": "TRUE (Initial state)",
        "effect": "score_adjustment",
        "value": 0.35,
        "legal_principle": "Initial Indictment Bias — Burden of proof integration sequence"
    },
    {
        "id": "R01",
        "condition": "forensic_value == right_handed AND suspect_laterality == left",
        "effect": "score_adjustment",
        "value": -0.20,
        "legal_principle": "Physical impossibility — exculpatory forensic evidence"
    },
    {
        "id": "R02",
        "condition": "witness_deception_indicator > 0.60",
        "effect": "score_adjustment",
        "value": -0.10,
        "legal_principle": "Unreliable testimony — reduced evidentiary weight"
    },
    {
        "id": "R03",
        "condition": "suspect_emotional_consistency > 0.70 AND suspect_dominant_emotion in [sad, fearful]",
        "effect": "score_adjustment",
        "value": -0.05,
        "legal_principle": "Emotional profile inconsistent with guilt concealment"
    },
    {
        "id": "R04",
        "condition": "argument_score == max(all_scores)",
        "effect": "flag",
        "value": "linguistic_optimization_detected",
        "legal_principle": "Score may reflect argument form, not argument substance"
    },
    {
        "id": "R05",
        "condition": "witness_reactivity > 0.60 AND witness_dominant_emotion == angry",
        "effect": "score_adjustment",
        "value": -0.15,
        "legal_principle": "Witness emotional instability — sudden aggressive reactivity reduces testimony weight"
    }
]


def apply_rules(raw_score: float,
                forensic_report: dict,
                suspect_report: dict,
                witness_report: dict,
                is_optimal_variant: bool = True,
                all_scores: dict = None) -> dict:
    """
    Apply all decision rules sequentially to a raw NLP score.

    Args:
        raw_score: The raw score from central_model.score()
        forensic_report: Physical evidence report dict
        suspect_report: Suspect video analysis report dict
        witness_report: Witness video analysis report dict
        is_optimal_variant: Whether this is the highest-scoring variant
        all_scores: Dict of all variant scores (for R04)

    Returns:
        Dict with adjusted score, applied rules, and reasoning log
    """
    # The raw_score from the NLP model represents 'Persuasion' (High = Good for defense).
    # We must invert it to represent 'Initial Guilt Probability' (High = Bad for defense).
    adjusted_score = 1.0 - raw_score
    
    applied_rules = []
    total_adjustment = 0.0
    flags = []

    print("  [RULES] Evaluating decision rules...")

    # R00: Initial Bias
    adjustment = RULES[0]['value']
    adjusted_score += adjustment
    total_adjustment += adjustment
    applied_rules.append({
        'rule_id': 'R00',
        'adjustment': adjustment,
        'principle': RULES[0]['legal_principle'],
        'fired': True
    })
    print(f"  [RULES] R00 applied (Base Bias): {adjustment:+.2f}")

    # R01: Forensic handedness incompatibility
    forensic_value = forensic_report.get('value', '')
    suspect_laterality = 'left'  # Daniel Navarro is left-handed (case fact)

    if forensic_value == 'right_handed' and suspect_laterality == 'left':
        adjustment = -0.40 # Decisive forensic mismatch
        adjusted_score += adjustment
        total_adjustment += adjustment
        applied_rules.append({
            'rule_id': 'R01',
            'adjustment': adjustment,
            'principle': "Forensic Impossibility: Perp is Right-Handed, Suspect is Left-Handed",
            'fired': True
        })
        print(f"  [RULES] R01 fired: {adjustment:+.2f} — {RULES[1]['legal_principle']}")
    else:
        applied_rules.append({'rule_id': 'R01', 'fired': False})

    # R02: Witness deception indicator
    witness_deception = witness_report.get('deception_indicator', 0)

    if witness_deception > 0.60:
        adjustment = RULES[2]['value']
        adjusted_score += adjustment
        total_adjustment += adjustment
        applied_rules.append({
            'rule_id': 'R02',
            'adjustment': adjustment,
            'principle': RULES[2]['legal_principle'],
            'fired': True,
            'condition_values': {
                'witness_deception_indicator': witness_deception
            }
        })
        print(f"  [RULES] R02 fired: {adjustment:+.2f} — {RULES[2]['legal_principle']}")
    else:
        applied_rules.append({'rule_id': 'R02', 'fired': False})

    # R03: Suspect emotional profile (Exculpatory if Sad/Fearful)
    suspect_consistency = suspect_report.get('emotional_consistency_score', 0)
    suspect_emotion = suspect_report.get('dominant_emotion', '').lower()

    if suspect_consistency > 0.40 and suspect_emotion in ['sad', 'fear', 'fearful']:
        adjustment = -0.15 # Strong reduction for stress-compatible profile
        adjusted_score += adjustment
        total_adjustment += adjustment
        applied_rules.append({
            'rule_id': 'R03',
            'adjustment': adjustment,
            'principle': "Suspect profile compatible with traumatic stress/innocence",
            'fired': True
        })
    elif suspect_emotion in ['angry', 'disgust']:
        adjustment = +0.10 # Slight increase for aggressive profile
        adjusted_score += adjustment
        total_adjustment += adjustment
        applied_rules.append({
            'rule_id': 'R03',
            'adjustment': adjustment,
            'principle': "Suspect exhibits aggressive/hostile emotional markers",
            'fired': True
        })
    else:
        applied_rules.append({'rule_id': 'R03', 'fired': False})

    # R04: Linguistic optimization flag
    if is_optimal_variant:
        flags.append({
            'flag': RULES[4]['value'],
            'principle': RULES[4]['legal_principle']
        })
        applied_rules.append({
            'rule_id': 'R04',
            'effect': 'flag',
            'flag_value': RULES[4]['value'],
            'principle': RULES[4]['legal_principle'],
            'fired': True
        })
        print(f"  [RULES] R04 fired: FLAG — {RULES[4]['legal_principle']}")
    else:
        applied_rules.append({'rule_id': 'R04', 'fired': False})

    # R05: Witness reactivity
    witness_reactivity = witness_report.get('reactivity_index', 0)
    witness_emotion = witness_report.get('dominant_emotion', '')

    if witness_reactivity > 0.60 and witness_emotion == 'angry':
        adjustment = RULES[5]['value']
        adjusted_score += adjustment
        total_adjustment += adjustment
        applied_rules.append({
            'rule_id': 'R05',
            'adjustment': adjustment,
            'principle': RULES[5]['legal_principle'],
            'fired': True,
            'condition_values': {
                'witness_reactivity': witness_reactivity,
                'witness_dominant_emotion': witness_emotion
            }
        })
        print(f"  [RULES] R05 fired: {adjustment:+.2f} — {RULES[5]['legal_principle']}")
    else:
        applied_rules.append({'rule_id': 'R05', 'fired': False})

    # Clamp score to [0, 1]
    adjusted_score = max(0.0, min(1.0, adjusted_score))

    # Get verdict
    verdict = get_verdict(adjusted_score)

    result = {
        'raw_score': raw_score,
        'adjusted_score': adjusted_score,
        'total_adjustment': total_adjustment,
        'verdict': verdict,
        'applied_rules': applied_rules,
        'flags': flags,
        'rules_fired': [r['rule_id'] for r in applied_rules if r.get('fired', False)]
    }

    print(f"  [RULES] Raw score: {raw_score:.4f} >> Adjusted: {adjusted_score:.4f}")
    print(f"  [RULES] Total adjustment: {total_adjustment:+.4f}")
    print(f"  [RULES] Final verdict: {verdict.upper()}")

    return result


if __name__ == "__main__":
    # Test with sample data
    forensic = {"value": "right_handed", "confidence": 0.91}
    suspect = {"emotional_consistency_score": 0.78, "dominant_emotion": "sad"}
    witness = {"deception_indicator": 0.65}

    # Adjusting for rule index shifts
    result = apply_rules(0.50, forensic, suspect, witness)
    print(json.dumps(result, indent=2))
