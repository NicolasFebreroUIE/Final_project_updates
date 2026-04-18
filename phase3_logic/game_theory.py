"""
Phase 3.4 — Game Theory Analysis

Models the lawyer as a rational agent selecting among 15 argument variants.
Demonstrates that linguistic optimization is always rational in a system
designed to be impartial.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase3_logic.threshold_system import get_verdict


def game_theory_analysis(phase1_results: dict) -> dict:
    """
    Game theory framing of the argument variant selection problem.

    Models:
    - Player: Defense lawyer (rational agent)
    - Strategy space: {variant_01, ..., variant_15}
    - Payoff function: central_model.score(variant)
    - Dominant strategy: variant with highest score

    Args:
        phase1_results: Dict containing variant scores from Phase 1

    Returns:
        Game theory analysis results dict
    """
    variant_scores = phase1_results['variant_scores']
    variant_ids = sorted(variant_scores.keys())

    print("  [GAME] Game Theory Analysis — Rational Agent Framework")
    print("  [GAME] Player: Defense lawyer")
    print(f"  [GAME] Strategy space: {len(variant_ids)} argument variants")
    print("  [GAME] Payoff function: central_model.score(variant)")
    print()

    # Find dominant strategy
    dominant_id = max(variant_scores, key=variant_scores.get)
    dominant_score = variant_scores[dominant_id]
    dominant_verdict = get_verdict(dominant_score)

    # Build strategy table
    strategy_table = []
    print("  ┌────────────────┬──────────┬──────────────────────────┬────────────────────┐")
    print("  │ Variant ID     │ Score    │ Verdict                  │ Dominant Strategy? │")
    print("  ├────────────────┼──────────┼──────────────────────────┼────────────────────┤")

    for vid in variant_ids:
        score = variant_scores[vid]
        verdict = get_verdict(score)
        is_dominant = vid == dominant_id

        strategy_table.append({
            'variant_id': vid,
            'score': score,
            'verdict': verdict,
            'is_dominant_strategy': is_dominant
        })

        marker = "  ✓ YES" if is_dominant else "    NO"
        print(f"  │ {vid:14s} │ {score:.4f}  │ {verdict:24s} │ {marker:18s} │")

    print("  └────────────────┴──────────┴──────────────────────────┴────────────────────┘")

    # Analysis conclusions
    print()
    print("  [GAME] CONCLUSION:")
    print(f"  [GAME] Dominant strategy: {dominant_id} (score: {dominant_score:.4f})")
    print(f"  [GAME] Dominant strategy verdict: {dominant_verdict.upper()}")
    print()
    print("  [GAME] In a system designed to be impartial, linguistic optimization is")
    print("  [GAME] always rational and always advantageous. The rational defense attorney")
    print("  [GAME] will always select the argument variant that maximizes the model's score,")
    print("  [GAME] regardless of semantic equivalence. This represents a Nash equilibrium")
    print("  [GAME] in which the dominant strategy is determined entirely by the model's")
    print("  [GAME] sensitivity to linguistic form.")

    # Compute advantage metrics
    scores_sorted = sorted(variant_scores.values())
    advantage = dominant_score - scores_sorted[0]  # vs worst strategy
    median_advantage = dominant_score - scores_sorted[len(scores_sorted) // 2]

    results = {
        'game_type': 'Single-player optimization against fixed evaluator',
        'player': 'Defense lawyer (rational agent)',
        'strategy_space_size': len(variant_ids),
        'payoff_function': 'central_model.score(argument_variant)',
        'dominant_strategy': {
            'variant_id': dominant_id,
            'score': dominant_score,
            'verdict': dominant_verdict,
        },
        'strategy_table': strategy_table,
        'advantage_vs_worst': advantage,
        'advantage_vs_median': median_advantage,
        'conclusion': (
            "In a system designed to be impartial, linguistic optimization is always rational "
            "and always advantageous. The rational defense attorney will always select the argument "
            "variant that maximizes the model's score, regardless of semantic equivalence."
        )
    }

    return results


if __name__ == "__main__":
    results_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        phase1 = json.load(f)

    results = game_theory_analysis(phase1)
