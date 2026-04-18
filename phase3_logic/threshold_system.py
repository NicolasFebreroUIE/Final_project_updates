"""
Phase 3.1 / 3.3 — Decision Threshold System (Professional Legal Scale)

Defines verdict thresholds and demonstrates threshold crossings caused by
linguistic form alone.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Decision thresholds (Professional Legal-Tech Scale)
THRESHOLDS = {
    "excessive_probative_deficit":    (0.00, 0.20),   # Absolute Innocence
    "insufficient_material_evidence": (0.20, 0.50),   # Release / Open Investigation
    "probable_cause":                 (0.50, 0.90),   # Further Investigation / Remand
    "high_probability_guilt":         (0.90, 0.97),   # Conviction Recommendation
    "beyond_reasonable_doubt":       (0.97, 1.00),   # Maximal Conviction
}


def get_verdict(score: float) -> str:
    """
    Map a score (0.0 to 1.0) to a verdict string based on professional thresholds.
    """
    for verdict, (low, high) in THRESHOLDS.items():
        if low <= score < high:
            return verdict
    # Edge case: score == 1.0
    if score >= 1.0:
        return "beyond_reasonable_doubt"
    return "excessive_probative_deficit"


def get_verdict_description(verdict: str) -> str:
    """Return a formal English-language description of the verdict."""
    descriptions = {
        "excessive_probative_deficit":    "EXCESSIVE PROBATIVE DEFICIT — Immediate Release Requested",
        "insufficient_material_evidence": "INSUFFICIENT MATERIAL EVIDENCE — Case remains open / Suspect released",
        "probable_cause":                 "PROBABLE CAUSE — Remand for further investigation or Indictment",
        "high_probability_guilt":         "HIGH PROBABILITY OF GUILT — Conviction Recommendation",
        "beyond_reasonable_doubt":       "BEYOND ALL REASONABLE DOUBT — Maximal Conviction",
    }
    return descriptions.get(verdict, "Unknown status code")


def get_all_thresholds() -> dict:
    """Returns thresholds for UI legend display."""
    return {
        v: {
            "range": list(r),
            "label": get_verdict_description(v)
        } for v, r in THRESHOLDS.items()
    }


def find_threshold_crossings(variant_scores: dict) -> list:
    """ Find pairs of variants that cross thresholds. """
    crossings = []
    variant_ids = sorted(variant_scores.keys())

    for i in range(len(variant_ids)):
        for j in range(i + 1, len(variant_ids)):
            vid_a = variant_ids[i]
            vid_b = variant_ids[j]
            score_a = variant_scores[vid_a]
            score_b = variant_scores[vid_b]

            verdict_a = get_verdict(score_a)
            verdict_b = get_verdict(score_b)

            if verdict_a != verdict_b:
                score_diff = abs(score_a - score_b)
                crossing = {
                    'variant_a': vid_a, 'variant_b': vid_b,
                    'score_a': score_a, 'score_b': score_b,
                    'verdict_a': verdict_a, 'verdict_b': verdict_b,
                    'score_difference': score_diff
                }
                crossings.append(crossing)
    return crossings
def print_threshold_crossings(crossings: list):
    """Print the crossings in a human-readable format."""
    print(f"  [THRESHOLD] Found {len(crossings)} crossings caused by argument variant form.")
    # Show only the top 3 most significant crossings
    sorted_crossings = sorted(crossings, key=lambda x: x['score_difference'], reverse=True)
    for c in sorted_crossings[:3]:
        print(f"    - {c['variant_a']} ({c['score_a']:.4f}, {c['verdict_a']}) vs "
              f"{c['variant_b']} ({c['score_b']:.4f}, {c['verdict_b']}) | Diff: {c['score_difference']:.4f}")


if __name__ == "__main__":
    results_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        crossings = find_threshold_crossings(results['variant_scores'])
        print(f"Found {len(crossings)} crossings.")
