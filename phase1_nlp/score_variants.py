"""
Phase 1.2 — Score all 15 argument variants

Loads lawyer_variants.md, parses all 15 variants, scores each through the
central model, computes statistics, cosine similarity matrix, and identifies
formal sensitivity pairs.
"""

import os
import sys
import re
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics.pairwise import cosine_similarity


def parse_variants(filepath: str) -> dict:
    """Parse lawyer_variants.md and extract each variant as individual text strings."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    variants = {}
    # Split by ## VARIANT XX headers
    pattern = r'## VARIANT (\d+)\s*\n\s*\n(.*?)(?=\n---|\n## EXPERIMENTAL|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        variant_id = int(match[0])
        variant_text = match[1].strip()
        if variant_text:
            variants[f"variant_{variant_id:02d}"] = variant_text

    return variants


def score_all_variants(central_model, variants_path: str, output_path: str) -> dict:
    """
    Score all 15 variants and compute statistics.

    Returns dict with all results for downstream use.
    """
    # Parse variants
    variants = parse_variants(variants_path)
    print(f"  [SCORE] Parsed {len(variants)} variants from {variants_path}")

    if len(variants) == 0:
        print("  [SCORE] ERROR: No variants parsed. Check file format.")
        return {}

    # Score each variant using batch scoring for Zero-Bias Balancing
    variant_ids = sorted(variants.keys())
    texts = [variants[vid] for vid in variant_ids]
    batch_scores = central_model.score_batch(texts)
    
    scores = {}
    embeddings = {}
    for i, vid in enumerate(variant_ids):
        score = batch_scores[i]
        embedding = central_model.embed(variants[vid])
        scores[vid] = score
        embeddings[vid] = embedding
        print(f"  [SCORE] {vid}: {score:.6f}")

    # Compute statistics
    score_values = [scores[vid] for vid in variant_ids]
    mean_score = float(np.mean(score_values))
    std_score = float(np.std(score_values))
    min_score = float(np.min(score_values))
    max_score = float(np.max(score_values))
    score_range = max_score - min_score

    # Compute pairwise cosine similarity matrix (15x15)
    embedding_matrix = np.array([embeddings[vid] for vid in variant_ids])
    cosine_sim_matrix = cosine_similarity(embedding_matrix)

    # Identify pairs with cosine_sim > 0.95 AND score_diff > 0.01
    sensitivity_pairs = []
    for i in range(len(variant_ids)):
        for j in range(i + 1, len(variant_ids)):
            cos_sim = cosine_sim_matrix[i][j]
            score_diff = abs(scores[variant_ids[i]] - scores[variant_ids[j]])
            if cos_sim > 0.95 and score_diff > 0.01:
                sensitivity_pairs.append({
                    'variant_a': variant_ids[i],
                    'variant_b': variant_ids[j],
                    'cosine_similarity': float(cos_sim),
                    'score_a': scores[variant_ids[i]],
                    'score_b': scores[variant_ids[j]],
                    'score_difference': float(score_diff),
                })

    # Identify optimal variant (highest score)
    optimal_id = max(scores, key=scores.get)
    optimal_score = scores[optimal_id]
    optimal_text = variants[optimal_id]

    # Identify worst variant (lowest score)
    worst_id = min(scores, key=scores.get)
    worst_score = scores[worst_id]
    worst_text = variants[worst_id]

    # Check for threshold crossings
    from phase3_logic.threshold_system import get_verdict, THRESHOLDS
    verdicts = {vid: get_verdict(scores[vid]) for vid in variant_ids}
    unique_verdicts = set(verdicts.values())
    threshold_crossings = len(unique_verdicts) > 1

    # Build complete results
    results = {
        'variant_scores': {vid: scores[vid] for vid in variant_ids},
        'variant_texts': {vid: variants[vid] for vid in variant_ids},
        'statistics': {
            'mean': mean_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'range': score_range,
        },
        'cosine_similarity_matrix': cosine_sim_matrix.tolist(),
        'variant_ids': variant_ids,
        'sensitivity_pairs': sensitivity_pairs,
        'optimal_variant': {
            'id': optimal_id,
            'score': optimal_score,
            'text': optimal_text,
        },
        'worst_variant': {
            'id': worst_id,
            'score': worst_score,
            'text': worst_text,
        },
        'verdicts': verdicts,
        'threshold_crossings_detected': threshold_crossings,
        'embeddings': {vid: embeddings[vid].tolist() for vid in variant_ids},
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save without embeddings (large) to a separate file
    results_save = {k: v for k, v in results.items() if k != 'embeddings'}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_save, f, indent=2, ensure_ascii=False)
    print(f"  [SCORE] Results saved to {output_path}")

    # Save embeddings separately for XAI
    embeddings_path = output_path.replace('.json', '_embeddings.json')
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        json.dump({'embeddings': results['embeddings'], 'variant_ids': variant_ids}, f)

    # Print console summary
    print(f"\n[PHASE 1 RESULTS]")
    print(f"Score range: {min_score:.4f} – {max_score:.4f} (range: {score_range:.4f})")
    print(f"Mean: {mean_score:.4f} | Std: {std_score:.4f}")
    print(f"Optimal variant: {optimal_id} (score: {optimal_score:.4f})")
    print(f"Pairs with cosine_sim > 0.95 and score_diff > 0.01: {len(sensitivity_pairs)} pairs found")
    print(f"Threshold crossings detected: {'YES' if threshold_crossings else 'NO'}")

    return results


if __name__ == "__main__":
    from models.central_model import CentralModel

    model = CentralModel()
    model.load(os.path.join(PROJECT_ROOT, "models", "central_model.pkl"))

    variants_path = os.path.join(PROJECT_ROOT, "data", "arguments", "lawyer_variants.md")
    output_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")

    results = score_all_variants(model, variants_path, output_path)
