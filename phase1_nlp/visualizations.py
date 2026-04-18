"""
Phase 1.4 — Visualizations

Generates:
1. Bar chart: 15 variant scores with decision thresholds
2. Heatmap: 15x15 cosine similarity matrix

Both annotated for academic presentation.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def generate_visualizations(phase1_results: dict, output_dir: str):
    """Generate all Phase 1 visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    variant_ids = phase1_results['variant_ids']
    scores = phase1_results['variant_scores']
    cosine_matrix = np.array(phase1_results['cosine_similarity_matrix'])
    optimal_id = phase1_results['optimal_variant']['id']

    # ========== 1. Bar Chart: Variant Scores ==========
    print("  [VIZ] Generating scores bar chart...")

    fig, ax = plt.subplots(figsize=(16, 8))

    score_values = [scores[vid] for vid in variant_ids]
    x_positions = range(len(variant_ids))

    # Color bars: highlight optimal variant
    colors = []
    for vid in variant_ids:
        if vid == optimal_id:
            colors.append('#4CAF50')  # Green for optimal
        else:
            colors.append('#2196F3')  # Blue for others

    bars = ax.bar(x_positions, score_values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add score labels on top of bars
    for bar, score in zip(bars, score_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.003,
                f'{score:.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Draw threshold lines
    thresholds = {
        'Insufficient Evidence': 0.40,
        'Inconclusive': 0.60,
        'Probable Guilt': 0.75,
    }
    threshold_colors = {
        'Insufficient Evidence': '#F44336',
        'Inconclusive': '#FF9800',
        'Probable Guilt': '#9C27B0',
    }

    for label, threshold in thresholds.items():
        ax.axhline(y=threshold, color=threshold_colors[label], linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'{label} ({threshold:.2f})')

    # Format
    ax.set_xticks(x_positions)
    ax.set_xticklabels([vid.replace('variant_', 'V') for vid in variant_ids],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Central Model Score', fontsize=12)
    ax.set_xlabel('Argument Variant', fontsize=12)
    ax.set_title('Argument Scores Across 15 Semantically Identical Variants\n'
                 'State v. Daniel Navarro — Algorithmic Impartiality Analysis',
                 fontsize=14, fontweight='bold')

    # Add statistics annotation
    stats = phase1_results['statistics']
    stats_text = (f"Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f}\n"
                  f"Range: {stats['min']:.4f} – {stats['max']:.4f} "
                  f"(Δ = {stats['range']:.4f})")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
            facecolor='lightyellow', alpha=0.8))

    # Legend
    optimal_patch = mpatches.Patch(color='#4CAF50', label=f'Optimal: {optimal_id}')
    other_patch = mpatches.Patch(color='#2196F3', label='Other variants')
    ax.legend(handles=[optimal_patch, other_patch] +
              [plt.Line2D([0], [0], color=threshold_colors[l], linestyle='--',
                          label=f'{l} ({t:.2f})') for l, t in thresholds.items()],
              loc='upper right', fontsize=8, framealpha=0.9)

    ax.set_ylim(0, max(score_values) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    scores_path = os.path.join(output_dir, "scores_chart.png")
    plt.savefig(scores_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [VIZ] Scores chart saved to {scores_path}")

    # ========== 2. Heatmap: Cosine Similarity Matrix ==========
    print("  [VIZ] Generating cosine similarity heatmap...")

    fig, ax = plt.subplots(figsize=(14, 12))

    labels = [vid.replace('variant_', 'V') for vid in variant_ids]

    # Create mask for upper triangle (optional — show full for clarity)
    mask = np.zeros_like(cosine_matrix, dtype=bool)

    # Custom colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Find min/max for better contrast
    off_diag = cosine_matrix[~np.eye(cosine_matrix.shape[0], dtype=bool)]
    vmin = max(0.85, off_diag.min() - 0.01)
    vmax = 1.0

    sns.heatmap(cosine_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                ax=ax, square=True, linewidths=0.5,
                vmin=vmin, vmax=vmax,
                annot_kws={'size': 7},
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8})

    ax.set_title('Pairwise Cosine Similarity — 15 Argument Variants\n'
                 'Embedding Space Proximity (all-MiniLM-L6-v2)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Variant', fontsize=11)
    ax.set_ylabel('Variant', fontsize=11)

    # Add annotation about high-similarity pairs with score differences
    n_pairs = len(phase1_results.get('sensitivity_pairs', []))
    annotation = (f"Pairs with cosine_sim > 0.95 AND score_diff > 0.01: {n_pairs}\n"
                  f"High semantic similarity + measurable score differences\n"
                  f"= empirical evidence of formal sensitivity")
    ax.text(0.5, -0.1, annotation, transform=ax.transAxes, fontsize=9,
            ha='center', va='top', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "cosine_heatmap.png")
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [VIZ] Cosine heatmap saved to {heatmap_path}")

    print("  [VIZ] All Phase 1 visualizations generated.")


if __name__ == "__main__":
    results_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    generate_visualizations(results, output_dir)
