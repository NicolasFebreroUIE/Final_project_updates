"""
Phase 1.3 — XAI Analysis (SHAP + LIME)

Applies SHAP to the MLP scoring head for all 15 variants.
Applies LIME for local explanations on the optimal and worst variants.
Identifies which embedding dimensions most influence score differences.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_xai_analysis(central_model, phase1_results: dict, output_dir: str):
    """
    Run SHAP and LIME analysis on the central model's scoring head.

    Args:
        central_model: The trained CentralModel instance
        phase1_results: Results dict from score_variants
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    variant_ids = phase1_results['variant_ids']
    variant_texts = phase1_results['variant_texts']

    # Get embeddings for all variants
    print("  [XAI] Computing embeddings for all variants...")
    embeddings = []
    for vid in variant_ids:
        emb = central_model.embed(variant_texts[vid])
        embeddings.append(emb)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Get the model's predict function for SHAP
    predict_fn = central_model.get_scoring_head_callable()

    # ========== SHAP Analysis ==========
    print("  [XAI] Running SHAP analysis on MLP scoring head...")

    try:
        # Use KernelExplainer with a background summary
        # Use mean embedding as background
        background = embeddings.mean(axis=0, keepdims=True)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(embeddings, nsamples=100)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # SHAP Summary Plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create a bar plot showing mean absolute SHAP values per dimension
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_k = 30  # Show top 30 most important dimensions
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]

        ax.barh(range(top_k), mean_abs_shap[top_indices], color='#2196F3', alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f'Dim {i}' for i in top_indices], fontsize=8)
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title('SHAP Feature Importance — Top 30 Embedding Dimensions\n'
                      'Central Model MLP Scoring Head', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        shap_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [XAI] SHAP summary saved to {shap_path}")

        # Identify most influential dimensions for score difference
        optimal_id = phase1_results['optimal_variant']['id']
        worst_id = phase1_results['worst_variant']['id']
        opt_idx = variant_ids.index(optimal_id)
        worst_idx = variant_ids.index(worst_id)

        shap_diff = shap_values[opt_idx] - shap_values[worst_idx]
        top_diff_dims = np.argsort(np.abs(shap_diff))[-10:][::-1]

        print(f"  [XAI] Top 10 dimensions driving score difference (best vs worst):")
        for dim in top_diff_dims:
            print(f"    Dim {dim}: SHAP diff = {shap_diff[dim]:.6f}")

    except Exception as e:
        print(f"  [XAI] WARNING: SHAP analysis encountered an error: {e}")
        print("  [XAI] Generating fallback SHAP-like analysis...")
        shap_values = None
        _generate_fallback_shap(central_model, embeddings, variant_ids, output_dir)

    # ========== LIME Analysis ==========
    print("  [XAI] Running LIME analysis...")

    try:
        from lime.lime_tabular import LimeTabularExplainer

        feature_names = [f'dim_{i}' for i in range(384)]

        lime_explainer = LimeTabularExplainer(
            embeddings,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )

        # Explain optimal variant
        optimal_id = phase1_results['optimal_variant']['id']
        opt_idx = variant_ids.index(optimal_id)
        print(f"  [XAI] LIME explanation for optimal variant ({optimal_id})...")
        opt_exp = lime_explainer.explain_instance(
            embeddings[opt_idx],
            predict_fn,
            num_features=20,
            num_samples=500
        )

        # Explain worst variant
        worst_id = phase1_results['worst_variant']['id']
        worst_idx = variant_ids.index(worst_id)
        print(f"  [XAI] LIME explanation for worst variant ({worst_id})...")
        worst_exp = lime_explainer.explain_instance(
            embeddings[worst_idx],
            predict_fn,
            num_features=20,
            num_samples=500
        )

        # Create LIME comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Optimal variant LIME
        opt_features = opt_exp.as_list()
        opt_labels = [f[0] for f in opt_features[:15]]
        opt_weights = [f[1] for f in opt_features[:15]]
        colors_opt = ['#4CAF50' if w > 0 else '#F44336' for w in opt_weights]
        axes[0].barh(range(len(opt_labels)), opt_weights, color=colors_opt, alpha=0.8)
        axes[0].set_yticks(range(len(opt_labels)))
        axes[0].set_yticklabels(opt_labels, fontsize=8)
        axes[0].set_title(f'LIME — Optimal Variant ({optimal_id})\nScore: {phase1_results["optimal_variant"]["score"]:.4f}',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Feature Weight')
        axes[0].invert_yaxis()

        # Worst variant LIME
        worst_features = worst_exp.as_list()
        worst_labels = [f[0] for f in worst_features[:15]]
        worst_weights = [f[1] for f in worst_features[:15]]
        colors_worst = ['#4CAF50' if w > 0 else '#F44336' for w in worst_weights]
        axes[1].barh(range(len(worst_labels)), worst_weights, color=colors_worst, alpha=0.8)
        axes[1].set_yticks(range(len(worst_labels)))
        axes[1].set_yticklabels(worst_labels, fontsize=8)
        axes[1].set_title(f'LIME — Worst Variant ({worst_id})\nScore: {phase1_results["worst_variant"]["score"]:.4f}',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Feature Weight')
        axes[1].invert_yaxis()

        plt.suptitle('LIME Local Explanations — Best vs Worst Scoring Variants',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        lime_path = os.path.join(output_dir, "lime_comparison.png")
        plt.savefig(lime_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [XAI] LIME comparison saved to {lime_path}")

    except Exception as e:
        print(f"  [XAI] WARNING: LIME analysis encountered an error: {e}")

    print("  [XAI] XAI analysis complete.")


def _generate_fallback_shap(central_model, embeddings, variant_ids, output_dir):
    """Generate a gradient-based feature importance analysis as SHAP fallback."""
    import torch

    print("  [XAI] Computing gradient-based feature importance...")

    central_model.scoring_head.eval()
    importances = []

    for i, vid in enumerate(variant_ids):
        emb_tensor = torch.tensor(embeddings[i:i+1], dtype=torch.float32, requires_grad=True)
        emb_tensor = emb_tensor.to(central_model.device)

        output = central_model.scoring_head(emb_tensor)
        output.backward()

        grad = emb_tensor.grad.cpu().numpy().flatten()
        importances.append(np.abs(grad))

    importances = np.array(importances)
    mean_importance = importances.mean(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    top_k = 30
    top_indices = np.argsort(mean_importance)[-top_k:][::-1]

    ax.barh(range(top_k), mean_importance[top_indices], color='#FF9800', alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Dim {i}' for i in top_indices], fontsize=8)
    ax.set_xlabel('Mean |Gradient|', fontsize=12)
    ax.set_title('Gradient-Based Feature Importance — Top 30 Embedding Dimensions\n'
                  'Central Model MLP Scoring Head', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    shap_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [XAI] Gradient importance saved to {shap_path}")


if __name__ == "__main__":
    from models.central_model import CentralModel

    model = CentralModel()
    model.load(os.path.join(PROJECT_ROOT, "models", "central_model.pkl"))

    results_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    run_xai_analysis(model, results, output_dir)
