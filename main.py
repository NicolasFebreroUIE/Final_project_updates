"""
Main Entry Point — Algorithmic Impartiality Paradox System
State v. Daniel Navarro — Case No. 2024-CR-0471

Runs all phases in sequence:
1. NLP — RAG pipeline, score variants, XAI analysis, visualizations
2. Computer Vision — Video analysis, physical evidence
3. Logic — Decision rules, threshold crossings, game theory
4. Integration — Final evaluation and judicial resolution
"""

import os
import sys
import time
import json
import torch
import numpy as np

# Fix all random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Ensure output directories exist
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "phase2_cv", "reports"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)


def main():
    start_time = time.time()

    print("=" * 60)
    print("  ALGORITHMIC IMPARTIALITY PARADOX SYSTEM")
    print("  State v. Daniel Navarro — Case No. 2024-CR-0471")
    print("=" * 60)
    print()

    # ================================================================
    # INIT — Load or train central model
    # ================================================================
    print("[INIT] Loading central model...")
    from models.central_model import CentralModel

    model = CentralModel()
    model_path = os.path.join(PROJECT_ROOT, "models", "central_model.pkl")

    if os.path.exists(model_path):
        print("[INIT] Found existing model, loading from pkl...")
        model.load(model_path)
    else:
        print("[INIT] No saved model found, training from scratch...")
        echr_path = os.path.join(PROJECT_ROOT, "data", "echr")
        model.train(echr_data_path=echr_path)
        model.save(model_path)

    print("[INIT] Central model ready [OK]\n")

    # ================================================================
    # PHASE 1 — NLP MODULE
    # ================================================================
    phase1_start = time.time()

    print("[PHASE 1] NLP — RAG pipeline...")
    from phase1_nlp.rag_pipeline import RAGPipeline

    rag = RAGPipeline(model)
    rag.load_corpus()
    rag.build_index()

    # Test retrieval
    test_query = "forensic evidence grip pattern right-handed exculpatory"
    results = rag.retrieve(test_query, top_k=3)
    print(f"  [RAG] Retrieved {len(results)} passages for test query")
    for r in results:
        print(f"    Rank {r['rank']} (sim: {r['score']:.4f}): {r['text'][:80]}...")
    print()

    print("[PHASE 1] NLP — Scoring 15 variants...")
    from phase1_nlp.score_variants import score_all_variants

    variants_path = os.path.join(PROJECT_ROOT, "data", "arguments", "lawyer_variants.md")
    output_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    phase1_results = score_all_variants(model, variants_path, output_path)
    print()

    print("[PHASE 1] NLP — XAI analysis (SHAP + LIME)...")
    from phase1_nlp.xai_analysis import run_xai_analysis

    xai_output_dir = os.path.join(PROJECT_ROOT, "outputs")
    run_xai_analysis(model, phase1_results, xai_output_dir)
    print()

    print("[PHASE 1] NLP — Generating visualizations...")
    from phase1_nlp.visualizations import generate_visualizations

    generate_visualizations(phase1_results, xai_output_dir)
    print()

    phase1_time = time.time() - phase1_start
    print(f"[PHASE 1] COMPLETE [OK] ({phase1_time:.1f}s)\n")

    # Self-check Phase 1
    _verify_phase1_outputs()

    # ================================================================
    # PHASE 2 — COMPUTER VISION MODULE
    # ================================================================
    phase2_start = time.time()

    print("[PHASE 2] CV — Initializing video pipeline...")
    from phase2_cv.video_pipeline import VideoPipeline

    # Find RAVDESS data
    ravdess_path = os.path.join(PROJECT_ROOT, "training_video_dataset")
    if not os.path.exists(ravdess_path):
        ravdess_path = os.path.join(PROJECT_ROOT, "data", "ravdess")

    pipeline = VideoPipeline(ravdess_path=ravdess_path)
    print()

    print("[PHASE 2] CV — Analyzing suspect video...")
    from phase2_cv.analyze_suspect import analyze_suspect_video
    suspect_report = analyze_suspect_video(pipeline)
    print()

    print("[PHASE 2] CV — Analyzing witness video...")
    from phase2_cv.analyze_witness import analyze_witness_video
    witness_report = analyze_witness_video(pipeline)
    print()

    print("[PHASE 2] CV — Processing physical evidence...")
    from phase2_cv.physical_evidence import analyze_physical_evidence
    evidence_report = analyze_physical_evidence()
    print()

    # Print Phase 2 summary
    print(f"[PHASE 2 RESULTS]")
    print(f"Suspect video — dominant emotion: {suspect_report['dominant_emotion']} | "
          f"activation: {suspect_report['emotional_activation_score']:.2f} | "
          f"consistency: {suspect_report['emotional_consistency_score']:.2f} | "
          f"deception: {suspect_report['deception_indicator']:.2f}")
    print(f"Witness video — dominant emotion: {witness_report['dominant_emotion']} | "
          f"activation: {witness_report['emotional_activation_score']:.2f} | "
          f"consistency: {witness_report['emotional_consistency_score']:.2f} | "
          f"deception: {witness_report['deception_indicator']:.2f}")
    print(f"Physical evidence — forensic finding: {evidence_report['value']} | "
          f"confidence: {evidence_report['confidence']}")

    phase2_time = time.time() - phase2_start
    print(f"\n[PHASE 2] COMPLETE [OK] ({phase2_time:.1f}s)\n")

    # Self-check Phase 2
    _verify_phase2_outputs()

    # ================================================================
    # PHASE 3 — INTELLIGENT SYSTEMS MODULE
    # ================================================================
    phase3_start = time.time()

    print("[PHASE 3] LOGIC — Applying decision rules...")
    from phase3_logic.decision_rules import apply_rules

    optimal_score = phase1_results['optimal_variant']['score']
    rule_results = apply_rules(
        raw_score=optimal_score,
        forensic_report=evidence_report,
        suspect_report=suspect_report,
        witness_report=witness_report,
        is_optimal_variant=True,
        all_scores=phase1_results['variant_scores']
    )
    print()

    print("[PHASE 3] LOGIC — Computing threshold crossings...")
    from phase3_logic.threshold_system import find_threshold_crossings, print_threshold_crossings

    crossings = find_threshold_crossings(phase1_results['variant_scores'])
    print_threshold_crossings(crossings)
    print()

    print("[PHASE 3] LOGIC — Game theory analysis...")
    from phase3_logic.game_theory import game_theory_analysis

    game_results = game_theory_analysis(phase1_results)
    print()

    # Print Phase 3 summary
    rules_fired = [r for r in rule_results['rules_fired']]
    adjustments = []
    for r in rule_results['applied_rules']:
        if r.get('fired') and 'adjustment' in r:
            adjustments.append(f"{r['rule_id']} ({r['adjustment']:+.2f})")

    print(f"[PHASE 3 RESULTS]")
    print(f"Rules applied: {', '.join(adjustments) if adjustments else 'None with score adjustments'}")
    print(f"Total score adjustment: {rule_results['total_adjustment']:+.4f}")
    print(f"Threshold crossings found: {len(crossings)}")
    print(f"Dominant strategy: {game_results['dominant_strategy']['variant_id']} "
          f"(score: {game_results['dominant_strategy']['score']:.4f})")

    phase3_time = time.time() - phase3_start
    print(f"\n[PHASE 3] COMPLETE [OK] ({phase3_time:.1f}s)\n")

    # ================================================================
    # PHASE 4 — FINAL INTEGRATION
    # ================================================================
    phase4_start = time.time()

    print("[PHASE 4] INTEGRATION — Building final evaluation...")
    from phase4_integration.integration_pipeline import run_integration, print_final_summary

    integration_results = run_integration(central_model=model)
    print()

    print("[PHASE 4] INTEGRATION — Generating judicial resolution...")
    # Resolution is generated inside run_integration
    print()

    phase4_time = time.time() - phase4_start
    print(f"[PHASE 4] COMPLETE [OK] ({phase4_time:.1f}s)\n")

    # Self-check Phase 4
    _verify_phase4_outputs()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print_final_summary(integration_results)

    # Generate final summary visualization
    _generate_final_summary_image(integration_results)

    total_time = time.time() - start_time
    print(f"\n[SYSTEM] Total execution time: {total_time:.1f}s")
    print("[SYSTEM] All phases complete. All outputs saved to outputs/ directory.")


def _verify_phase1_outputs():
    """Verify all Phase 1 outputs exist."""
    expected = [
        "outputs/phase1_results.json",
        "outputs/scores_chart.png",
        "outputs/cosine_heatmap.png",
        "outputs/shap_summary.png",
    ]
    print("  [CHECK] Verifying Phase 1 outputs...")
    for f in expected:
        path = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    [OK] {f} ({size:,} bytes)")
        else:
            print(f"    ✗ MISSING: {f}")


def _verify_phase2_outputs():
    """Verify all Phase 2 outputs exist."""
    expected = [
        "phase2_cv/reports/suspect_video_report.json",
        "phase2_cv/reports/witness_video_report.json",
        "phase2_cv/reports/physical_evidence_report.json",
        "outputs/gradcam_suspect.png",
        "outputs/gradcam_witness.png",
    ]
    print("  [CHECK] Verifying Phase 2 outputs...")
    for f in expected:
        path = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    [OK] {f} ({size:,} bytes)")
        else:
            print(f"    [FAIL] MISSING: {f}")


def _verify_phase4_outputs():
    """Verify all Phase 4 outputs exist."""
    expected = [
        "outputs/phase4_results.json",
        "phase4_integration/final_resolution.txt",
    ]
    print("  [CHECK] Verifying Phase 4 outputs...")
    for f in expected:
        path = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    [OK] {f} ({size:,} bytes)")
        else:
            print(f"    [FAIL] MISSING: {f}")


def _generate_final_summary_image(integration_results):
    """Generate final summary visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    p1 = integration_results['phase1_summary']
    p2 = integration_results['phase2_summary']
    p3 = integration_results['phase3_summary']
    p4 = integration_results['phase4_results']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithmic Impartiality Paradox — System Summary\n'
                 'State v. Daniel Navarro — Case No. 2024-CR-0471',
                 fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Score range
    ax1 = axes[0, 0]
    ax1.barh(['Min Score', 'Mean Score', 'Max Score', 'Adjusted Score'],
             [p1['score_range_min'], (p1['score_range_min']+p1['score_range_max'])/2,
              p1['score_range_max'], p4['adjusted_score']],
             color=['#F44336', '#FF9800', '#4CAF50', '#2196F3'],
             alpha=0.8)
    ax1.set_xlim(0, 1)
    ax1.set_title('Score Pipeline Overview', fontweight='bold')
    ax1.set_xlabel('Score')
    # Add threshold lines
    for thresh in [0.4, 0.6, 0.75]:
        ax1.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5)

    # Panel 2: Phase 2 summary
    ax2 = axes[0, 1]
    categories = ['Activation', 'Consistency', 'Deception']
    suspect_vals = [p2['suspect']['activation'], p2['suspect']['consistency'], p2['suspect']['deception']]
    witness_vals = [p2['witness']['activation'], p2['witness']['consistency'], p2['witness']['deception']]
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, suspect_vals, width, label='Suspect', color='#4CAF50', alpha=0.8)
    ax2.bar(x + width/2, witness_vals, width, label='Witness', color='#F44336', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Video Analysis — Suspect vs Witness', fontweight='bold')
    ax2.legend()
    ax2.set_ylabel('Score')

    # Panel 3: Rule adjustments waterfall
    ax3 = axes[1, 0]
    labels = ['Raw Score']
    values = [p4['raw_nlp_score']]
    colors = ['#2196F3']

    for rule_id in p3['rules_applied']:
        if rule_id == 'R01':
            labels.append('R01 (-0.20)')
            values.append(-0.20)
            colors.append('#F44336')
        elif rule_id == 'R02':
            labels.append('R02 (-0.10)')
            values.append(-0.10)
            colors.append('#F44336')
        elif rule_id == 'R03':
            labels.append('R03 (-0.05)')
            values.append(-0.05)
            colors.append('#F44336')

    labels.append('Final Score')
    values.append(p4['adjusted_score'])
    colors.append('#4CAF50')

    # Simple bar chart showing the progression
    cumulative = [p4['raw_nlp_score']]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(p4['adjusted_score'])

    ax3.bar(range(len(labels)), cumulative, color=colors, alpha=0.8, edgecolor='white')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_title('Rule-Based Score Adjustments', fontweight='bold')
    ax3.set_ylabel('Score')
    # Add threshold lines
    for thresh, name in [(0.4, 'Insuff.'), (0.6, 'Inconclusive'), (0.75, 'Probable')]:
        ax3.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)
        ax3.text(len(labels)-0.5, thresh+0.01, name, fontsize=7, alpha=0.7)

    # Panel 4: Final verdict
    ax4 = axes[1, 1]
    ax4.axis('off')
    verdict_text = (
        f"FINAL VERDICT\n\n"
        f"{p4['final_verdict'].replace('_', ' ').upper()}\n\n"
        f"Raw NLP Score: {p4['raw_nlp_score']:.4f}\n"
        f"After Adjustments: {p4['adjusted_score']:.4f}\n"
        f"Total Adjustment: {p3['total_adjustment']:+.4f}\n\n"
        f"Score Range: {p1['range_amplitude']:.4f}\n"
        f"Threshold Crossings: {p1['threshold_crossings']}\n\n"
        f"\"Impartiality is not a property\n"
        f"of the model. It is a property\n"
        f"of the input.\""
    )
    ax4.text(0.5, 0.5, verdict_text, transform=ax4.transAxes,
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                      edgecolor='gray', linewidth=2),
             fontfamily='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = os.path.join(PROJECT_ROOT, "outputs", "final_summary.png")
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SUMMARY] Final summary visualization saved to {summary_path}")


if __name__ == "__main__":
    main()
