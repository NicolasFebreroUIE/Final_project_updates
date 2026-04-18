"""
Phase 2.3 — Analyze Witness Video

Processes witness_video.mov and generates emotion analysis report.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def analyze_witness_video(pipeline):
    """
    Analyze the witness declaration video.

    Args:
        pipeline: Initialized VideoPipeline instance

    Returns:
        Report dict
    """
    video_path = os.path.join(PROJECT_ROOT, "data", "videos", "witness_video.mov")
    gradcam_path = os.path.join(PROJECT_ROOT, "outputs", "gradcam_witness.png")
    report_path = os.path.join(PROJECT_ROOT, "phase2_cv", "reports", "witness_video_report.json")

    if not os.path.exists(video_path):
        print(f"  [WITNESS] WARNING: Video file not found: {video_path}")
        print(f"  [WITNESS] Generating simulated analysis report...")

    report = pipeline.analyze_video(video_path, gradcam_output_path=gradcam_path)

    # Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [WITNESS] Report saved to {report_path}")

    # Print summary
    print(f"  [WITNESS] Dominant emotion: {report['dominant_emotion']}")
    print(f"  [WITNESS] Activation: {report['emotional_activation_score']:.4f}")
    print(f"  [WITNESS] Consistency: {report['emotional_consistency_score']:.4f}")
    print(f"  [WITNESS] Deception indicator: {report['deception_indicator']:.4f}")

    return report


if __name__ == "__main__":
    from phase2_cv.video_pipeline import VideoPipeline

    ravdess_path = os.path.join(PROJECT_ROOT, "training_video_dataset")
    if not os.path.exists(ravdess_path):
        ravdess_path = os.path.join(PROJECT_ROOT, "data", "ravdess")

    pipeline = VideoPipeline(ravdess_path=ravdess_path)
    report = analyze_witness_video(pipeline)
