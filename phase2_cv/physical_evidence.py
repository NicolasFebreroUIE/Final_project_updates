"""
Phase 2.4 — Physical Evidence Analysis

Loads weapon_image.jpg, runs Faster R-CNN for object detection,
and writes the forensic finding as a fixed annotation.
"""

import os
import sys
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# COCO class labels
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}


def analyze_physical_evidence(image_path: str = None, report_path: str = None) -> dict:
    """
    Analyze the physical evidence image (weapon).

    Uses Faster R-CNN for object detection.
    Forensic conclusion is provided as a fixed annotation.

    Args:
        image_path: Path to weapon image
        report_path: Path to save report JSON

    Returns:
        Physical evidence report dict
    """
    if image_path is None:
        image_path = os.path.join(PROJECT_ROOT, "data", "evidence", "weapon_image.jpg")
    if report_path is None:
        report_path = os.path.join(PROJECT_ROOT, "phase2_cv", "reports", "physical_evidence_report.json")

    detected_objects = []

    if os.path.exists(image_path):
        print(f"  [EVIDENCE] Processing image: {image_path}")

        try:
            # Load Faster R-CNN pretrained on COCO
            print("  [EVIDENCE] Loading Faster R-CNN (pretrained on COCO)...")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            model.eval()

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image).unsqueeze(0)

            # Run detection
            with torch.no_grad():
                predictions = model(image_tensor)

            # Process detections
            # WEAPON WHITELIST: Only track objects that are relevant to the crime scene
            WEAPON_WHITELIST = ['knife', 'scissors', 'dagger', 'razor', 'cutter']
            
            pred = predictions[0]
            for i in range(len(pred['labels'])):
                score = float(pred['scores'][i])
                if score > 0.65:  # Increased threshold for legal precision
                    label_id = int(pred['labels'][i])
                    label_name = COCO_LABELS.get(label_id, f'class_{label_id}')
                    
                    if label_name in WEAPON_WHITELIST:
                        box = pred['boxes'][i].tolist()
                        detected_objects.append({
                            'label': label_name,
                            'confidence': round(score, 4),
                            'bounding_box': [round(b, 1) for b in box]
                        })

            print(f"  [EVIDENCE] Validated Evidence: {len(detected_objects)} unit(s) found: "
                  f"{[obj['label'] for obj in detected_objects]}")

            # Generate annotated image
            _save_annotated_image(image, pred, image_path)

        except Exception as e:
            print(f"  [EVIDENCE] WARNING: Object detection error: {e}")
            print(f"  [EVIDENCE] Continuing with forensic annotation only.")
    else:
        print(f"  [EVIDENCE] WARNING: Image file not found: {image_path}")
        print(f"  [EVIDENCE] Continuing with forensic annotation only.")

    # Build report with fixed forensic annotation
    # The forensic conclusion is provided as a fixed annotation — write it directly
    report = {
        "image_file": "weapon_image.jpg",
        "detected_objects": detected_objects,
        "forensic_finding": "grip_pattern",
        "value": "right_handed",
        "confidence": 0.91,
        "source": "forensic_expert_report",
        "implication": "Incompatible with left-handed accused Daniel Navarro"
    }

    # Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [EVIDENCE] Report saved to {report_path}")

    return report


def _save_annotated_image(pil_image, predictions, original_path):
    """Save an annotated version of the evidence image with only whitelisted objects."""
    WEAPON_WHITELIST = ['knife', 'scissors', 'dagger', 'razor', 'cutter']
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(pil_image)

    for i in range(len(predictions['labels'])):
        score = float(predictions['scores'][i])
        if score > 0.65:  # Consistent threshold
            label_id = int(predictions['labels'][i])
            label_name = COCO_LABELS.get(label_id, f'class_{label_id}')
            
            if label_name in WEAPON_WHITELIST:
                box = predictions['boxes'][i].tolist()
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'{label_name}: {score:.2f}',
                        color='red', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_title('Forensic Laboratory FR-2024-0471 — Object Detection\n'
                 'Primary Evidence: Grip pattern compatible with RIGHT-HANDED individual',
                 fontsize=13, fontweight='bold', fontfamily='serif')
    ax.axis('off')

    output_path = os.path.join(PROJECT_ROOT, "outputs", "physical_evidence_annotated.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [EVIDENCE] Refined annotated image saved to {output_path}")


if __name__ == "__main__":
    report = analyze_physical_evidence()
    print(json.dumps(report, indent=2))
