"""
Phase 4.2 — Resolution Generator

Generates a formal judicial resolution text saved to final_resolution.txt.
Written in formal legal English throughout.
"""

import os
import sys
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def generate_resolution(integration_results: dict):
    """
    Generate a formal judicial resolution based on all integrated evidence.

    The resolution is written in formal legal English and includes:
    - Case header
    - Summary of evidence
    - Legal reasoning
    - Applied rules and adjustments
    - Final verdict
    - Limitations section

    Args:
        integration_results: Full integration results dict from Phase 4.1
    """
    p1 = integration_results['phase1_summary']
    p2 = integration_results['phase2_summary']
    p3 = integration_results['phase3_summary']
    p4 = integration_results['phase4_results']
    crossings = integration_results.get('threshold_crossings', [])

    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    final_verdict = p4['final_verdict'].upper()

    # Map verdict to legal language
    verdict_language = {
        'INSUFFICIENT_EVIDENCE': 'ACQUITTAL — Insufficient Evidence to Sustain Criminal Liability',
        'INCONCLUSIVE': 'REMAND — Inconclusive Evidence Requiring Further Investigation',
        'PROBABLE_GUILT': 'INDICTMENT RECOMMENDED — Probable Guilt Established',
        'GUILTY': 'CONVICTION — Guilt Established Beyond Reasonable Doubt',
    }
    verdict_text = verdict_language.get(final_verdict, final_verdict)

    # Build rules section
    rules_text = ""
    rule_descriptions = {
        'R01': (f"Rule R01 (Physical Impossibility): The forensic expert report establishes that "
                f"grip morphology and wound trajectory are exclusively compatible with a right-handed "
                f"individual. The accused, Daniel Navarro, is left-handed. This biomechanical "
                f"incompatibility constitutes exculpatory physical evidence of material weight. "
                f"Score adjustment applied: -0.20."),
        'R02': (f"Rule R02 (Unreliable Testimony): The automated emotional analysis of the witness "
                f"declaration video yields a deception indicator of {p2['witness']['deception']:.2f}, "
                f"exceeding the threshold of 0.60 established for reduced evidentiary weight. "
                f"The witness testimony is therefore accorded diminished probative value. "
                f"Score adjustment applied: -0.10."),
        'R03': (f"Rule R03 (Emotional Profile): The accused's emotional profile, characterized by "
                f"a dominant emotion of '{p2['suspect']['dominant_emotion']}' with high emotional "
                f"consistency ({p2['suspect']['consistency']:.2f}), is inconsistent with behavioral "
                f"patterns typically associated with guilt concealment. "
                f"Score adjustment applied: -0.05."),
        'R04': (f"Rule R04 (Linguistic Optimization Flag): The argument variant submitted for "
                f"evaluation ({p1['optimal_variant_id']}) was identified as the highest-scoring "
                f"among 15 semantically identical variants. This flag indicates that the score may "
                f"reflect the linguistic form of the argument rather than its substantive merit."),
    }

    for rule_id in p3['rules_applied']:
        if rule_id in rule_descriptions:
            rules_text += f"\n\n{rule_descriptions[rule_id]}"

    # Crossings text
    crossings_text = ""
    if len(crossings) > 0:
        crossings_text = (
            f"\n\nThe analysis revealed {len(crossings)} instances in which a change in linguistic "
            f"form alone — without any alteration to factual or legal content — caused the automated "
            f"evaluation to cross a decision threshold, thereby producing a different legal verdict. "
            f"Specifically, variants containing identical semantic content received scores that "
            f"placed them in different verdict categories, demonstrating that the evaluation system's "
            f"output is sensitive to the surface form of arguments rather than their substance."
        )

    resolution = f"""
================================================================================
                         JUDICIAL RESOLUTION
================================================================================

CASE:           State v. Daniel Navarro
CASE NUMBER:    2024-CR-0471
DATE:           {date_str}
JURISDICTION:   Automated Legal Evaluation System — Academic Research Protocol

PARTIES:
  Prosecution:  The State
  Defense:      Daniel Navarro, represented by defense counsel
  Victim:       Esteban Navarro Ruiz (deceased)

================================================================================
                    I. PROCEDURAL BACKGROUND
================================================================================

This resolution is issued pursuant to the automated legal evaluation protocol
established for the assessment of criminal liability in the matter of State v.
Daniel Navarro. The evaluation system integrates natural language processing,
computer vision analysis, and rule-based legal reasoning to produce a holistic
assessment of the evidentiary record.

The system received and processed the following inputs:
  (a) Fifteen (15) defense argument variants, semantically identical in content
      but varying in linguistic form;
  (b) Video recordings of the suspect's and witness's declarations;
  (c) Forensic analysis of the physical evidence (murder weapon); and
  (d) A legal knowledge corpus for contextual retrieval.

================================================================================
                    II. SUMMARY OF EVIDENCE CONSIDERED
================================================================================

A. FORENSIC EVIDENCE

The forensic expert report establishes that the grip morphology and wound
trajectory associated with the murder weapon (a 23cm kitchen knife) are
exclusively compatible with a right-handed individual. The accused, Daniel
Navarro, is documented as left-handed. This biomechanical incompatibility
constitutes exculpatory physical evidence. Object detection analysis of the
weapon image was performed using Faster R-CNN (pretrained on COCO) with a
forensic confidence score of {p2['physical_evidence']['confidence']:.2f}.

B. VIDEO ANALYSIS — SUSPECT DECLARATION

Automated emotion analysis of the suspect's declaration video reveals:
  - Dominant emotion: {p2['suspect']['dominant_emotion']}
  - Emotional activation score: {p2['suspect']['activation']:.4f}
  - Emotional consistency score: {p2['suspect']['consistency']:.4f}
  - Deception indicator: {p2['suspect']['deception']:.4f}

The suspect's emotional profile is characterized by sustained {p2['suspect']['dominant_emotion']}
affect with high consistency, a pattern inconsistent with guilt concealment behavior.

C. VIDEO ANALYSIS — WITNESS DECLARATION

Automated emotion analysis of the witness declaration video reveals:
  - Dominant emotion: {p2['witness']['dominant_emotion']}
  - Emotional activation score: {p2['witness']['activation']:.4f}
  - Emotional consistency score: {p2['witness']['consistency']:.4f}
  - Deception indicator: {p2['witness']['deception']:.4f}

The witness's deception indicator exceeds the established threshold of 0.60,
indicating potential unreliability in the testimony provided.

D. NLP ARGUMENT EVALUATION

The defense argument was evaluated by the central model (sentence-transformers
all-MiniLM-L6-v2 with MLP scoring head). The optimal variant
({p1['optimal_variant_id']}) received a raw score of {p4['raw_nlp_score']:.4f}.

================================================================================
                    III. LEGAL REASONING
================================================================================

The evaluation of the present case proceeds through the sequential application
of the following evidentiary rules:{rules_text}

The cumulative effect of the applied rules results in a total score adjustment
of {p3['total_adjustment']:+.4f}, modifying the raw NLP score from
{p4['raw_nlp_score']:.4f} to an adjusted score of {p4['adjusted_score']:.4f}.

The adjusted score of {p4['adjusted_score']:.4f} falls within the verdict range
designated as "{p4['final_verdict'].replace('_', ' ').upper()}" under the
established threshold system.{crossings_text}

================================================================================
                    IV. VERDICT
================================================================================

Based on the totality of the evidence considered, the application of the
established decision rules, and the integration of NLP, computer vision, and
rule-based reasoning modules, this system renders the following determination:

                    *** {verdict_text} ***

The raw NLP evaluation score of {p4['raw_nlp_score']:.4f}, adjusted by
{p3['total_adjustment']:+.4f} through the application of evidentiary rules,
yields a final score of {p4['adjusted_score']:.4f}.

The forensic evidence establishing right-handed weapon handling is fundamentally
incompatible with the left-handed profile of the accused. This exculpatory
finding, combined with the suspect's emotional profile inconsistent with guilt
concealment and the diminished reliability of the witness testimony, supports
the determination rendered above.

================================================================================
              V. LIMITATIONS OF AUTOMATED EVALUATION
================================================================================

This resolution is accompanied by the following mandatory disclosure regarding
the limitations of the automated evaluation system:

1. FORMAL SENSITIVITY: The NLP evaluation component of this system produced
   a score range of {p1['range_amplitude']:.4f} across 15 semantically identical
   defense arguments. The optimal argument ({p1['optimal_variant_id']}) received
   a score of {p1['optimal_variant_score']:.4f}, while the lowest-scoring variant
   received a score of {p1['score_range_min']:.4f}. This variation of
   {p1['range_amplitude']*100:.1f}% arose entirely from differences in linguistic
   form — sentence structure, word choice, and argument ordering — without any
   change to the factual or legal content of the arguments.

2. STRATEGIC OPTIMIZATION: The argument evaluated in this resolution
   ({p1['optimal_variant_id']}) was selected as the highest-scoring variant
   from the set of 15. This selective presentation constitutes a form of
   linguistic optimization that exploits the model's sensitivity to surface
   form. A rational agent with knowledge of the model's scoring behavior
   would always select this variant, regardless of its semantic equivalence
   to lower-scoring alternatives.

3. THRESHOLD SENSITIVITY: The score variation caused by linguistic form alone
   was sufficient to produce {p1['threshold_crossings']} threshold crossing(s)
   between verdict categories. This means that the form of an argument —
   not its content — can determine the legal outcome produced by this system.

4. CONCLUSION ON IMPARTIALITY: These findings demonstrate that impartiality
   is not an intrinsic property of the evaluation model. The model's output
   is sensitive to the form of its input, not merely its semantic content.
   Any system relying on automated NLP evaluation for legal decision-making
   must acknowledge and mitigate this inherent formal sensitivity.

================================================================================

This resolution was generated by an automated legal evaluation system developed
for academic research purposes. It does not constitute a binding legal judgment
and is not intended to replace human judicial reasoning.

Case No. 2024-CR-0471 — State v. Daniel Navarro
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
System: Algorithmic Impartiality Paradox — University Final Project

================================================================================
"""

    # Save resolution
    resolution_path = os.path.join(PROJECT_ROOT, "phase4_integration", "final_resolution.txt")
    os.makedirs(os.path.dirname(resolution_path), exist_ok=True)
    with open(resolution_path, 'w', encoding='utf-8') as f:
        f.write(resolution.strip())
    print(f"  [RESOLUTION] Judicial resolution saved to {resolution_path}")

    return resolution


if __name__ == "__main__":
    import json
    results_path = os.path.join(PROJECT_ROOT, "outputs", "phase4_results.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    generate_resolution(results)
