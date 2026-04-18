"""
Phase 1.1 — RAG Pipeline over Legal Knowledge

Loads MultiEURLEX Spanish dataset, embeds documents using the central model,
indexes with FAISS for vector similarity search, and retrieves relevant legal passages.
"""

import os
import sys
import json
import numpy as np
import faiss
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline over legal corpus."""

    def __init__(self, central_model, index_path=None):
        self.model = central_model
        self.documents = []
        self.index = None
        self.embeddings = None
        self.index_path = index_path or os.path.join(PROJECT_ROOT, "data", "multieurlex", "faiss_index")

    def load_corpus(self, max_docs=500):
        """Load MultiEURLEX Spanish documents at runtime."""
        print("  [RAG] Loading legal corpus...")

        try:
            from datasets import load_dataset
            print("  [RAG] Loading MultiEURLEX (Spanish) from HuggingFace...")
            ds = load_dataset("multi_eurlex", "all_languages", split="train",
                              trust_remote_code=True)

            count = 0
            for item in tqdm(ds, desc="  [RAG] Extracting Spanish texts", total=min(max_docs, len(ds))):
                if count >= max_docs:
                    break
                try:
                    # MultiEURLEX has 'text' field with language keys
                    text = None
                    if isinstance(item.get('text'), dict):
                        text = item['text'].get('es', item['text'].get('en', ''))
                    elif isinstance(item.get('text'), str):
                        text = item['text']

                    if text and len(text) > 50:
                        # Truncate to reasonable length for embedding
                        self.documents.append(text[:1000])
                        count += 1
                except Exception:
                    continue

            print(f"  [RAG] Loaded {len(self.documents)} documents from MultiEURLEX.")

        except Exception as e:
            print(f"  [RAG] WARNING: Could not load MultiEURLEX: {e}")
            print("  [RAG] Using fallback legal corpus...")
            self._load_fallback_corpus()

    def _load_fallback_corpus(self):
        """Fallback corpus of legal texts for RAG if MultiEURLEX unavailable."""
        fallback_texts = [
            "Article 6 of the European Convention on Human Rights guarantees the right to a fair trial. "
            "This includes the right to be heard by an impartial tribunal, the presumption of innocence, "
            "and the right to adequate time and facilities for the preparation of a defense.",

            "The presumption of innocence is a fundamental principle of criminal law recognized in Article 48 "
            "of the Charter of Fundamental Rights of the European Union. It requires that the burden of proof "
            "rests on the prosecution and that any reasonable doubt must benefit the accused.",

            "Physical evidence in criminal proceedings must satisfy strict chain of custody requirements. "
            "Forensic evidence that has not been properly preserved, documented, and authenticated may be "
            "excluded from consideration by the trier of fact.",

            "The right to an effective remedy under Article 13 ECHR requires that domestic courts provide "
            "adequate procedural safeguards. Automated decision-making systems used in judicial proceedings "
            "must be transparent and subject to human oversight.",

            "Forensic handedness analysis is an established technique in criminal investigation. "
            "The determination of whether a weapon was wielded by a left-handed or right-handed individual "
            "relies on grip pattern morphology and wound trajectory analysis.",

            "The standard of proof beyond reasonable doubt requires the prosecution to establish guilt "
            "to such a degree that a reasonable person would have no logical reason to doubt it. "
            "Mere suspicion or probability is insufficient for criminal conviction.",

            "Expert witness testimony in forensic science must be evaluated for scientific validity, "
            "methodology, and relevance. Courts must assess whether the expert's conclusions are supported "
            "by the underlying data and whether the analytical methods employed are generally accepted.",

            "Article 24.2 of the Spanish Constitution enshrines the right to the presumption of innocence. "
            "This constitutional guarantee operates not merely as a procedural rule but as a substantive "
            "right requiring affirmative evidentiary proof before any finding of criminal liability.",

            "The principle of in dubio pro reo requires that in cases of evidentiary doubt, "
            "the interpretation most favorable to the accused must prevail. This principle operates "
            "as a corollary to the presumption of innocence in criminal proceedings.",

            "Automated legal evaluation systems present challenges to the principle of judicial independence. "
            "The use of algorithmic tools in criminal proceedings requires careful consideration of "
            "transparency, accountability, and the right to an individualized assessment.",

            "The European Court of Human Rights has established that the right to a fair trial "
            "under Article 6 requires equality of arms between prosecution and defense. "
            "This principle demands that each party be afforded a reasonable opportunity to present "
            "its case under conditions that do not put it at a substantial disadvantage.",

            "Criminal liability requires both actus reus and mens rea. The physical element of the offense "
            "must be established through credible evidence, and the mental element must demonstrate "
            "the accused's intent or knowledge at the time of the alleged offense.",

            "The evaluation of witness credibility is a matter within the exclusive competence of the trier "
            "of fact. Factors relevant to credibility assessment include consistency of testimony, "
            "demeanor, opportunity to observe, and potential bias or interest in the outcome.",

            "Regulation (EU) 2016/679 (GDPR) Article 22 establishes the right not to be subject to "
            "decisions based solely on automated processing that produce legal effects. This provision "
            "is particularly relevant to the use of AI systems in judicial decision-making.",

            "The principle of proportionality in criminal sentencing requires that penalties be "
            "commensurate with the gravity of the offense and the degree of culpability of the offender. "
            "Excessive punishment violates fundamental rights protections.",

            "Exculpatory evidence that tends to negate guilt or reduce the severity of the offense "
            "must be disclosed by the prosecution to the defense. Failure to disclose such evidence "
            "may constitute a violation of the right to a fair trial.",

            "The doctrine of fruit of the poisonous tree holds that evidence derived from illegally "
            "obtained evidence is itself inadmissible. This exclusionary rule serves to deter "
            "constitutional violations by law enforcement.",

            "Lateral dominance analysis in forensic science examines patterns of handedness through "
            "tool mark analysis, grip impression morphology, and biomechanical wound trajectory. "
            "This technique has been validated through peer-reviewed scientific literature.",

            "The right to liberty under Article 5 ECHR requires that detention be based on "
            "reasonable suspicion grounded in objective facts. Pre-trial detention must be "
            "proportionate and subject to periodic judicial review.",

            "Algorithmic bias in automated decision-making systems may arise from training data, "
            "feature selection, or model architecture. In the legal context, such bias may result "
            "in disparate treatment of similarly situated individuals, raising equal protection concerns.",
        ]

        self.documents = fallback_texts
        print(f"  [RAG] Loaded {len(self.documents)} fallback legal documents.")

    def build_index(self):
        """Build FAISS index from document embeddings."""
        if len(self.documents) == 0:
            print("  [RAG] WARNING: No documents loaded. Call load_corpus() first.")
            return

        print(f"  [RAG] Embedding {len(self.documents)} documents...")
        self.embeddings = self.model.encoder.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=64
        )
        self.embeddings = np.array(self.embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim after normalization
        self.index.add(self.embeddings)

        print(f"  [RAG] FAISS index built with {self.index.ntotal} vectors (dim={dimension}).")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve top-k most relevant legal passages for a given query.

        Returns:
            List of dicts with 'text', 'score', and 'rank' keys.
        """
        if self.index is None:
            print("  [RAG] WARNING: Index not built. Building now...")
            if len(self.documents) == 0:
                self.load_corpus()
            self.build_index()

        # Embed query
        query_embedding = self.model.encoder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(dist),
                    'rank': rank + 1
                })

        return results


if __name__ == "__main__":
    from models.central_model import CentralModel
    model = CentralModel()
    model.load(os.path.join(PROJECT_ROOT, "models", "central_model.pkl"))

    rag = RAGPipeline(model)
    rag.load_corpus()
    rag.build_index()

    results = rag.retrieve("forensic evidence grip pattern right-handed")
    for r in results:
        print(f"  Rank {r['rank']} (score: {r['score']:.4f}): {r['text'][:100]}...")
