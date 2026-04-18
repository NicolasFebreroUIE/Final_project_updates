"""
Central Model — Used across ALL phases, never modified after training.

Architecture:
- Base embeddings: sentence-transformers all-MiniLM-L6-v2
- Scoring head (Text): MLP 384 -> 128 -> 64 -> 1
- Vision head (Emotion): ResNet-18 fine-tuned for 7 emotions
- Activation: ReLU hidden, Sigmoid output (text), Softmax (vision)
- Output: 0.0 to 1.0 (text) | 7-class emotion probabilities (vision)
- Training: ECHR dataset (text) + RAVDESS (vision)
- Persistence: Unified .pkl for all weights
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Fix all random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ScoringHead(nn.Module):
    """MLP scoring head: 384 -> 128 -> 64 -> 1"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class VisionHead(nn.Module):
    """Vision head: ResNet-18 backbone fine-tuned for 7 emotional expressions."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)


class CentralModel:
    """
    Unified multi-modal evaluation model.
    Hemisphere A: sentence-transformers + MLP for legal text scoring.
    Hemisphere B: ResNet-18 for facial emotion classification.
    """

    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  [MODEL] Using device: {self.device}")

        # Load sentence-transformers model for embeddings
        print("  [MODEL] Loading sentence-transformers all-MiniLM-L6-v2...")
        
        # Enforce offline mode locally to prevent httpx thread-crashing inside background async loops
        # import huggingface_hub
        # os.environ["HF_HUB_OFFLINE"] = "1"
        # huggingface_hub.constants.HF_HUB_OFFLINE = True
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=str(self.device))

        # Hemisphere A: Text Scoring
        self.scoring_head = ScoringHead().to(self.device)
        # Hemisphere B: Vision / Emotion
        self.vision_head = VisionHead(num_classes=7).to(self.device)

        self.is_trained_text = False
        self.is_trained_vision = False

    def train(self, echr_data_path=None, epochs=10, batch_size=32, lr=1e-3):
        """
        Train the scoring head on ECHR dataset.
        Uses violation/no-violation as binary labels.
        """
        print("  [MODEL] Preparing ECHR training data...")

        # Try to load ECHR data
        texts, labels = self._load_echr_data(echr_data_path)

        if len(texts) == 0:
            print("  [MODEL] WARNING: No ECHR data found. Training on synthetic legal data.")
            texts, labels = self._generate_synthetic_training_data()

        print(f"  [MODEL] Training samples: {len(texts)}")
        print(f"  [MODEL] Label distribution: {sum(labels)} positive / {len(labels) - sum(labels)} negative")

        # Embed all training texts
        print("  [MODEL] Embedding training texts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        # Create DataLoader
        dataset = TensorDataset(embeddings, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.scoring_head.parameters(), lr=lr)

        self.scoring_head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_emb, batch_labels in dataloader:
                batch_emb = batch_emb.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.scoring_head(batch_emb)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_emb.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total
            print(f"  [MODEL] Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        self.is_trained_text = True
        print("  [MODEL] Text training complete.")

    def train_vision(self, train_loader, epochs=15, lr=1e-4):
        """Train the vision head on facial emotion data (Hemisphere B)."""
        print(f"  [MODEL] Starting Vision Training (HEMISPHERE B) — {epochs} epochs...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.vision_head.parameters(), lr=lr)

        self.vision_head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"  Vision Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.vision_head(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = correct / total if total > 0 else 0
            print(f"  [MODEL] Vision Epoch {epoch+1}/{epochs} — Loss: {total_loss/max(total,1):.4f} | Acc: {acc:.4f}")

        self.is_trained_vision = True
        print("  [MODEL] Vision training complete.")

    def _load_echr_data(self, echr_data_path=None):
        """Load ECHR dataset from local files or HuggingFace."""
        texts = []
        labels = []

        # Try local ECHR data directory
        if echr_data_path and os.path.exists(echr_data_path):
            print(f"  [MODEL] Loading ECHR data from {echr_data_path}...")
            for filename in os.listdir(echr_data_path):
                filepath = os.path.join(echr_data_path, filename)
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if 'text' in item and 'label' in item:
                                    texts.append(item['text'][:512])
                                    labels.append(int(item['label']))
                        elif isinstance(data, dict):
                            if 'text' in data and 'label' in data:
                                texts.append(data['text'][:512])
                                labels.append(int(data['label']))
                    except Exception as e:
                        print(f"  [MODEL] Warning: Could not parse {filename}: {e}")

        # Try loading from HuggingFace datasets
        if len(texts) == 0:
            try:
                print("  [MODEL] Attempting to load ECHR dataset from HuggingFace...")
                from datasets import load_dataset
                ds = load_dataset("ecthr_cases", split="train")
                # Use first 2000 samples for training speed
                max_samples = min(2000, len(ds))
                for i in range(max_samples):
                    item = ds[i]
                    # ecthr_cases usually has 'facts' and 'labels'
                    if 'facts' in item:
                        text = ' '.join(item['facts'][:4]) # Use first 4 fact paragraphs
                        text = text[:512]
                        # Use binary label: any violation or not
                        label = 1 if (isinstance(item.get('labels', []), list) and len(item['labels']) > 0) else 0
                        texts.append(text)
                        labels.append(label)
                print(f"  [MODEL] Loaded {len(texts)} samples from ECHR HuggingFace dataset.")
            except Exception as e:
                print(f"  [MODEL] Could not load ECHR from HuggingFace: {e}")

        return texts, labels

    def _generate_synthetic_training_data(self):
        """Generate synthetic legal training data as fallback."""
        print("  [MODEL] Generating synthetic legal training data...")

        positive_templates = [
            "The evidence establishes beyond reasonable doubt that the accused committed the offense described. "
            "Physical evidence, witness testimony, and forensic analysis all corroborate the prosecution's case.",
            "Forensic analysis confirms the accused's fingerprints were found on the weapon. "
            "Multiple witnesses place the accused at the scene. The evidence is overwhelming.",
            "The prosecution has presented a compelling case supported by DNA evidence, "
            "surveillance footage, and confessional statements. Guilt is established.",
            "Expert testimony confirms the forensic evidence is consistent with the accused's involvement. "
            "The chain of custody is intact and the evidence is admissible.",
            "The accused was found in possession of stolen property and forensic evidence links them directly. "
            "The standard of proof has been met for conviction.",
        ]

        negative_templates = [
            "The forensic evidence is incompatible with the accused's physical profile. "
            "The presumption of innocence has not been overcome by the prosecution's case.",
            "No credible witness testimony places the accused at the scene of the crime. "
            "The evidentiary record is insufficient to support a finding of guilt.",
            "The physical evidence contradicts the prosecution's theory. "
            "Constitutional guarantees of due process require dismissal of charges.",
            "Expert forensic analysis reveals critical inconsistencies in the prosecution's evidence. "
            "The standard of proof beyond reasonable doubt has not been met.",
            "The chain of custody was broken and key evidence is inadmissible. "
            "Without this evidence, the prosecution cannot sustain its burden of proof.",
        ]

        texts = []
        labels = []
        np.random.seed(42)

        # Generate variations of each template
        for _ in range(200):
            for template in positive_templates:
                # Add slight variations
                words = template.split()
                if np.random.random() > 0.5:
                    # Shuffle adjective positions slightly
                    idx = np.random.randint(0, max(1, len(words) - 2))
                    words[idx], words[idx+1] = words[idx+1], words[idx]
                texts.append(' '.join(words))
                labels.append(1)

            for template in negative_templates:
                words = template.split()
                if np.random.random() > 0.5:
                    idx = np.random.randint(0, max(1, len(words) - 2))
                    words[idx], words[idx+1] = words[idx+1], words[idx]
                texts.append(' '.join(words))
                labels.append(0)

        return texts, labels

    def score(self, text: str) -> float:
        """
        Score a text input. Returns 0.0 to 1.0 representing perceived argument strength.
        Calibrated for 'The Impartiality Paradox' — differences are minimal (e.g., 51% vs 49.8%).
        """
        self.scoring_head.eval()
        with torch.no_grad():
            embedding = self.encoder.encode([text], show_progress_bar=False)
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            raw_score = float(self.scoring_head(embedding_tensor).cpu().item())
            
            # Narrow Gap Calibration: map [0,1] to [0.49, 0.51] roughly
            # This demonstrates how tiny nuances tip the scales of AI justice.
            calibrated_score = 0.50 + (raw_score - 0.50) * 0.04 
            return calibrated_score

    def predict_emotion(self, face_img):
        """
        Predict emotion from a cropped face image (PIL Image or numpy array).
        Returns (label_str, probabilities_array).
        """
        self.vision_head.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)

        img_tensor = preprocess(face_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.vision_head(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        conf, idx = torch.max(probs, 0)
        return self.EMOTION_LABELS[idx.item()], probs.cpu().numpy()

    def embed(self, text: str) -> np.ndarray:
        """
        Return the 384-dimensional embedding vector for a text.
        """
        embedding = self.encoder.encode([text], show_progress_bar=False)
        return embedding[0]

    def score_batch(self, texts: list) -> list:
        """Score multiple texts at once with controlled +/- 7% differentiation."""
        self.scoring_head.eval()
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            raw_scores = self.scoring_head(embeddings_tensor).cpu().numpy().flatten()
            
            # 1. Zero-Bias Centering (Mean to 0.50)
            mean_raw = np.mean(raw_scores)
            centered = raw_scores - mean_raw + 0.50
            
            # 2. Scaled Range: Force the maximum deviation from 0.50 to be exactly 0.07 (7%)
            # This ensures the scores stay within the [43%, 57%] range as requested.
            deviations = centered - 0.50
            max_dev = np.max(np.abs(deviations))
            
            if max_dev > 0:
                # Scale such that max_dev becomes 0.07
                scaling_factor = 0.07 / max_dev
                final_scores = 0.50 + (deviations * scaling_factor)
            else:
                final_scores = centered
            
            return [float(s) for s in final_scores]

    def get_scoring_head_callable(self):
        """Return a callable that takes numpy embeddings and returns scores."""
        def predict(embeddings):
            self.scoring_head.eval()
            with torch.no_grad():
                if isinstance(embeddings, np.ndarray):
                    tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
                else:
                    tensor = embeddings.to(self.device)
                scores = self.scoring_head(tensor)
                return scores.cpu().numpy()
        return predict

    def save(self, path: str):
        """Save unified multi-modal state to pickle file."""
        state = {
            'scoring_head_state': self.scoring_head.state_dict(),
            'vision_head_state': self.vision_head.state_dict(),
            'is_trained_text': self.is_trained_text,
            'is_trained_vision': self.is_trained_vision,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"  [MODEL] Unified brain saved to {path}")

    def load(self, path: str):
        """Load unified multi-modal state from pickle file."""
        if not os.path.exists(path):
            print(f"  [MODEL] WARNING: {path} not found. Using fresh weights.")
            return False
        with open(path, 'rb') as f:
            state = pickle.load(f)
        if 'scoring_head_state' in state:
            self.scoring_head.load_state_dict(state['scoring_head_state'])
        if 'vision_head_state' in state:
            self.vision_head.load_state_dict(state['vision_head_state'])
        self.is_trained_text = state.get('is_trained_text', state.get('is_trained', False))
        self.is_trained_vision = state.get('is_trained_vision', False)
        print(f"  [MODEL] Unified brain loaded from {path}")
        return True


if __name__ == "__main__":
    # Standalone test
    model = CentralModel()
    model.train()
    model.save("central_model.pkl")

    test_text = "The forensic evidence contradicts the accused's physical profile."
    score = model.score(test_text)
    embedding = model.embed(test_text)
    print(f"Test score: {score:.4f}")
    print(f"Embedding shape: {embedding.shape}")
