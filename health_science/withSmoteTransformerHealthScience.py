
import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import random
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SimpleHealthScienceDataset(Dataset):
    """Simplified dataset for health science papers"""

    def __init__(self, texts, labels, tokenizer, handcrafted_features, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.handcrafted_features = handcrafted_features
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        features = self.handcrafted_features[idx]

        # Simple tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'handcrafted_features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SimpleHealthScienceFeatureExtractor:
    """Simplified feature extractor for health science papers"""

    def __init__(self):
        # Core health science terms
        self.health_terms = [
            'patient', 'clinical', 'treatment', 'diagnosis', 'therapy',
            'disease', 'symptom', 'medication', 'intervention', 'outcome',
            'healthcare', 'medical', 'physiological', 'pathology', 'epidemiology'
        ]

        # Statistical patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.\d+'
        self.correlation_pattern = r'r\s*=\s*[-+]?[0-9]*\.?[0-9]+'
        self.odds_ratio_pattern = r'or\s*=\s*[-+]?[0-9]*\.?[0-9]+'
        self.hazard_ratio_pattern = r'hr\s*=\s*[-+]?[0-9]*\.?[0-9]+'

        # Certainty indicators
        self.certainty_words = ['significant', 'clearly', 'definitely', 'proves', 'demonstrates', 'established', 'confirmed']
        self.hedge_words = ['may', 'might', 'suggests', 'appears', 'likely', 'potentially', 'possibly', 'could']

        self.stopwords = set(stopwords.words('english'))

    def extract_features(self, text):
        """Extract key features from health science text"""
        if not text or pd.isna(text):
            return [0] * 12

        text = str(text).lower()
        words = text.split()
        word_count = len(words) + 1

        features = []

        # 1. Text length (normalized)
        features.append(min(len(text) / 1000, 5.0))  # Cap at 5000 chars

        # 2. Health science terminology density
        health_count = sum(text.count(term) for term in self.health_terms)
        features.append(min(health_count / word_count * 100, 10.0))

        # 3. Statistical reporting (normalized)
        p_values = len(re.findall(self.p_value_pattern, text))
        correlations = len(re.findall(self.correlation_pattern, text))
        odds_ratios = len(re.findall(self.odds_ratio_pattern, text))
        hazard_ratios = len(re.findall(self.hazard_ratio_pattern, text))
        features.append(min(p_values, 5.0))
        features.append(min(correlations + odds_ratios + hazard_ratios, 5.0))

        # 4. Certainty vs hedging
        certainty_count = sum(text.count(word) for word in self.certainty_words)
        hedge_count = sum(text.count(word) for word in self.hedge_words)
        features.append(min(certainty_count / word_count * 100, 10.0))
        features.append(min(hedge_count / word_count * 100, 10.0))

        # 5. Clinical methods mentions
        method_terms = ['trial', 'study', 'cohort', 'randomized', 'clinical', 'intervention', 'protocol']
        method_count = sum(text.count(term) for term in method_terms)
        features.append(min(method_count / word_count * 100, 15.0))

        # 6. Data quality indicators
        quality_terms = ['control', 'placebo', 'blinded', 'randomized', 'sample size', 'validation', 'follow-up']
        quality_count = sum(text.count(term) for term in quality_terms)
        features.append(min(quality_count / word_count * 100, 8.0))

        # 7. Figure/table references
        figures = text.count('figure') + text.count('fig.')
        tables = text.count('table')
        features.append(min(figures + tables, 10.0))

        # 8. Uncertainty acknowledgment
        uncertainty_terms = ['limitation', 'error', 'uncertainty', 'assumption', 'confound', 'bias']
        uncertainty_count = sum(text.count(term) for term in uncertainty_terms)
        features.append(min(uncertainty_count / word_count * 100, 8.0))

        # 9. Industry/funding mentions (potential bias indicator)
        industry_terms = ['industry', 'commercial', 'company', 'funding', 'sponsor', 'pharmaceutical']
        industry_count = sum(text.count(term) for term in industry_terms)
        features.append(min(industry_count / word_count * 100, 5.0))

        # 10. Word diversity (vocabulary richness)
        unique_words = len(set(words) - self.stopwords)
        features.append(min(unique_words / word_count, 1.0) if word_count > 0 else 0)

        return features

class SimpleHealthScienceModel(nn.Module):
    """Simplified model for health science bias detection"""

    def __init__(self, bert_model_name, num_classes=3, feature_dim=12, dropout=0.4):
        super(SimpleHealthScienceModel, self).__init__()

        # BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

        # Unfreeze last 4 layers for fine-tuning
        for name, param in self.bert.named_parameters():
            if 'layer.8' in name or 'layer.9' in name or 'layer.10' in name or 'layer.11' in name or 'pooler' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # Combined classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, handcrafted_features):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token embedding
        bert_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        bert_embedding = self.dropout(bert_embedding)

        # Process handcrafted features
        processed_features = self.feature_processor(handcrafted_features)

        # Combine features
        combined_features = torch.cat([bert_embedding, processed_features], dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits

def load_health_science_papers_from_json(json_file_path):
    """
    Load health science research papers from a JSON file.

    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing paper data

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing paper texts and bias labels
    """
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found.")
        return create_sample_data()

    try:
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        papers = []

        # Case 1: List of paper objects
        if isinstance(data, list):
            for i, paper in enumerate(data):
                if not isinstance(paper, dict):
                    print(f"Warning: Item at index {i} is not a dictionary. Skipping.")
                    continue

                text = paper.get('Body', paper.get('text', paper.get('content',
                              paper.get('Abstract', paper.get('abstract', '')))))

                label = (
                    paper.get('Overall Bias') or
                    paper.get('OverallBias') or
                    paper.get('bias_label') or
                    paper.get('bias_type') or
                    paper.get('CognitiveBias') or
                    paper.get('PublicationBias') or
                    paper.get('NoBias', 0)
                )

                # Convert string labels to integers if needed
                if isinstance(label, str):
                    label_map = {
                        'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0,
                        'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1,
                        'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2
                    }
                    label = label_map.get(label.strip().lower(), 0)

                if isinstance(label, float):
                    label = int(label)

                papers.append({'text': text, 'label': label})

        # Case 2: Dictionary with paper data
        elif isinstance(data, dict):
            if 'papers' in data and isinstance(data['papers'], list):
                papers_data = data['papers']
                for i, paper in enumerate(papers_data):
                    if not isinstance(paper, dict):
                        print(f"Warning: Item at index {i} in 'papers' is not a dictionary. Skipping.")
                        continue

                    text = paper.get('text', paper.get('content', paper.get('abstract', '')))
                    label = (
                        paper.get('Overall Bias') or
                        paper.get('OverallBias') or
                        paper.get('label') or
                        paper.get('bias_label') or
                        paper.get('bias_type') or
                        paper.get('CognitiveBias') or
                        paper.get('PublicationBias') or
                        paper.get('NoBias', 0)
                    )

                    if isinstance(label, str):
                        label_map = {
                            'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0,
                            'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1,
                            'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2
                        }
                        label = label_map.get(label.strip().lower(), 0)

                    if isinstance(label, float):
                        label = int(label)

                    papers.append({'text': text, 'label': label})
            else:
                for paper_id, paper_data in data.items():
                    if isinstance(paper_data, dict):
                        text = paper_data.get('Body', paper_data.get('text', paper_data.get('content',
                                          paper_data.get('Abstract', paper_data.get('abstract', '')))))

                        label = (
                            paper_data.get('Overall Bias') or
                            paper_data.get('OverallBias') or
                            paper_data.get('label') or
                            paper_data.get('bias_label') or
                            paper_data.get('bias_type') or
                            paper_data.get('CognitiveBias') or
                            paper_data.get('PublicationBias') or
                            paper_data.get('NoBias', 0)
                        )

                        if isinstance(label, str):
                            label_map = {
                                'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0,
                                'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1,
                                'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2
                            }
                            label = label_map.get(label.strip().lower(), 0)

                        if isinstance(label, float):
                            label = int(label)

                        papers.append({'text': text, 'label': label})

        else:
            print(f"Error: Unsupported JSON structure in {json_file_path}")
            return create_sample_data()

        # Create DataFrame
        df = pd.DataFrame(papers)

        if len(df) == 0:
            print(f"Warning: No valid paper data found in {json_file_path}")
            return create_sample_data()

        if 'text' not in df.columns:
            print(f"Warning: 'text' column not found in the JSON data from {json_file_path}")
            df['text'] = ""

        if 'label' not in df.columns:
            print(f"Warning: 'label' column not found in the JSON data from {json_file_path}")
            df['label'] = 0

        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)

        original_count = len(df)
        df = df[df['text'].str.strip().str.len() > 10].reset_index(drop=True)
        if len(df) < original_count:
            print(f"Info: Removed {original_count - len(df)} rows with empty text")

        if not all(df['label'].isin([0, 1, 2])):
            invalid_labels = df[~df['label'].isin([0, 1, 2])]['label'].unique()
            print(f"Warning: Found invalid label values: {invalid_labels}. Converting to 0.")
            df.loc[~df['label'].isin([0, 1, 2]), 'label'] = 0

        print(f"Successfully loaded {len(df)} health science papers from {json_file_path}")
        return df

    except json.JSONDecodeError as e:
        print(f"Error: {json_file_path} is not a valid JSON file: {str(e)}")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading papers from {json_file_path}: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample health science data"""
    sample_texts = [
        "This clinical study demonstrates significant treatment effects with p < 0.01 using randomized controlled trial methodology.",
        "Our therapeutic intervention shows potential for patient outcomes in healthcare applications.",
        "Medical analysis reveals pathological patterns requiring immediate clinical intervention strategies.",
        "Health assessment indicates beneficial patient impacts from therapeutic processes implemented.",
        "Laboratory experiments confirm treatment efficacy using evidence-based medical methods.",
        "Field studies suggest potential health outcomes requiring comprehensive further investigation and analysis."
    ] * 50

    sample_labels = [0, 1, 2, 0, 1, 2] * 50

    return pd.DataFrame({'text': sample_texts, 'label': sample_labels})

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char.isalpha() or char == " "])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(text, n=10):
    words = word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    new_text = ' '.join(new_words)
    return new_text

def augment_data(texts, labels, features, target_count, minority_classes=[0,1]):
    augmented_texts = []
    augmented_labels = []
    augmented_features = []

    label_counts = Counter(labels)

    for cls in minority_classes:
        current_count = label_counts[cls]
        if current_count == 0:
            continue
        needed = target_count - current_count
        if needed <= 0:
            continue

        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        cls_texts = [texts[i] for i in cls_indices]
        cls_features = [features[i] for i in cls_indices]

        aug_per_sample = max(1, needed // len(cls_texts)) + 1
        generated = 0

        while generated < needed:
            for t, f in zip(cls_texts, cls_features):
                new_t = synonym_replacement(t, n=random.randint(5, 15))
                augmented_texts.append(new_t)
                augmented_labels.append(cls)
                augmented_features.append(f)  # Keep same features, or could recompute
                generated += 1
                if generated >= needed:
                    break

    return augmented_texts, augmented_labels, augmented_features

def train_simple_model(model, train_loader, val_loader, device, epochs=20, lr=2e-5, class_weights=None, patience=5, accumulation_steps=4):
    """Train the simplified model with early stopping and memory optimization for GTX 1650"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Add gradient clipping
    max_grad_norm = 1.0

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Aggressive memory management for GTX 1650
            if torch.cuda.is_available():
                if batch_idx % 2 == 0:  # Clear cache every 2 batches
                    torch.cuda.empty_cache()
                if batch_idx % 5 == 0:  # Synchronize every 5 batches
                    torch.cuda.synchronize()

            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)

                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected: {loss}")
                    # Free memory
                    del input_ids, attention_mask, features, labels, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()

                # Free intermediate tensors to save memory
                del input_ids, attention_mask, features, labels, outputs

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clear cache after gradient updates
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                total_train_loss += loss.item() * accumulation_steps

                # Memory monitoring for GTX 1650
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"Batch {batch_idx}/{len(train_loader)}: GPU Memory - Used: {memory_used:.2f}GB, Cached: {memory_cached:.2f}GB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}. Clearing cache and skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

        # Final gradient update if there are remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                # Memory management during validation
                if torch.cuda.is_available() and val_batch_idx % 3 == 0:
                    torch.cuda.empty_cache()

                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    features = batch['handcrafted_features'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids, attention_mask, features)
                    loss = criterion(outputs, labels)

                    total_val_loss += loss.item()

                    _, predictions = torch.max(outputs, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

                    # Free memory
                    del input_ids, attention_mask, features, labels, outputs, loss, predictions

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM during validation at batch {val_batch_idx}. Clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss = {avg_train_loss:.4f}, '
              f'Val Loss = {avg_val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}')

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping check on val acc
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, test_loader, device, label_names):
    """Evaluate the model with memory optimization"""
    model.eval()
    all_predictions = []
    all_labels = []

    # Clear cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for test_batch_idx, batch in enumerate(test_loader):
            # Memory management during evaluation
            if torch.cuda.is_available() and test_batch_idx % 3 == 0:
                torch.cuda.empty_cache()

            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, features)
                _, predictions = torch.max(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Free memory
                del input_ids, attention_mask, features, labels, outputs, predictions

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM during evaluation at batch {test_batch_idx}. Clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Health Science Bias Detection - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('health_science_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, all_predictions, all_labels

def main():
    # Configuration with memory management for GTX 1650
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear cache and set memory management
        torch.cuda.empty_cache()
        # Enable memory-efficient settings for GTX 1650
        torch.backends.cudnn.benchmark = False  # More stable for small batches
        torch.backends.cudnn.deterministic = True
    else:
        print("CUDA not available. Using CPU. For GPU support, install CUDA toolkit and PyTorch with CUDA.")

    # Load data
    print("Loading health science papers...")
    json_file_path = 'health_sciences_papers.json'
    papers_df = load_health_science_papers_from_json(json_file_path)

    print("Class distribution:")
    print(papers_df['label'].value_counts())

    # Create label mapping (for saving model)
    unique_labels = sorted(papers_df['label'].unique())
    label_mapping = {
        'No Bias': 0,
        'Cognitive Bias': 1,
        'Publication Bias': 2
    }
    print(f"Label mapping: {label_mapping}")

    # Extract features
    print("Extracting features...")
    extractor = SimpleHealthScienceFeatureExtractor()
    
    # Define feature names for saving
    feature_names = [
        'text_length', 'health_terminology_density', 'p_value_count',
        'statistical_metrics_count', 'certainty_ratio', 'hedge_ratio',
        'clinical_methods_ratio', 'data_quality_ratio', 'figure_table_mentions',
        'uncertainty_acknowledgment', 'industry_funding_mentions', 'word_diversity'
    ]

    features = []
    for text in papers_df['text']:
        feature_vector = extractor.extract_features(text)
        features.append(feature_vector)

    features = np.array(features)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        papers_df['text'], papers_df['label'],
        test_size=0.4, random_state=42, stratify=papers_df['label']
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, random_state=42, stratify=y_temp
    )

    # Get corresponding features
    train_indices = X_train.index
    val_indices = X_val.index
    test_indices = X_test.index

    train_features = features[train_indices]
    val_features = features[val_indices]
    test_features = features[test_indices]

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Augment train data
    print("Augmenting data...")
    train_label_counts = Counter(y_train)
    max_count = max(train_label_counts.values())
    aug_texts, aug_labels, aug_features = augment_data(X_train.values.tolist(), y_train.values.tolist(), train_features.tolist(), max_count)

    # Combine original and augmented
    X_train_aug = list(X_train.values) + aug_texts
    y_train_aug = list(y_train.values) + aug_labels
    train_features_aug = np.vstack([train_features, aug_features])

    print(f"Augmented train size: {len(X_train_aug)}")
    print("Augmented class distribution:")
    print(Counter(y_train_aug))

    # Compute class weights
    classes = np.unique(y_train_aug)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_aug)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Setup tokenizer
    print("Setting up tokenizer...")
    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = SimpleHealthScienceDataset(
        X_train_aug, y_train_aug, tokenizer, train_features_aug
    )
    val_dataset = SimpleHealthScienceDataset(
        X_val.values, y_val.values, tokenizer, val_features
    )
    test_dataset = SimpleHealthScienceDataset(
        X_test.values, y_test.values, tokenizer, test_features
    )

    # Create dataloaders - optimized for GTX 1650 (4GB VRAM)
    if torch.cuda.is_available():
        # Use batch size 1 for GTX 1650 memory constraints
        batch_size = 1
        accumulation_steps = 8  # Gradient accumulation to simulate larger batch
        print("Using batch size of 1 with gradient accumulation (8 steps) for GTX 1650")
    else:
        batch_size = 4
        accumulation_steps = 4
        print(f"Using batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True if torch.cuda.is_available() else False)

    # Initialize model
    print("Initializing model...")
    num_classes = 3
    model = SimpleHealthScienceModel(
        bert_model_name=model_name,
        num_classes=num_classes,
        feature_dim=12,
        dropout=0.4
    )

    model.to(device)

    # Train model
    print("Training model...")
    model = train_simple_model(
        model, train_loader, val_loader, device,
        epochs=8, lr=2e-5, class_weights=class_weights, patience=5,
        accumulation_steps=accumulation_steps
    )

    # Save the final model and metadata (in comp format)
    model_info = {
        'model_state_dict': model.state_dict(),
        'label_mapping': label_mapping,
        'feature_names': feature_names,
        'num_classes': num_classes
    }

    model_save_path = 'best_hierarchical_health_science_model.pt'
    torch.save(model_info, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate model
    print("Evaluating model...")
    label_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
    evaluate_model(model, test_loader, device, label_names)

    print("Training complete!")

if __name__ == "__main__":
    main()

