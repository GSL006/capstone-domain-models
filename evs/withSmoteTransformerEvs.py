
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SimpleEnvironmentalDataset(Dataset):
    """Simplified dataset for environmental science papers"""
    
    def __init__(self, texts, labels, tokenizer, handcrafted_features, max_length=256):
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

class SimpleFeatureExtractor:
    """Simplified feature extractor for environmental science papers"""
    
    def __init__(self):
        # Core environmental terms
        self.env_terms = [
            'toxic', 'contamination', 'pollution', 'sustainable', 'environmental',
            'ecosystem', 'biodegradation', 'remediation', 'hazardous', 'green chemistry'
        ]
        
        # Statistical patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.\d+'
        self.correlation_pattern = r'r\s*=\s*[-+]?[0-9]*\.?[0-9]+'
        
        # Certainty indicators
        self.certainty_words = ['significant', 'clearly', 'definitely', 'proves', 'demonstrates']
        self.hedge_words = ['may', 'might', 'suggests', 'appears', 'likely', 'potentially']
        
        self.stopwords = set(stopwords.words('english'))
        
    def extract_features(self, text):
        """Extract key features from environmental science text"""
        if not text or pd.isna(text):
            return [0] * 12
        
        text = str(text).lower()
        words = text.split()
        word_count = len(words) + 1
        
        features = []
        
        # 1. Text length (normalized)
        features.append(min(len(text) / 1000, 5.0))  # Cap at 5000 chars
        
        # 2. Environmental terminology density
        env_count = sum(text.count(term) for term in self.env_terms)
        features.append(min(env_count / word_count * 100, 10.0))
        
        # 3. Statistical reporting (normalized)
        p_values = len(re.findall(self.p_value_pattern, text))
        correlations = len(re.findall(self.correlation_pattern, text))
        features.append(min(p_values, 5.0))
        features.append(min(correlations, 5.0))
        
        # 4. Certainty vs hedging
        certainty_count = sum(text.count(word) for word in self.certainty_words)
        hedge_count = sum(text.count(word) for word in self.hedge_words)
        features.append(min(certainty_count / word_count * 100, 10.0))
        features.append(min(hedge_count / word_count * 100, 10.0))
        
        # 5. Methods mentions
        method_terms = ['analysis', 'experiment', 'study', 'method', 'test']
        method_count = sum(text.count(term) for term in method_terms)
        features.append(min(method_count / word_count * 100, 15.0))
        
        # 6. Data quality indicators
        quality_terms = ['control', 'replicate', 'sample size', 'validation']
        quality_count = sum(text.count(term) for term in quality_terms)
        features.append(min(quality_count / word_count * 100, 8.0))
        
        # 7. Figure/table references
        figures = text.count('figure') + text.count('fig.')
        tables = text.count('table')
        features.append(min(figures + tables, 10.0))
        
        # 8. Uncertainty acknowledgment
        uncertainty_terms = ['limitation', 'error', 'uncertainty', 'assumption']
        uncertainty_count = sum(text.count(term) for term in uncertainty_terms)
        features.append(min(uncertainty_count / word_count * 100, 8.0))
        
        # 9. Industry mentions (potential bias indicator)
        industry_terms = ['industry', 'commercial', 'company', 'funding']
        industry_count = sum(text.count(term) for term in industry_terms)
        features.append(min(industry_count / word_count * 100, 5.0))
        
        # 10. Word diversity (vocabulary richness)
        unique_words = len(set(words) - self.stopwords)
        features.append(min(unique_words / word_count, 1.0) if word_count > 0 else 0)
        
        return features

class SimpleEnvironmentalModel(nn.Module):
    """Simplified model for environmental bias detection"""
    
    def __init__(self, bert_model_name, num_classes=3, feature_dim=12, dropout=0.3):
        super(SimpleEnvironmentalModel, self).__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # Freeze some BERT layers to reduce complexity
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim + 32, 128),
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

def load_environmental_papers_from_json(json_file_path):
    """
    Load environmental research papers from a JSON file.
    
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
        
        print(f"Successfully loaded {len(df)} environmental science papers from {json_file_path}")
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: {json_file_path} is not a valid JSON file: {str(e)}")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading papers from {json_file_path}: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample environmental science data"""
    sample_texts = [
        "This environmental study demonstrates significant contamination with p < 0.01 using advanced analysis methods.",
        "Our sustainable green chemistry approach shows potential for environmental remediation applications.",
        "Toxic effects analysis reveals hazardous pollution patterns requiring immediate intervention strategies.",
        "Environmental assessment indicates beneficial ecosystem impacts from biodegradation processes implemented.",
        "Laboratory experiments confirm hazardous waste treatment using environmentally friendly sustainable methods.",
        "Field studies suggest potential environmental damage requiring comprehensive further investigation and analysis."
    ] * 50
    
    sample_labels = [0, 1, 2, 0, 1, 2] * 50
    
    return pd.DataFrame({'text': sample_texts, 'label': sample_labels})

def train_simple_model(model, train_loader, val_loader, device, epochs=5, lr=5e-5):
    """Train the simplified model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss}")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
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
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss = {avg_train_loss:.4f}, '
              f'Val Loss = {avg_val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}')
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader, device, label_names):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            _, predictions = torch.max(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    plt.title('Environmental Science Bias Detection - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, all_predictions, all_labels

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading environmental science papers...")
    papers_df = load_environmental_papers_from_json('/kaggle/input/environmental-science-papers-json/environmental_science_papers.json')
    
    print("Class distribution:")
    print(papers_df['label'].value_counts())
    
    # Extract features
    print("Extracting features...")
    extractor = SimpleFeatureExtractor()
    
    features = []
    for text in papers_df['text']:
        feature_vector = extractor.extract_features(text)
        features.append(feature_vector)
    
    features = np.array(features)
    
    # Split data (no SMOTE as requested)
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
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    try:
        model_name = 'distilbert-base-uncased'  # Simpler model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = SimpleEnvironmentalDataset(
        X_train.values, y_train.values, tokenizer, train_features
    )
    val_dataset = SimpleEnvironmentalDataset(
        X_val.values, y_val.values, tokenizer, val_features
    )
    test_dataset = SimpleEnvironmentalDataset(
        X_test.values, y_test.values, tokenizer, test_features
    )
    
    # Create dataloaders
    batch_size = 8 if device.type == 'cuda' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = SimpleEnvironmentalModel(
        bert_model_name=model_name,
        num_classes=3,
        feature_dim=12  # Reduced feature count
    )
    
    model.to(device)
    
    # Train model
    print("Training model...")
    model = train_simple_model(
        model, train_loader, val_loader, device, 
        epochs=5, lr=2e-5
    )
    
    # Evaluate model
    print("Evaluating model...")
    label_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
    evaluate_model(model, test_loader, device, label_names)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
