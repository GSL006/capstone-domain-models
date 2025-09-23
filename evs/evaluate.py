#!/usr/bin/env python3
"""
Environmental Science Bias Detection - Model Evaluation
Evaluates multiple papers using the trained environmental science model
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import os
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SimpleEnvironmentalDataset(Dataset):
    """Simplified dataset for environmental science papers"""
    
    def __init__(self, texts, tokenizer, handcrafted_features, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.handcrafted_features = handcrafted_features
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'handcrafted_features': torch.tensor(features, dtype=torch.float32)
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

def load_trained_model(model_path, device):
    """
    Load the trained environmental science model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    device : torch.device
        Device to load the model on
        
    Returns:
    --------
    tuple
        (model, label_mapping, feature_names)
    """
    try:
        # Load model state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Model configuration (based on withSmoteTransformerEvs.py)
        num_classes = 3
        feature_names = [
            'text_length', 'env_terminology_density', 'p_values', 'correlations',
            'certainty_ratio', 'hedge_ratio', 'method_mentions', 'quality_indicators',
            'figure_table_refs', 'uncertainty_acknowledgment', 'industry_mentions',
            'word_diversity'
        ]
        
        # Label mapping for environmental science
        label_mapping = {'No Bias': 0, 'Cognitive Bias': 1, 'Publication Bias': 2}
        
        # Initialize model with correct architecture
        try:
            model_name = 'distilbert-base-uncased'  # As per training script
            model = SimpleEnvironmentalModel(
                bert_model_name=model_name,
                num_classes=num_classes,
                feature_dim=12  # 12 features as per training script
            )
        except:
            model_name = 'bert-base-uncased'
            model = SimpleEnvironmentalModel(
                bert_model_name=model_name,
                num_classes=num_classes,
                feature_dim=12
            )
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"🏗️  Model: {model_name}")
        print(f"📊 Classes: {num_classes}")
        print(f"🔧 Features: {len(feature_names)}")
        
        return model, label_mapping, feature_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("❌ Failed to load model!")
        return None, None, None

def load_papers(json_file_path):
    """
    Load papers from JSON file for evaluation
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing papers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with paper texts
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        for paper in data:
            # Get text content (multiple possible field names)
            text_content = (
                paper.get('Body') or 
                paper.get('text') or 
                paper.get('content') or
                paper.get('Abstract') or
                paper.get('abstract', '')
            )
            
            papers.append({
                'text': str(text_content),
                'title': paper.get('Title', paper.get('title', 'Unknown'))
            })
        
        df = pd.DataFrame(papers)
        print(f"📖 Loaded {len(df)} papers for evaluation")
        return df
        
    except Exception as e:
        print(f"❌ Error loading papers: {e}")
        return pd.DataFrame()

def predict_papers(dataloader, model, device):
    """
    Predict bias labels for papers
    
    Parameters:
    -----------
    dataloader : DataLoader
        DataLoader containing the papers
    model : torch.nn.Module
        Trained model
    device : torch.device
        Device for computation
        
    Returns:
    --------
    list
        List of predicted class indices
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['handcrafted_features'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            _, batch_predictions = torch.max(outputs, dim=1)
            
            predictions.extend(batch_predictions.cpu().numpy())
    
    return predictions

def main():
    """Main evaluation function"""
    print("🌍 Environmental Science Bias Detection - Evaluation")
    print("=" * 55)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # File paths
    test_data_path = 'random_papers.json'
    model_path = 'evs.pt'  # Updated model path
    
    # Check if files exist
    if not os.path.exists(test_data_path):
        print(f"❌ Test data file not found: {test_data_path}")
        print("💡 Run: python split_dataset.py")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Train the model first using: python withSmoteTransformerEvs.py")
        return
    
    # Load model
    print(f"🤖 Loading model from: {model_path}")
    model, label_mapping, feature_names = load_trained_model(model_path, device)
    
    if model is None:
        return
    
    # Create reverse mapping for predictions
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Load test data
    print(f"📖 Loading test papers from: {test_data_path}")
    test_df = load_papers(test_data_path)
    
    if len(test_df) == 0:
        print("❌ No papers loaded for evaluation")
        return
    
    # Extract features
    print("🔧 Extracting features...")
    extractor = SimpleFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'])
        test_features.append(features)
    
    test_features = np.array(test_features)
    
    # Setup tokenizer  
    print("🔤 Setting up tokenizer...")
    try:
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset and dataloader
    max_length = 256  # As per training script
    test_dataset = SimpleEnvironmentalDataset(
        test_df['text'].values, 
        tokenizer, 
        test_features,
        max_length=max_length
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = predict_papers(test_dataloader, model, device)
    
    # Output predictions (one per line as requested)
    print("🎯 Predictions:")
    print("-" * 20)
    for prediction in predictions:
        predicted_label = reverse_mapping[prediction]
        print(predicted_label)
    
    print(f"\n✅ Evaluation completed for {len(predictions)} papers")

if __name__ == "__main__":
    main()
