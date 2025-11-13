import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os
import sys

# Import classes from the training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from withSmoteTransPhySciences import (
    HierarchicalPaperDataset,
    PhysicalScienceFeatureExtractor,
    HierarchicalBiasPredictionModel
)

def load_papers_from_json(json_file_path):
    """Load papers from JSON file"""
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found!")
        return pd.DataFrame(columns=['text', 'label'])
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        for paper in data:
            if not isinstance(paper, dict):
                continue
            
            text = paper.get('Body', paper.get('text', ''))
            if not text:
                continue
            
            # Get label
            label = paper.get('Overall Bias') or paper.get('OverallBias', '')
            if isinstance(label, str):
                label_map = {
                    'No Bias': 0, 'no bias': 0, 'NoBias': 0,
                    'Cognitive Bias': 1, 'cognitive bias': 1, 'CognitiveBias': 1,
                    'Publication Bias': 2, 'publication bias': 2, 'PublicationBias': 2
                }
                label = label_map.get(label, 0)
            elif isinstance(label, (int, float)):
                label = int(label)
            else:
                label = 0
            
            papers.append({'text': text, 'label': label})
        
        return pd.DataFrame(papers)
    except Exception as e:
        print(f"Error loading papers: {e}")
        return pd.DataFrame(columns=['text', 'label'])

def preprocess_papers(papers_df, feature_names):
    """Preprocess papers and extract features"""
    print("Extracting physical science-specific handcrafted features...")
    extractor = PhysicalScienceFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Normalize features
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
    return papers_df['text'].tolist(), features_array, labels_array

def evaluate_model(model_path, papers_df, feature_names, device='cpu'):
    """Evaluate the trained model on the given papers"""
    label_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
    
    # Preprocess papers
    texts, features_array, labels_array = preprocess_papers(papers_df, feature_names)
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}, falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Model configuration (same as training)
    max_seq_length = 128
    max_sections = 8
    max_sents = 8
    
    # Create dataset
    print("Creating evaluation dataset...")
    eval_dataset = HierarchicalPaperDataset(
        texts, labels_array, tokenizer, features_array,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Create dataloader
    batch_size = 4
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model_state = torch.load(model_path, map_location=device)
        
        model = HierarchicalBiasPredictionModel(
            bert_model_name='allenai/scibert_scivocab_uncased',
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.2
        )
        
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Evaluate
    print("Running evaluation...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, handcrafted_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\nEvaluation Results:")
    print(f"="*50)
    print(f"Total papers evaluated: {len(all_labels)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, all_preds, all_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature names (29 features for physical science)
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'correlation_count', 'percentage_count', 'hedge_ratio', 'certainty_ratio',
        'theory_term_ratio', 'promo_term_ratio', 'method_term_count',
        'validation_pattern_count', 'abstract_claim_ratio', 'results_stat_density',
        'limitations_mentioned', 'efficiency_claim_ratio', 'claim_consistency',
        'figure_mentions', 'table_mentions', 'chart_mentions',
        'citation_but_count', 'self_reference_count', 'flesch_reading_ease',
        'flesch_kincaid_grade', 'sentence_length_variance', 'avg_sentence_length',
        'citation_density', 'methodology_detail_ratio', 'data_availability'
    ]
    
    # Load papers
    data_file = 'random.json'
    if not os.path.exists(data_file):
        data_file = 'random_papers.json'
    
    papers_df = load_papers_from_json(data_file)
    
    if len(papers_df) == 0:
        print("No papers found. Exiting.")
        return
    
    # Limit to 1000 papers
    if len(papers_df) > 1000:
        papers_df = papers_df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    # Evaluate model
    model_path = 'physical.pt'
    result = evaluate_model(model_path, papers_df, feature_names, device)
    
    if result:
        accuracy, predictions, true_labels = result
        print(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()

