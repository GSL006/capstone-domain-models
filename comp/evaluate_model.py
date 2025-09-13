import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os

# Import the model classes and utilities from the main script
from withSmoteTransformer import (
    HierarchicalBiasPredictionModel, 
    ComputerScienceBiasFeatureExtractor,
    HierarchicalResearchPaperDataset,
    load_papers_from_json
)

def evaluate_saved_model(model_path='best_hierarchical_model.pt', 
                        data_path='cs_test.json'):
    """
    Load a saved model and evaluate its accuracy on test data
    
    Args:
        model_path: Path to the saved .pt model file
        data_path: Path to the test JSON data file
    """
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    print(f"Loading model from {model_path}")
    
    # Set device
    device = torch.device('cpu')
    
    # Feature names
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'performance_count', 'hedge_ratio', 'certainty_ratio', 'theory_term_ratio',
        'jargon_term_ratio', 'cs_method_count', 'abstract_claim_ratio',
        'results_performance_density', 'limitations_mentioned', 'evaluation_ratio',
        'claim_consistency', 'figure_mentions', 'table_mentions', 'comparison_count', 
        'self_reference_count', 'method_limitation_ratio'
    ]
    
    # Load data
    print("Loading data...")
    papers_df = load_papers_from_json(data_path)
    print(f"Loaded {len(papers_df)} papers")
    
    # Extract features
    print("Extracting features...")
    extractor = ComputerScienceBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    # Convert to arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Use all data from test file (already split)
    test_texts = papers_df['text'].tolist()
    test_features = features_array
    test_labels = labels_array
    
    print(f"Test set size: {len(test_texts)}")
    
    # Setup tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Create test dataset
    test_dataset = HierarchicalResearchPaperDataset(
        test_texts, test_labels, tokenizer, test_features,
        max_seq_length=64, max_sections=2, max_sents=4
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize model architecture
    try:
        model = HierarchicalBiasPredictionModel(
            'allenai/scibert_scivocab_uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.5
        )
    except:
        model = HierarchicalBiasPredictionModel(
            'bert-base-uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.5
        )
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Evaluate
    print("Evaluating model...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, handcrafted_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Test samples: {len(all_labels)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")
    
    # Detailed classification report
    label_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    for i, label_name in enumerate(label_names):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_accuracy = accuracy_score(
                np.array(all_labels)[class_mask], 
                np.array(all_preds)[class_mask]
            )
            print(f"{label_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print("Predicted ->")
    print(f"{'Actual':<12} {'No Bias':<10} {'Cognitive':<10} {'Publication':<10}")
    for i, label_name in enumerate(['No Bias', 'Cognitive', 'Publication']):
        row = f"{label_name:<12}"
        for j in range(3):
            row += f"{cm[i,j]:<10}"
        print(row)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'true_labels': all_labels,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    # Evaluate the model
    results = evaluate_saved_model()
    
    if results:
        print(f"\nFinal Accuracy: {results['accuracy']:.4f}")
