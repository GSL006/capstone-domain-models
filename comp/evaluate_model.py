import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our custom classes from the training script
from withSmoteTransformer import (
    CSBiasFeatureExtractor, 
    HierarchicalCSPaperDataset, 
    CSBiasPredictionModel
)

def load_single_paper(json_file_path):
    """Load single paper from JSON file for prediction - only needs Body and Reason"""
    
    if not os.path.exists(json_file_path):
        return pd.DataFrame(columns=['text', 'reason'])
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        
        for paper in data:
            if not isinstance(paper, dict):
                continue
            
            # Extract text (Body field) - INPUT for model
            text = paper.get('Body', '')
            if not text:
                continue
            
            # Extract reason - INPUT for feature extraction
            reason = paper.get('Reason', '')
            
            # NOTE: We do NOT load Overall Bias - that's what we're trying to PREDICT!
            
            papers.append({
                'text': text,
                'reason': reason
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        return df
        
    except Exception as e:
        return pd.DataFrame(columns=['text', 'reason'])

def load_trained_model(model_path, device):
    """Load the trained CS bias prediction model"""
    
    try:
        # Load model info (contains everything including label mapping)
        model_info = torch.load(model_path, map_location=device)
        
        # Get label mapping from model info
        label_mapping = model_info['label_mapping']
        
        # Get model parameters
        num_classes = model_info['num_classes']
        feature_names = model_info['feature_names']
        
        # Initialize model with bert-base-uncased to match training
        model = CSBiasPredictionModel(
            'bert-base-uncased',
            num_classes=num_classes,
            feature_dim=len(feature_names)
        )
        
        # Load trained weights
        model.load_state_dict(model_info['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, label_mapping, feature_names
        
    except Exception as e:
        return None, None, None

def predict_single_paper(dataloader, model, device):
    """Predict bias for a single paper"""
    
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            
            try:
                outputs = model(input_ids, attention_mask, handcrafted_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                return predicted.cpu().numpy()[0], probabilities.cpu().numpy()[0]
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return None, None
                else:
                    raise e
    
    return None, None

def generate_detailed_results(predictions, true_labels, probabilities, label_mapping):
    """Generate detailed evaluation results"""
    
    # Create reverse label mapping (index to label)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Get unique classes present in the data
    unique_labels = sorted(set(np.concatenate([predictions, true_labels])))
    class_names = [reverse_mapping[label] for label in unique_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=unique_labels
    )
    
    # Calculate weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Calculate macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    # Create detailed results dictionary
    results = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro)
        },
        'class_metrics': {},
        'confusion_matrix': confusion_matrix(true_labels, predictions, labels=unique_labels).tolist(),
        'class_names': class_names,
        'label_mapping': label_mapping,
        'test_samples': len(true_labels),
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        results['class_metrics'][class_name] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }
    
    return results

def create_visualizations(results):
    """Create visualization plots for the evaluation results"""
    
    class_names = results['class_names']
    cm = np.array(results['confusion_matrix'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CS Bias Prediction Model Evaluation Results', fontsize=16)
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,1])
    axes[0,1].set_title('Normalized Confusion Matrix')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('True')
    
    # 3. Per-class Performance Metrics
    class_metrics = results['class_metrics']
    metrics_df = pd.DataFrame(class_metrics).T
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[1,0].bar(x - width, metrics_df['precision'], width, label='Precision', alpha=0.8)
    axes[1,0].bar(x, metrics_df['recall'], width, label='Recall', alpha=0.8)
    axes[1,0].bar(x + width, metrics_df['f1_score'], width, label='F1-Score', alpha=0.8)
    
    axes[1,0].set_xlabel('Classes')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Per-Class Performance Metrics')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(class_names, rotation=45)
    axes[1,0].legend()
    axes[1,0].set_ylim(0, 1)
    
    # 4. Overall Metrics Comparison
    overall_metrics = results['overall_metrics']
    metric_names = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 
                   'F1 (Weighted)', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
    metric_values = [overall_metrics['accuracy'], overall_metrics['precision_weighted'],
                    overall_metrics['recall_weighted'], overall_metrics['f1_weighted'],
                    overall_metrics['precision_macro'], overall_metrics['recall_macro'],
                    overall_metrics['f1_macro']]
    
    bars = axes[1,1].bar(range(len(metric_names)), metric_values, alpha=0.8, color='skyblue')
    axes[1,1].set_xlabel('Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title('Overall Performance Metrics')
    axes[1,1].set_xticks(range(len(metric_names)))
    axes[1,1].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cs_bias_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Evaluation visualizations saved as 'cs_bias_evaluation_results.png'")

def print_detailed_report(results):
    """Print detailed evaluation report to console"""
    
    print("\n" + "="*80)
    print("CS BIAS PREDICTION MODEL - EVALUATION REPORT")
    print("="*80)
    
    print(f"\nEvaluation completed at: {results['evaluation_timestamp']}")
    print(f"Total test samples: {results['test_samples']}")
    
    print("\n" + "-"*50)
    print("OVERALL PERFORMANCE METRICS")
    print("-"*50)
    
    overall = results['overall_metrics']
    print(f"Accuracy:                 {overall['accuracy']:.4f}")
    print(f"Precision (Weighted):     {overall['precision_weighted']:.4f}")
    print(f"Recall (Weighted):        {overall['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted):      {overall['f1_weighted']:.4f}")
    print(f"Precision (Macro):        {overall['precision_macro']:.4f}")
    print(f"Recall (Macro):           {overall['recall_macro']:.4f}")
    print(f"F1-Score (Macro):         {overall['f1_macro']:.4f}")
    
    print("\n" + "-"*50)
    print("PER-CLASS PERFORMANCE METRICS")
    print("-"*50)
    
    class_metrics = results['class_metrics']
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    for class_name, metrics in class_metrics.items():
        print(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} {metrics['support']:<10}")
    
    print("\n" + "-"*50)
    print("CONFUSION MATRIX")
    print("-"*50)
    
    cm = np.array(results['confusion_matrix'])
    class_names = results['class_names']
    
    # Print header
    print(f"{'True \\ Predicted':<20}", end="")
    for name in class_names:
        print(f"{name[:8]:<10}", end="")
    print()
    
    # Print matrix
    for i, true_class in enumerate(class_names):
        print(f"{true_class[:18]:<20}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:<10}", end="")
        print()

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths
    test_data_path = 'single_random_paper.json'
    model_path = 'cs_bias_model_complete.pt'
    
    # Check if required files exist
    if not os.path.exists(test_data_path):
        return
    
    if not os.path.exists(model_path):
        return
    
    # Load single paper
    test_df = load_single_paper(test_data_path)
    
    if len(test_df) == 0:
        return
    
    # Load trained model
    model, label_mapping, feature_names = load_trained_model(
        model_path, device
    )
    
    if model is None:
        return
    
    # Create reverse label mapping (index to label name)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Extract features for the single paper
    extractor = CSBiasFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'], row['reason'])
        test_features.append(features)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    # Setup tokenizer - use bert-base-uncased to match training
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset with same parameters as training
    max_seq_length = 64   # Match training parameters
    max_sections = 3      # Match training parameters  
    max_sents = 4         # Match training parameters
    
    # Use dummy labels since we only need prediction
    dummy_labels = [0] * len(test_df)
    
    test_dataset = HierarchicalCSPaperDataset(
        test_df['text'].tolist(), 
        dummy_labels, 
        tokenizer, 
        test_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Use batch size 1 for single paper
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Get prediction
    predicted_class, probabilities = predict_single_paper(test_dataloader, model, device)
    
    if predicted_class is not None:
        # Output only the prediction
        predicted_label = reverse_mapping[predicted_class]
        print(predicted_label)

if __name__ == "__main__":
    main()
