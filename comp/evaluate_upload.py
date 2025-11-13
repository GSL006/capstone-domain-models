import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import os
import sys
import contextlib

# Import our custom classes from the training script
from withSmoteTransformerComp import (
    CSBiasFeatureExtractor, 
    HierarchicalCSPaperDataset, 
    CSBiasPredictionModel
)

def load_papers(json_file_path):
    """Load papers from JSON file for prediction and evaluation"""
    
    if not os.path.exists(json_file_path):
        return pd.DataFrame(columns=['text', 'reason', 'true_label'])
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        
        for paper in data:
            if not isinstance(paper, dict):
                continue
            
            # Extract fields
            body = paper.get('Body', '')
            reason = paper.get('Reason', '')
            true_label = paper.get('Overall Bias') or paper.get('OverallBias')
            
            # Only process papers with body
            if body and body.strip():
                text = body
                if reason and reason.strip():
                    text += " " + reason
                
                paper_data = {
                    'text': text,
                    'reason': reason
                }
                
                # Add true label if available
                if true_label:
                    paper_data['true_label'] = true_label
                
                papers.append(paper_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        return df
        
    except Exception as e:
        return pd.DataFrame(columns=['text', 'reason'])

def load_trained_model(model_path, device):
    """Load the trained CS bias prediction model"""
    
    try:
        # Suppress loading messages
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
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

def predict_papers(dataloader, model, device):
    """Predict bias for papers"""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            
            try:
                outputs = model(input_ids, attention_mask, handcrafted_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions
                batch_predictions = predicted.cpu().numpy()
                predictions.extend(batch_predictions)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Skip this batch and add default predictions
                    batch_size = input_ids.size(0)
                    predictions.extend([0] * batch_size)  # Default to first class
                else:
                    raise e
    
    return predictions


def main():
    if len(sys.argv) != 2:
        return
    
    json_file_path = sys.argv[1]
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths
    test_data_path = json_file_path
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'comp_sci.pt')
    
    # Check if required files exist
    if not os.path.exists(test_data_path):
        return
    
    if not os.path.exists(model_path):
        return
    
    # Load papers
    test_df = load_papers(test_data_path)
    
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
    
    # Extract features for papers
    extractor = CSBiasFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'], row.get('reason', ''))
        test_features.append(features)
    
    # Convert features to numpy array
    test_features = np.array(test_features)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenization parameters (match training exactly)
    max_seq_length = 64
    max_sections = 3
    max_sents = 4
    
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
    
    # Create DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Predict
    predictions = predict_papers(test_dataloader, model, device)
    
    # Check if we have true labels for evaluation
    if 'true_label' in test_df.columns:
        true_indices = []
        valid_predictions = []
        valid_true_labels = []
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            true_label = row.get('true_label')
            if true_label and true_label.strip() and true_label in label_mapping:
                true_indices.append(label_mapping[true_label])
                valid_predictions.append(predictions[i])
                valid_true_labels.append(true_label)
        
        # Just show predictions
        for i, prediction in enumerate(predictions):
            bias_type = reverse_mapping[prediction]
            print(f"{bias_type}")
    else:
        # Just show predictions
        for i, prediction in enumerate(predictions):
            bias_type = reverse_mapping[prediction]
            print(f"{bias_type}")

if __name__ == "__main__":
    main()