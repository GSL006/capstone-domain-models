import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import os

# Import our custom classes from the training script
from withSmoteTransformer import (
    CSBiasFeatureExtractor, 
    HierarchicalCSPaperDataset, 
    CSBiasPredictionModel
)

def load_papers(json_file_path):
    """Load papers from JSON file for prediction - only needs Body and Reason"""
    
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
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_mapping
    # File paths
    test_data_path = 'random_papers.json'
    model_path = 'computer_science.pt'
    
    # Check if required files exist
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    # Load papers
    test_df = load_papers(test_data_path)
    
    if len(test_df) == 0:
        print("No valid papers found in the test file!")
        return
    
    # Load trained model
    model, label_mapping, feature_names = load_trained_model(
        model_path, device
    )
    
    if model is None:
        print("Failed to load model!")
        return
    
    # Create reverse label mapping (index to label name)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Extract features for papers
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
    
    # Use batch size 1 for stability
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Get predictions
    predictions = predict_papers(test_dataloader, model, device)
    
    # Output predictions only (one per line)
    for prediction in predictions:
        predicted_label = reverse_mapping[prediction]
        print(predicted_label)

if __name__ == "__main__":
    main()
