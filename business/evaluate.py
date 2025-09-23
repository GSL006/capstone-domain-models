import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import os

# Import our custom classes from the training script
from withSmoteTransBusiness import (
    BusinessBiasFeatureExtractor, 
    HierarchicalBusinessPaperDataset, 
    HierarchicalBiasPredictionModel
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
    """Load the trained Business bias prediction model"""
    
    try:
        # The business model was saved as just state_dict, so we need to recreate the structure
        state_dict = torch.load(model_path, map_location=device)
        
        # Fixed parameters matching the training script
        num_classes = 3
        feature_names = [
            'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
            'correlation_count', 'percentage_count', 'hedge_ratio', 'certainty_ratio', 
            'theory_term_ratio', 'jargon_term_ratio', 'method_term_count',
            'validation_pattern_count', 'abstract_claim_ratio', 'results_stat_density',
            'limitations_mentioned', 'performance_claim_ratio', 'claim_consistency',
            'figure_mentions', 'table_mentions', 'chart_mentions', 
            'citation_but_count', 'self_reference_count'
        ]
        
        # Fixed label mapping from training script
        label_mapping = {'No Bias': 0, 'Cognitive Bias': 1, 'Publication Bias': 2}
        
        # Initialize model - try SciBERT first, fall back to bert-base-uncased
        try:
            model = HierarchicalBiasPredictionModel(
                'allenai/scibert_scivocab_uncased',
                num_classes=num_classes,
                feature_dim=len(feature_names),
                dropout_rate=0.5
            )
        except Exception as e:
            print(f"SciBERT not available, using bert-base-uncased: {e}")
            model = HierarchicalBiasPredictionModel(
                'bert-base-uncased',
                num_classes=num_classes,
                feature_dim=len(feature_names),
                dropout_rate=0.5
            )
        
        # Load trained weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, label_mapping, feature_names
        
    except Exception as e:
        print(f"Error loading model: {e}")
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
    
    # File paths
    test_data_path = 'random_papers.json'
    model_path = 'best_hierarchical_business_model.pt'  # Match the actual saved file name
    
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
    extractor = BusinessBiasFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'])  # BusinessBiasFeatureExtractor only takes text
        test_features.append(features)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    # Setup tokenizer - match the model's tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset with same parameters as training
    max_seq_length = 64    # Match training parameters
    max_sections = 2       # Match training parameters  
    max_sents = 4          # Match training parameters
    
    # Use dummy labels since we only need prediction
    dummy_labels = [0] * len(test_df)
    
    test_dataset = HierarchicalBusinessPaperDataset(
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
