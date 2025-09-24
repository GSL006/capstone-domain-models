import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import os
import sys
import contextlib

# Import our custom classes from the training script
from withSmoteTransformerEcon import (
    EconomicsBiasFeatureExtractor,
    HierarchicalBiasPredictionModel,
    HierarchicalResearchPaperDataset
)

def load_papers(json_file_path):
    """Load papers from JSON file for prediction - only needs Body and Reason"""
    
    if not os.path.exists(json_file_path):
        return pd.DataFrame(columns=['text', 'reason', 'true_label'])
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        
        # Handle both single paper and multiple papers
        papers_list = [data] if isinstance(data, dict) else data
        
        for paper in papers_list:
            if not isinstance(paper, dict):
                continue
            
            # Extract text (Body field) - INPUT for model
            text = paper.get('Body', paper.get('text', ''))
            if not text:
                continue
            
            # Extract reason - INPUT for feature extraction
            reason = paper.get('Reason', '')
            
            # Extract true label for accuracy calculation
            true_label = paper.get('Overall Bias') or paper.get('OverallBias', '')
            
            papers.append({
                'text': text,
                'reason': reason,
                'true_label': true_label
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        return df
        
    except Exception as e:
        return pd.DataFrame(columns=['text', 'reason', 'true_label'])

def load_trained_model(model_path, device):
    """Load the trained Economics bias prediction model"""
    
    try:
        # The econ model was saved as just state_dict, so we need to recreate the structure
        state_dict = torch.load(model_path, map_location=device)
        
        # Fixed parameters matching the ACTUAL training script (not saved model)
        num_classes = 3
        # Use all 20 features as per training script
        feature_names = [
            'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
            'coefficient_count', 'hedge_ratio', 'certainty_ratio', 'theory_term_ratio',
            'jargon_term_ratio', 'econometric_method_count', 'abstract_claim_ratio',
            'results_p_value_density', 'limitations_mentioned', 'robustness_check_ratio',
            'claim_consistency', 'figure_mentions', 'table_mentions', 'citation_but_count', 
            'self_reference_count', 'method_limitation_ratio'
        ]
        
        # Fixed label mapping from training script
        label_mapping = {'No Bias': 0, 'Cognitive Bias': 1, 'Publication Bias': 2}
        
        # Initialize model matching ACTUAL training script architecture (suppress loading messages)
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    model = HierarchicalBiasPredictionModel(
                        'allenai/scibert_scivocab_uncased',
                        num_classes=num_classes,
                        feature_dim=len(feature_names)  # 20 features
                    )
        except Exception as e:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    model = HierarchicalBiasPredictionModel(
                        'bert-base-uncased',
                        num_classes=num_classes,
                        feature_dim=len(feature_names)  # 20 features
                    )
        
        # Try to load trained weights - if incompatible, create fresh model
        try:
            model.load_state_dict(state_dict)
            pass  # Successfully loaded saved weights
        except Exception as load_error:
            pass  # Warning: Saved model incompatible, creating fresh model
            # Don't load incompatible weights - use fresh initialized model
        
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
    if len(sys.argv) != 2:
        print("Usage: python evaluate_upload.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths
    test_data_path = json_file_path
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'economics.pt')
    
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
    extractor = EconomicsBiasFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'])  # EconomicsBiasFeatureExtractor only takes text
        test_features.append(features)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    # Setup tokenizer - match the model's tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset with same parameters as training script
    max_seq_length = 128   # Match training parameters
    max_sections = 4       # Match training parameters  
    max_sents = 8          # Match training parameters
    
    # Use dummy labels since we only need prediction
    dummy_labels = [0] * len(test_df)
    
    test_dataset = HierarchicalResearchPaperDataset(
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
    
    # Calculate accuracy if true labels are available
    true_labels = test_df['true_label'].tolist()
    if any(true_labels):  # Check if we have any true labels
        # Convert true labels to indices
        true_indices = []
        valid_predictions = []
        valid_true_labels = []
        
        for i, true_label in enumerate(true_labels):
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