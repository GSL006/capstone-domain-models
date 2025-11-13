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

# Import classes from the training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from humanities import (
    HierarchicalHumanitiesPaperDataset,
    HumanitiesBiasFeatureExtractor,
    HierarchicalBiasPredictionModel
)

def load_papers(json_file_path):
    """Load papers from JSON file for prediction"""
    if not os.path.exists(json_file_path):
        return pd.DataFrame(columns=['text', 'reason', 'true_label'])
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        papers_list = [data] if isinstance(data, dict) else data
        
        for paper in papers_list:
            if not isinstance(paper, dict):
                continue
            
            text = paper.get('Body', paper.get('text', ''))
            if not text:
                continue
            
            reason = paper.get('Reason', '')
            true_label = paper.get('Overall Bias') or paper.get('OverallBias', '')
            
            papers.append({
                'text': text,
                'reason': reason,
                'true_label': true_label
            })
        
        return pd.DataFrame(papers)
    except Exception as e:
        return pd.DataFrame(columns=['text', 'reason', 'true_label'])

def load_trained_model(model_path, device):
    """Load the trained Humanities bias prediction model"""
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        num_classes = 3
        feature_names = [
            'length', 'avg_word_length', 'hedge_ratio', 'certainty_ratio',
            'interpretive_ratio', 'theory_term_ratio', 'normative_ratio',
            'emotional_ratio', 'authority_ratio', 'source_ratio',
            'comparative_ratio', 'method_ratio', 'first_person_ratio',
            'quotation_density', 'abstract_interpretive_ratio',
            'analysis_certainty_ratio', 'limitations_mentioned',
            'perspective_diversity', 'claim_count', 'evidence_count',
            'citation_density', 'temporal_ratio', 'claim_consistency',
            'self_reference_count', 'passive_voice_ratio'
        ]
        
        label_mapping = {'No Bias': 0, 'Cognitive Bias': 1, 'Publication Bias': 2}
        
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    model = HierarchicalBiasPredictionModel(
                        'allenai/scibert_scivocab_uncased',
                        num_classes=num_classes,
                        feature_dim=len(feature_names),
                        dropout_rate=0.4
                    )
        except Exception as e:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    model = HierarchicalBiasPredictionModel(
                        'bert-base-uncased',
                        num_classes=num_classes,
                        feature_dim=len(feature_names),
                        dropout_rate=0.4
                    )
        
        try:
            model.load_state_dict(state_dict)
        except Exception as load_error:
            pass
        
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
                
                batch_predictions = predicted.cpu().numpy()
                predictions.extend(batch_predictions)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_size = input_ids.size(0)
                    predictions.extend([0] * batch_size)
                else:
                    raise e
    
    return predictions

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_upload.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'humanities.pt')
    
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    test_df = load_papers(json_file_path)
    
    if len(test_df) == 0:
        print("No valid papers found in the test file!")
        return
    
    model, label_mapping, feature_names = load_trained_model(model_path, device)
    
    if model is None:
        print("Failed to load model!")
        return
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    extractor = HumanitiesBiasFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'])
        test_features.append(features)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    max_seq_length = 512
    max_sections = 8
    max_sents = 16
    
    dummy_labels = [0] * len(test_df)
    
    test_dataset = HierarchicalHumanitiesPaperDataset(
        test_df['text'].tolist(),
        dummy_labels,
        tokenizer,
        test_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    predictions = predict_papers(test_dataloader, model, device)
    
    for prediction in predictions:
        bias_type = reverse_mapping[prediction]
        print(f"{bias_type}")

if __name__ == "__main__":
    main()

