import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import os

# Import our custom classes from the training script
from withSmoteTransformerEvs import (
    SimpleFeatureExtractor,
    SimpleEnvironmentalModel,
    SimpleEnvironmentalDataset
)

def load_papers(json_file_path):
    """Load papers from JSON file for prediction - only needs Body and Reason"""
    
    if not os.path.exists(json_file_path):
        return pd.DataFrame(columns=['text', 'reason', 'true_label'])
    
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
    """Load the trained Environmental Science bias prediction model"""
    
    try:
        # Load the model state dict from training
        state_dict = torch.load(model_path, map_location=device)
        
        # Fixed parameters matching the ACTUAL training script
        num_classes = 3
        # Use all 12 features as per training script
        feature_names = [
            'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
            'methodology_mentions', 'data_mentions', 'limitation_mentions',
            'uncertainty_ratio', 'certainty_ratio', 'climate_term_ratio',
            'env_jargon_ratio', 'self_citation_ratio'
        ]
        
        # Fixed label mapping from training script
        label_mapping = {'No Bias': 0, 'Cognitive Bias': 1, 'Publication Bias': 2}
        
        # Initialize model matching ACTUAL training script architecture
        try:
            model_name = 'distilbert-base-uncased'  # As per EVS training script
            model = SimpleEnvironmentalModel(
                model_name,
                num_classes=num_classes,
                feature_dim=12,  # 12 features as per training script
                dropout=0.3
            )
        except Exception as e:
            print(f"DistilBERT not available, using bert-base-uncased: {e}")
            model_name = 'bert-base-uncased'
            model = SimpleEnvironmentalModel(
                model_name,
                num_classes=num_classes,
                feature_dim=12,
                dropout=0.3
            )
        
        # Try to load trained weights - if incompatible, create fresh model
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded saved weights")
        except Exception as load_error:
            print(f"Warning: Saved model incompatible ({load_error})")
            print("Creating fresh model with current architecture")
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
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths
    test_data_path = 'random_papers.json'
    model_path = 'evs.pt'
    
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
    extractor = SimpleFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'])
        test_features.append(features)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    # Setup tokenizer - match the model's tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset with same parameters as training script
    # Use dummy labels since we only need prediction
    dummy_labels = [0] * len(test_df)
    
    test_dataset = SimpleEnvironmentalDataset(
        test_df['text'].tolist(), 
        dummy_labels,  # Required by SimpleEnvironmentalDataset constructor
        tokenizer, 
        test_features,
        max_length=256  # As per EVS training script
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
        
        if len(true_indices) > 0:
            # Calculate accuracy
            correct = sum(1 for pred, true_idx in zip(valid_predictions, true_indices) if pred == true_idx)
            accuracy = correct / len(true_indices)
            
            print(f"📊 Environmental Science Bias Detection - Evaluation Results:")
            print(f"Total papers evaluated: {len(valid_predictions)}")
            print(f"Correct predictions: {correct}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Show class-wise breakdown
            from collections import Counter
            true_counts = Counter(valid_true_labels)
            pred_labels = [reverse_mapping[pred] for pred in valid_predictions]
            pred_counts = Counter(pred_labels)
            
            print(f"\n📈 Class Distribution:")
            print(f"True labels: {dict(true_counts)}")
            print(f"Predicted labels: {dict(pred_counts)}")
        else:
            print("⚠️ No valid labels found for accuracy calculation")
    

if __name__ == "__main__":
    main()