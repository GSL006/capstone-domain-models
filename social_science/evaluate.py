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
from withSmoteTransSocialScience import (
    HierarchicalPaperDataset,
    SocialScienceFeatureExtractor,
    HierarchicalBiasPredictionModel,
    FeatureFusionLayer
)
from transformers import AutoModel

# Create a custom FeatureFusionLayer that outputs 2*fusion_dim (for compatibility with saved models)
class CompatibleFeatureFusionLayer(nn.Module):
    """Feature fusion layer that outputs 2*fusion_dim to match saved model architecture"""
    def __init__(self, bert_dim, handcrafted_dim, fusion_dim=256):
        super(CompatibleFeatureFusionLayer, self).__init__()
        self.bert_projection = nn.Linear(bert_dim, fusion_dim)
        self.handcrafted_projection = nn.Linear(handcrafted_dim, fusion_dim)
        
        # Add batch normalization for better training stability
        self.bert_bn = nn.BatchNorm1d(fusion_dim)
        self.handcrafted_bn = nn.BatchNorm1d(fusion_dim)
        
        # Attention mechanism for dynamic fusion
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, dropout=0.1)
        
        # Gate mechanism with regularization
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # Layer norm expects 2*fusion_dim (concatenated output)
        self.layer_norm = nn.LayerNorm(fusion_dim * 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, bert_embedding, handcrafted_features):
        import torch.nn.functional as F
        
        # Project both embeddings to the fusion dimension
        bert_proj = self.bert_projection(bert_embedding)
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)

        # The model passes 2D tensors [batch, dim], so we handle that case.
        if bert_proj.dim() == 2:
            bert_proj = self.bert_bn(bert_proj)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj)

            # Attention requires a sequence dimension, so we add one.
            attn_output, _ = self.attention(
                bert_proj.unsqueeze(1), 
                handcrafted_proj.unsqueeze(1), 
                handcrafted_proj.unsqueeze(1)
            )
            attn_output = attn_output.squeeze(1)
        else:
            bert_proj = self.bert_bn(bert_proj.transpose(1, 2)).transpose(1, 2)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj.transpose(1, 2)).transpose(1, 2)
            attn_output, _ = self.attention(bert_proj, handcrafted_proj, handcrafted_proj)

        # Gated fusion
        combined = torch.cat([bert_proj, handcrafted_proj], dim=-1)  # [batch_size, 2*fusion_dim]
        gate_values = self.gate(combined)
        fused = gate_values * bert_proj + (1 - gate_values) * handcrafted_proj
        
        # For saved model compatibility: output should be concatenated [fused, handcrafted_proj] = 2*fusion_dim
        # This matches the saved model's layer_norm which expects 1024 (2*512)
        output = torch.cat([fused, handcrafted_proj], dim=-1)  # [batch_size, 2*fusion_dim]
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output

# Create a custom model class that can handle different fusion_dim values
class CompatibleHierarchicalBiasPredictionModel(nn.Module):
    """Compatible model class that can handle different fusion_dim and feature_dim values"""
    def __init__(self, bert_model_name, num_classes=3, dropout_rate=0.3, feature_dim=22, fusion_dim=256):
        super(CompatibleHierarchicalBiasPredictionModel, self).__init__()
        
        print(f"Loading BERT model: {bert_model_name}")
        self.bert = AutoModel.from_pretrained(bert_model_name, add_pooling_layer=False, use_safetensors=True)
        self.bert.gradient_checkpointing_enable()  # Memory optimization for both CPU and GPU
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bert_dim = self.bert.config.hidden_size  # 768 for standard BERT
        self.handcrafted_dim = feature_dim
        self.fusion_dim = fusion_dim
        
        # Simplified attention layers for sentence and section aggregation
        self.sent_attention = nn.Linear(self.bert_dim, 1)
        self.sect_attention = nn.Linear(self.bert_dim, 1)
        
        # Feature fusion layer that outputs 2*fusion_dim
        self.feature_fusion = CompatibleFeatureFusionLayer(
            bert_dim=self.bert_dim, 
            handcrafted_dim=self.handcrafted_dim,
            fusion_dim=self.fusion_dim
        )
        
        # Classification layers - expect input of 2*fusion_dim
        self.fc1 = nn.Linear(self.fusion_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Residual connection for 2*fusion_dim -> 128 shortcut
        self.shortcut = nn.Linear(self.fusion_dim * 2, 128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Increased dropout
    
    def forward(self, input_ids, attention_mask, handcrafted_features):
        # Copy the forward method from the training script
        import torch.nn.functional as F
        
        # input_ids: [batch, sections, sents, seq_len]
        batch_size, n_sections, n_sents, seq_len = input_ids.shape
        
        # 1. Reshape for batched BERT processing
        input_ids_flat = input_ids.view(batch_size * n_sections * n_sents, seq_len)
        attention_mask_flat = attention_mask.view(batch_size * n_sections * n_sents, seq_len)
        
        # 2. Get BERT embeddings for all sentences in one pass
        bert_output = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        sent_embeddings = bert_output.last_hidden_state[:, 0, :]  # [batch*sections*sents, hidden_dim]
        
        # 3. Reshape back to hierarchical structure
        sent_embeddings = sent_embeddings.view(batch_size, n_sections, n_sents, self.bert_dim)
        
        # 4. Hierarchical Attention
        sent_mask = (attention_mask.sum(dim=3) > 0).float()  # [batch, sections, sents]
        
        # Sentence-level attention to get section vectors
        sent_att_weights = self.sent_attention(sent_embeddings)
        sent_att_weights[sent_mask.unsqueeze(-1) == 0] = -1e9
        sent_att_weights = F.softmax(sent_att_weights, dim=2)
        sect_embeddings = torch.einsum("bsnh,bsn->bsh", sent_embeddings, sent_att_weights.squeeze(-1))
        
        # Section-level attention to get document vector
        sect_mask = (sent_mask.sum(dim=2) > 0).float()  # [batch, sections]
        sect_att_weights = self.sect_attention(sect_embeddings)
        sect_att_weights[sect_mask.unsqueeze(-1) == 0] = -1e9
        sect_att_weights = F.softmax(sect_att_weights, dim=1)
        doc_embedding = torch.einsum("bsh,bs->bh", sect_embeddings, sect_att_weights.squeeze(-1))
        
        # 5. Feature Fusion
        fused_features = self.feature_fusion(doc_embedding, handcrafted_features)
        
        # 6. Classification Head
        x = self.fc1(fused_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x2 = self.fc2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        
        shortcut = self.shortcut(fused_features)
        x2 = x2 + shortcut
        x2 = self.dropout(x2)
        output = self.fc3(x2)
        
        return output

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

def preprocess_papers(papers_df, feature_names, expected_feature_dim=None):
    """Preprocess papers and extract features"""
    print("Extracting social science-specific handcrafted features...")
    extractor = SocialScienceFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    features_array = np.array(handcrafted_features, dtype=np.float32)
    
    # Pad or trim features if needed to match expected model input dimension
    if expected_feature_dim is not None and features_array.shape[1] != expected_feature_dim:
        if features_array.shape[1] < expected_feature_dim:
            # Pad with zeros
            padding = np.zeros((features_array.shape[0], expected_feature_dim - features_array.shape[1]), dtype=np.float32)
            features_array = np.concatenate([features_array, padding], axis=1)
            print(f"Padded features from {len(feature_names)} to {expected_feature_dim} dimensions")
        else:
            # Trim excess features
            features_array = features_array[:, :expected_feature_dim]
            print(f"Trimmed features from {len(feature_names)} to {expected_feature_dim} dimensions")
    
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Normalize features
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
    return papers_df['text'].tolist(), features_array, labels_array

def evaluate_model(model_path, papers_df, feature_names, device='cpu', expected_feature_dim=None):
    """Evaluate the trained model on the given papers"""
    label_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
    
    # First, detect model architecture to determine expected_feature_dim
    print(f"Loading model from {model_path}...")
    try:
        model_state = torch.load(model_path, map_location=device)
        
        # Try to infer model parameters from state_dict
        if 'feature_fusion.bert_projection.weight' in model_state:
            bert_proj_shape = model_state['feature_fusion.bert_projection.weight'].shape
            handcrafted_proj_shape = model_state['feature_fusion.handcrafted_projection.weight'].shape
            fusion_dim = bert_proj_shape[0]  # First dimension is fusion_dim
            detected_feature_dim = handcrafted_proj_shape[1]  # Second dimension is feature_dim
            
            print(f"Detected model architecture: fusion_dim={fusion_dim}, feature_dim={detected_feature_dim}")
            
            # Use detected feature_dim if not provided
            if expected_feature_dim is None:
                expected_feature_dim = detected_feature_dim if detected_feature_dim != len(feature_names) else None
        else:
            expected_feature_dim = None
    except Exception as e:
        print(f"Warning: Could not inspect model architecture: {e}")
        expected_feature_dim = None
    
    # Preprocess papers (with optional feature dimension adjustment)
    texts, features_array, labels_array = preprocess_papers(papers_df, feature_names, expected_feature_dim=expected_feature_dim)
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}, falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Model configuration (same as training)
    max_seq_length = 128
    max_sections = 4
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
    try:
        # Try to infer model parameters from state_dict
        if 'feature_fusion.bert_projection.weight' in model_state:
            bert_proj_shape = model_state['feature_fusion.bert_projection.weight'].shape
            handcrafted_proj_shape = model_state['feature_fusion.handcrafted_projection.weight'].shape
            fusion_dim = bert_proj_shape[0]  # First dimension is fusion_dim
            feature_dim = handcrafted_proj_shape[1]  # Second dimension is feature_dim
            
            # If feature_dim doesn't match, use compatible model
            if feature_dim != len(feature_names):
                print(f"Warning: Model expects {feature_dim} features but extractor provides {len(feature_names)}")
                print("Using compatible model architecture to match saved model")
                model = CompatibleHierarchicalBiasPredictionModel(
                    bert_model_name='allenai/scibert_scivocab_uncased',
                    num_classes=3,
                    feature_dim=feature_dim,  # Use detected feature_dim
                    fusion_dim=fusion_dim,  # Use detected fusion_dim
                    dropout_rate=0.3
                )
                model.load_state_dict(model_state, strict=False)
                model.to(device)  # Move to GPU if available
                model.eval()
                print(f"Model loaded with compatible architecture on {device}")
            else:
                # Try standard model first
                try:
                    model = HierarchicalBiasPredictionModel(
                        bert_model_name='allenai/scibert_scivocab_uncased',
                        num_classes=3,
                        feature_dim=len(feature_names),
                        dropout_rate=0.3
                    )
                    model.load_state_dict(model_state, strict=False)
                    model.to(device)  # Move to GPU if available
                    model.eval()
                    print(f"Model loaded with strict=False (architecture mismatch handled) on {device}")
                except Exception as e2:
                    print(f"Standard model failed, trying compatible model: {e2}")
                    model = CompatibleHierarchicalBiasPredictionModel(
                        bert_model_name='allenai/scibert_scivocab_uncased',
                        num_classes=3,
                        feature_dim=feature_dim,
                        fusion_dim=fusion_dim,
                        dropout_rate=0.3
                    )
                    model.load_state_dict(model_state, strict=False)
                    model.to(device)  # Move to GPU if available
                    model.eval()
                    print(f"Model loaded with compatible architecture on {device}")
        else:
            # Standard loading
            model = HierarchicalBiasPredictionModel(
                bert_model_name='allenai/scibert_scivocab_uncased',
                num_classes=3,
                feature_dim=len(feature_names),
                dropout_rate=0.3
            )
            model.load_state_dict(model_state)
            model.to(device)  # Move to GPU if available
            model.eval()
            print(f"Model loaded successfully on {device}!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading with strict=False as fallback
        try:
            print("Attempting to load model with strict=False...")
            model = HierarchicalBiasPredictionModel(
                bert_model_name='allenai/scibert_scivocab_uncased',
                num_classes=3,
                feature_dim=len(feature_names),
                dropout_rate=0.3
            )
            model.load_state_dict(model_state, strict=False)
            model.to(device)
            model.eval()
            print(f"Model loaded with strict=False (some layers may not match) on {device}")
        except Exception as e2:
            print(f"Failed to load model even with strict=False: {e2}")
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
    
    # Feature names (22 features for social science)
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'correlation_count', 'percentage_count', 'hedge_ratio', 'certainty_ratio',
        'theory_term_ratio', 'jargon_term_ratio', 'method_term_count',
        'validation_pattern_count', 'abstract_claim_ratio', 'results_stat_density',
        'limitations_mentioned', 'performance_claim_ratio', 'claim_consistency',
        'figure_mentions', 'table_mentions', 'chart_mentions',
        'citation_but_count', 'self_reference_count'
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
    model_path = 'social.pt'
    result = evaluate_model(model_path, papers_df, feature_names, device)
    
    if result:
        accuracy, predictions, true_labels = result
        print(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()

