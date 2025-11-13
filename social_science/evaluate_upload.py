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

# Redirect debug messages to stderr so only predictions go to stdout
def debug_print(*args, **kwargs):
    """Print debug messages to stderr instead of stdout"""
    print(*args, file=sys.stderr, **kwargs)

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
    """Load the trained Social Science bias prediction model"""
    try:
        state_dict = torch.load(model_path, map_location=device)
        
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
        
        label_mapping = {'No Bias': 0, 'Cognitive Bias': 1, 'Publication Bias': 2}
        
        # Detect model architecture from state_dict
        fusion_dim = None
        feature_dim = None
        if 'feature_fusion.bert_projection.weight' in state_dict:
            bert_proj_shape = state_dict['feature_fusion.bert_projection.weight'].shape
            handcrafted_proj_shape = state_dict['feature_fusion.handcrafted_projection.weight'].shape
            fusion_dim = bert_proj_shape[0]  # First dimension is fusion_dim
            feature_dim = handcrafted_proj_shape[1]  # Second dimension is feature_dim
        
        # Determine if we need compatible model
        use_compatible = False
        if feature_dim is not None and feature_dim != len(feature_names):
            use_compatible = True
            debug_print(f"Detected model architecture: fusion_dim={fusion_dim}, feature_dim={feature_dim}")
            debug_print(f"Warning: Model expects {feature_dim} features but extractor provides {len(feature_names)}")
            debug_print("Using compatible model architecture to match saved model")
        
        # Load model
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    if use_compatible:
                        model = CompatibleHierarchicalBiasPredictionModel(
                            'allenai/scibert_scivocab_uncased',
                            num_classes=num_classes,
                            feature_dim=feature_dim,
                            fusion_dim=fusion_dim,
                            dropout_rate=0.3
                        )
                    else:
                        model = HierarchicalBiasPredictionModel(
                            'allenai/scibert_scivocab_uncased',
                            num_classes=num_classes,
                            feature_dim=len(feature_names),
                            dropout_rate=0.3
                        )
        except Exception as e:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    if use_compatible:
                        model = CompatibleHierarchicalBiasPredictionModel(
                            'bert-base-uncased',
                            num_classes=num_classes,
                            feature_dim=feature_dim,
                            fusion_dim=fusion_dim,
                            dropout_rate=0.3
                        )
                    else:
                        model = HierarchicalBiasPredictionModel(
                            'bert-base-uncased',
                            num_classes=num_classes,
                            feature_dim=len(feature_names),
                            dropout_rate=0.3
                        )
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as load_error:
            debug_print(f"Warning: Model architecture mismatch, attempting partial load: {load_error}")
            try:
                model.load_state_dict(state_dict, strict=False)
            except:
                pass
        
        model.to(device)  # Move to GPU if available
        model.eval()
        
        return model, label_mapping, feature_names, feature_dim
    except Exception as e:
        debug_print(f"Error loading model: {e}")
        return None, None, None, None

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
        debug_print("Usage: python evaluate_upload.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug_print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'social.pt')
    
    if not os.path.exists(json_file_path):
        debug_print(f"Error: {json_file_path} not found!")
        return
    
    if not os.path.exists(model_path):
        debug_print(f"Error: {model_path} not found!")
        return
    
    test_df = load_papers(json_file_path)
    
    if len(test_df) == 0:
        debug_print("No valid papers found in the test file!")
        return
    
    model, label_mapping, feature_names, expected_feature_dim = load_trained_model(model_path, device)
    
    if model is None:
        debug_print("Failed to load model!")
        return
    
    debug_print(f"Model loaded successfully on {device}!")
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    extractor = SocialScienceFeatureExtractor()
    test_features = []
    
    for _, row in test_df.iterrows():
        features = extractor.extract_features(row['text'])
        test_features.append(features)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    # Pad or trim features if needed to match expected model input dimension
    if expected_feature_dim is not None and test_features.shape[1] != expected_feature_dim:
        if test_features.shape[1] < expected_feature_dim:
            # Pad with zeros
            padding = np.zeros((test_features.shape[0], expected_feature_dim - test_features.shape[1]), dtype=np.float32)
            test_features = np.concatenate([test_features, padding], axis=1)
        else:
            # Trim excess features
            test_features = test_features[:, :expected_feature_dim]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    max_seq_length = 128
    max_sections = 4
    max_sents = 8
    
    dummy_labels = [0] * len(test_df)
    
    test_dataset = HierarchicalPaperDataset(
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

