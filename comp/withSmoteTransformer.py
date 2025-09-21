import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import re
import json
from collections import Counter
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Focal Loss for handling extreme class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Enhanced dataset class to support hierarchical structure for CS papers
class HierarchicalCSPaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, handcrafted_features, 
                 max_seq_length=512, max_sections=6, max_sents=12):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.handcrafted_features = handcrafted_features
        self.max_seq_length = max_seq_length
        self.max_sections = max_sections
        self.max_sents = max_sents
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        features = self.handcrafted_features[idx]
        
        # Process text hierarchically
        sections = self._extract_sections(text)
        
        # Process each section
        token_ids = []
        attention_masks = []
        for section in sections[:self.max_sections]:
            # Tokenize sentences in section
            sentences = sent_tokenize(section)
            section_token_ids = []
            section_attention_masks = []
            
            for sent in sentences[:self.max_sents]:
                # Tokenize each sentence
                encoding = self.tokenizer(
                    sent,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                section_token_ids.append(encoding['input_ids'])
                section_attention_masks.append(encoding['attention_mask'])
            
            # Pad sentences to max_sents
            while len(section_token_ids) < self.max_sents:
                empty_encoding = self.tokenizer(
                    "",
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                section_token_ids.append(empty_encoding['input_ids'])
                section_attention_masks.append(empty_encoding['attention_mask'])
            
            # Stack sentence encodings
            token_ids.append(torch.cat(section_token_ids, dim=0))
            attention_masks.append(torch.cat(section_attention_masks, dim=0))
        
        # Pad sections to max_sections
        while len(token_ids) < self.max_sections:
            empty_encoding = self.tokenizer(
                [""] * self.max_sents,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            token_ids.append(empty_encoding['input_ids'])
            attention_masks.append(empty_encoding['attention_mask'])
        
        # Stack section encodings
        token_ids = torch.stack(token_ids)  # [max_sections, max_sents, max_seq_length]
        attention_masks = torch.stack(attention_masks)  # [max_sections, max_sents, max_seq_length]
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_masks,
            'handcrafted_features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _extract_sections(self, text):
        """Extract sections from CS paper text"""
        section_markers = [
            'abstract', 'introduction', 'related work', 'methodology', 
            'implementation', 'results', 'experiments', 'evaluation', 
            'discussion', 'conclusion', 'references'
        ]
        
        # Extract sections
        sections = []
        text_lower = text.lower()
        
        for i, marker in enumerate(section_markers):
            start_pos = text_lower.find('\n' + marker + '\n')
            if start_pos == -1:
                start_pos = text_lower.find(marker + '\n')
            
            if start_pos != -1:
                start_pos += len(marker) + 1
                
                # Find the next section marker
                end_pos = float('inf')
                for next_marker in section_markers[i+1:]:
                    next_pos = text_lower.find('\n' + next_marker + '\n', start_pos)
                    if next_pos == -1:
                        next_pos = text_lower.find(next_marker + '\n', start_pos)
                    
                    if next_pos != -1 and next_pos < end_pos:
                        end_pos = next_pos
                
                if end_pos == float('inf'):
                    end_pos = len(text)
                
                section_text = text[start_pos:end_pos].strip()
                sections.append(section_text)
        
        # If no sections were found, treat the entire text as one section
        if not sections:
            sections = [text]
        
        return sections

# Computer Science bias feature extractor
class CSBiasFeatureExtractor:
    """Extract features that might indicate bias in computer science research papers"""
    
    def __init__(self):
        # Basic patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.0\d+'
        self.significance_stars = r'\*{1,3}\s*'
        self.performance_pattern = r'(?:accuracy|precision|recall|f1|performance|improvement)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+[%]?'
        
        # CS-specific hedge words
        self.hedge_words = [
            'may', 'might', 'could', 'possibly', 'potentially', 'suggests', 
            'appears', 'seems', 'likely', 'unlikely', 'perhaps', 'arguably',
            'tend to', 'tends to', 'tended to', 'indicate', 'indicates',
            'approximately', 'roughly', 'around', 'about'
        ]
        
        # CS-specific certainty words
        self.certainty_words = [
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'conclusively', 'absolutely', 'always', 'never', 'established',
            'proves', 'demonstrates', 'robust', 'significant', 'substantial',
            'strong evidence', 'strongly supports', 'decisive', 'optimal',
            'best', 'superior', 'outperforms', 'state-of-the-art'
        ]
        
        # CS theory and method references
        self.theory_terms = [
            'machine learning', 'deep learning', 'neural network', 'algorithm',
            'optimization', 'theoretical analysis', 'computational complexity',
            'big o notation', 'time complexity', 'space complexity',
            'theoretical framework', 'mathematical model'
        ]
        
        # CS-specific claim words
        self.claim_terms = [
            'performance', 'accuracy', 'efficiency', 'scalability', 'robustness',
            'improvement', 'optimization', 'enhancement', 'breakthrough',
            'novel', 'innovative', 'superior', 'outperforms', 'beats',
            'faster', 'better', 'more accurate', 'highly effective'
        ]
        
        # CS jargon that might signal overconfidence
        self.cs_jargon = [
            'state-of-the-art', 'cutting-edge', 'novel approach', 'breakthrough',
            'paradigm shift', 'revolutionary', 'game-changing', 'unprecedented',
            'significantly outperforms', 'dramatically improves', 'orders of magnitude'
        ]
        
        # Experimental rigor terms
        self.rigor_terms = [
            'baseline', 'benchmark', 'cross-validation', 'ablation study',
            'statistical significance', 'error bars', 'confidence interval',
            'reproducible', 'replication', 'validation set', 'test set'
        ]
        
        # CS-specific method patterns
        self.cs_method_patterns = [
            r'CNN', r'RNN', r'LSTM', r'GRU', r'transformer', r'attention',
            r'SVM', r'random forest', r'gradient descent', r'backpropagation',
            r'reinforcement learning', r'supervised learning', r'unsupervised learning'
        ]
        
        # Stopwords for cleaning text
        self.stopwords = set(stopwords.words('english'))
        
    def extract_features(self, text, reason_text=None):
        features = {}
        
        # Handle None or empty text
        if text is None or not text:
            return [0] * 25  # Updated feature count
        
        # Extract sections 
        sections = self._extract_sections(text)
        
        # 1. Basic text statistics
        features['length'] = len(text)
        word_count = len(text.split()) + 1  # Add 1 to avoid division by zero
        features['avg_word_length'] = sum(len(w) for w in text.split()) / word_count
        
        # 2. Statistical reporting patterns
        features['p_value_count'] = len(re.findall(self.p_value_pattern, text, re.IGNORECASE))
        features['significance_stars_count'] = len(re.findall(self.significance_stars, text))
        features['performance_metrics_count'] = len(re.findall(self.performance_pattern, text, re.IGNORECASE))
        
        # 3. Linguistic features
        hedge_count = sum(text.lower().count(word) for word in self.hedge_words)
        certainty_count = sum(text.lower().count(word) for word in self.certainty_words)
        features['hedge_ratio'] = hedge_count / word_count * 1000
        features['certainty_ratio'] = certainty_count / word_count * 1000
        
        # 4. CS-specific patterns
        theory_count = sum(text.lower().count(term) for term in self.theory_terms)
        jargon_count = sum(text.lower().count(term) for term in self.cs_jargon)
        features['theory_term_ratio'] = theory_count / word_count * 1000
        features['jargon_term_ratio'] = jargon_count / word_count * 1000
        
        # 5. CS method patterns
        cs_method_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in self.cs_method_patterns)
        features['cs_method_count'] = cs_method_count
        
        # 6. Section-specific features
        # Abstract claims
        if 'abstract' in sections:
            abstract = sections['abstract']
            abstract_words = len(abstract.split()) + 1
            abstract_claim_count = sum(abstract.lower().count(term) for term in self.claim_terms)
            features['abstract_claim_ratio'] = abstract_claim_count / abstract_words * 1000
        else:
            features['abstract_claim_ratio'] = 0
        
        # Results section features
        if 'results' in sections or 'experiments' in sections:
            results = sections.get('results', sections.get('experiments', ''))
            results_perf_metrics = len(re.findall(self.performance_pattern, results, re.IGNORECASE))
            results_words = len(results.split()) + 1
            features['results_metric_density'] = results_perf_metrics / results_words * 1000
        else:
            features['results_metric_density'] = 0
        
        # 7. Limitations acknowledgment
        features['limitations_mentioned'] = 1 if "limitation" in text.lower() or "limitations" in text.lower() else 0
        
        # 8. Experimental rigor
        rigor_count = sum(text.lower().count(term) for term in self.rigor_terms)
        features['experimental_rigor_ratio'] = rigor_count / word_count * 1000
        
        # 9. Claim consistency (abstract vs conclusion)
        if 'abstract' in sections and 'conclusion' in sections:
            abstract = sections['abstract']
            conclusion = sections['conclusion']
            abstract_claims = self._extract_key_claims(abstract)
            conclusion_claims = self._extract_key_claims(conclusion)
            features['claim_consistency'] = self._compare_claims(abstract_claims, conclusion_claims)
        else:
            features['claim_consistency'] = 0
            
        # 10. Figure and table mentions
        features['figure_mentions'] = text.lower().count("figure") + text.lower().count("fig.")
        features['table_mentions'] = text.lower().count("table")
        
        # 11. Citation patterns
        features['citation_count'] = text.count("et al.") + text.count("[")
        features['self_citation_count'] = (text.lower().count("our previous") + 
                                        text.lower().count("our prior") + 
                                        text.lower().count("our earlier") +
                                        text.lower().count("our work"))
        
        # 12. Reason-based features (if available)
        if reason_text:
            reason_features = self._extract_reason_features(reason_text)
            features.update(reason_features)
        else:
            # Default reason features
            features['fake_peer_review'] = 0
            features['data_issues'] = 0
            features['paper_mill'] = 0
            features['authorship_issues'] = 0
            features['duplication_issues'] = 0
        
        # Return feature values as a list in consistent order
        feature_values = [
            features['length'], 
            features['avg_word_length'],
            features['p_value_count'], 
            features['significance_stars_count'],
            features['performance_metrics_count'],
            features['hedge_ratio'], 
            features['certainty_ratio'],
            features['theory_term_ratio'],
            features['jargon_term_ratio'],
            features['cs_method_count'],
            features['abstract_claim_ratio'],
            features['results_metric_density'],
            features['limitations_mentioned'],
            features['experimental_rigor_ratio'],
            features['claim_consistency'],
            features['figure_mentions'],
            features['table_mentions'],
            features['citation_count'],
            features['self_citation_count'],
            features['fake_peer_review'],
            features['data_issues'],
            features['paper_mill'],
            features['authorship_issues'],
            features['duplication_issues'],
            features.get('reproducibility_mentions', 0)
        ]
        
        return feature_values
    
    def _extract_reason_features(self, reason_text):
        """Extract features from reason field"""
        reason_features = {}
        
        if not reason_text:
            return {
                'fake_peer_review': 0,
                'data_issues': 0,
                'paper_mill': 0,
                'authorship_issues': 0,
                'duplication_issues': 0
            }
        
        reason_lower = reason_text.lower()
        
        # Fake peer review indicators
        reason_features['fake_peer_review'] = 1 if 'fake peer review' in reason_lower else 0
        
        # Data issues
        data_keywords = ['concerns/issues about data', 'concerns/issues about results', 'unreliable results']
        reason_features['data_issues'] = 1 if any(keyword in reason_lower for keyword in data_keywords) else 0
        
        # Paper mill indicators
        reason_features['paper_mill'] = 1 if 'paper mill' in reason_lower else 0
        
        # Authorship issues
        auth_keywords = ['concerns/issues about authorship', 'authorship/affiliation']
        reason_features['authorship_issues'] = 1 if any(keyword in reason_lower for keyword in auth_keywords) else 0
        
        # Duplication issues
        dup_keywords = ['duplication', 'euphemisms for duplication']
        reason_features['duplication_issues'] = 1 if any(keyword in reason_lower for keyword in dup_keywords) else 0
        
        return reason_features
    
    def _extract_sections(self, text):
        """Extract sections from CS paper text"""
        section_dict = {}
        section_markers = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', 'i. introduction'],
            'related_work': ['related work', 'literature review', 'background'],
            'methodology': ['method', 'methods', 'methodology', 'approach'],
            'implementation': ['implementation', 'system design', 'architecture'],
            'experiments': ['experiments', 'experimental setup', 'evaluation'],
            'results': ['results', 'findings', 'experimental results'],
            'discussion': ['discussion'],
            'conclusion': ['conclusion', 'conclusions', 'concluding remarks']
        }
        
        text_lower = text.lower()
        
        for section_key, markers in section_markers.items():
            for marker in markers:
                patterns = [
                    f'\n{marker}\n',
                    f'\n{marker}.\n',
                    f'\n{marker}:\n',
                    f'\n{marker.title()}\n',
                    f'\n{marker.upper()}\n'
                ]
                
                for pattern in patterns:
                    start_pos = text_lower.find(pattern)
                    if start_pos != -1:
                        start_pos += len(pattern) - 1
                        
                        # Find the next section
                        end_pos = float('inf')
                        for next_section_key, next_markers in section_markers.items():
                            if next_section_key != section_key:
                                for next_marker in next_markers:
                                    for next_pattern in patterns:
                                        next_start = text_lower.find(next_pattern, start_pos)
                                        if next_start != -1 and next_start < end_pos:
                                            end_pos = next_start
                        
                        if end_pos == float('inf'):
                            end_pos = len(text)
                        
                        section_dict[section_key] = text[start_pos:end_pos].strip()
                        break
                
                if section_key in section_dict:
                    break
        
        return section_dict
    
    def _extract_key_claims(self, text):
        """Extract key claims from text using CS-specific terms"""
        claim_sentences = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in self.claim_terms):
                claim_sentences.append(sentence)
        
        return claim_sentences
    
    def _compare_claims(self, claims1, claims2):
        """Compare claims for consistency"""
        if not claims1 or not claims2:
            return 0
        
        # Tokenize and remove stopwords
        words1 = []
        for claim in claims1:
            words1.extend([w.lower() for w in word_tokenize(claim) 
                           if w.lower() not in self.stopwords and w.isalpha()])
        
        words2 = []
        for claim in claims2:
            words2.extend([w.lower() for w in word_tokenize(claim) 
                           if w.lower() not in self.stopwords and w.isalpha()])
        
        # Calculate overlap
        common_words = set(words1).intersection(set(words2))
        all_words = set(words1).union(set(words2))
        
        if not all_words:
            return 0
            
        return len(common_words) / len(all_words)

# Feature fusion layer (same as economics model)
class FeatureFusionLayer(nn.Module):
    def __init__(self, bert_dim, handcrafted_dim, fusion_dim=256):
        super(FeatureFusionLayer, self).__init__()
        self.bert_projection = nn.Linear(bert_dim, fusion_dim)
        self.handcrafted_projection = nn.Linear(handcrafted_dim, fusion_dim)
        
        # Attention mechanism for dynamic fusion
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, bert_embedding, handcrafted_features):
        # Project both feature types to the same dimension
        bert_proj = self.bert_projection(bert_embedding)
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)
        
        # Ensure correct dimensions for attention (seq_len, batch_size, embed_dim)
        # bert_proj and handcrafted_proj should be [batch_size, embed_dim]
        if bert_proj.dim() == 2:
            # Reshape to [1, batch_size, embed_dim] for attention
            bert_proj_attn = bert_proj.unsqueeze(0)  # [1, batch_size, embed_dim]
            handcrafted_proj_attn = handcrafted_proj.unsqueeze(0)  # [1, batch_size, embed_dim]
        else:
            # Handle unexpected dimensions
            bert_proj_attn = bert_proj.view(1, -1, bert_proj.size(-1))
            handcrafted_proj_attn = handcrafted_proj.view(1, -1, handcrafted_proj.size(-1))
        
        # Cross-attention between features
        attn_output, _ = self.attention(bert_proj_attn, handcrafted_proj_attn, handcrafted_proj_attn)
        attn_output = attn_output.squeeze(0)  # Remove seq_len dimension
        
        # Calculate gate values for feature importance
        combined = torch.cat([bert_proj, handcrafted_proj], dim=-1)
        gate_values = self.gate(combined)
        
        # Gated fusion
        fused = gate_values * bert_proj + (1 - gate_values) * handcrafted_proj
        
        # Normalization and dropout
        output = self.layer_norm(fused)
        output = self.dropout(output)
        
        return output

# Hierarchical attention (same as economics model)
class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(HierarchicalAttention, self).__init__()
        
        # Word-level attention
        self.word_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.word_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Sentence-level attention
        self.sentence_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.sentence_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Section-level attention
        self.section_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.section_layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, word_embeddings, attention_mask):
        batch_size, max_sections, max_sents, max_words, hidden_dim = word_embeddings.shape
        doc_embeddings = []

        for b in range(batch_size):
            section_embeddings = []
            for s in range(max_sections):
                sentence_embeddings = []
                for i in range(max_sents):
                    words = word_embeddings[b, s, i]
                    mask = attention_mask[b, s, i]

                    if mask.sum() > 0:
                        key_padding_mask = (mask == 0).unsqueeze(0)
                        words = words.unsqueeze(1)
                        words = words.transpose(0, 1)
                        words = words.transpose(0, 1)

                        attn_output, _ = self.word_attention(
                            words, words, words,
                            key_padding_mask=key_padding_mask
                        )
                        attn_output = attn_output.squeeze(1)
                        sent_embedding = (attn_output * mask.unsqueeze(-1)).sum(dim=0) / (mask.sum() + 1e-10)
                    else:
                        sent_embedding = torch.zeros(hidden_dim, device=word_embeddings.device)
                    sentence_embeddings.append(sent_embedding)

                if sentence_embeddings:
                    section_sent_embeddings = torch.stack(sentence_embeddings)
                    sent_mask = (attention_mask[b, s].sum(dim=1) > 0).float()

                    if sent_mask.sum() > 0:
                        key_padding_mask = (sent_mask == 0).unsqueeze(0)
                        section_sent_embeddings = section_sent_embeddings.unsqueeze(1)
                        section_sent_embeddings = section_sent_embeddings.transpose(0, 1)
                        section_sent_embeddings = section_sent_embeddings.transpose(0, 1)

                        attn_output, _ = self.sentence_attention(
                            section_sent_embeddings, section_sent_embeddings, section_sent_embeddings,
                            key_padding_mask=key_padding_mask
                        )
                        attn_output = attn_output.squeeze(1)
                        section_embedding = (attn_output * sent_mask.unsqueeze(-1)).sum(dim=0) / (sent_mask.sum() + 1e-10)
                    else:
                        section_embedding = torch.zeros(hidden_dim, device=word_embeddings.device)
                    section_embeddings.append(section_embedding)
                else:
                    section_embeddings.append(torch.zeros(hidden_dim, device=word_embeddings.device))

            if section_embeddings:
                doc_section_embeddings = torch.stack(section_embeddings)
                section_mask = (attention_mask[b].sum(dim=(1, 2)) > 0).float()

                if section_mask.sum() > 0:
                    key_padding_mask = (section_mask == 0).unsqueeze(0)
                    doc_section_embeddings = doc_section_embeddings.unsqueeze(1)
                    doc_section_embeddings = doc_section_embeddings.transpose(0, 1)
                    doc_section_embeddings = doc_section_embeddings.transpose(0, 1)

                    attn_output, _ = self.section_attention(
                        doc_section_embeddings, doc_section_embeddings, doc_section_embeddings,
                        key_padding_mask=key_padding_mask
                    )
                    attn_output = attn_output.squeeze(1)
                    doc_embedding = (attn_output * section_mask.unsqueeze(-1)).sum(dim=0) / (section_mask.sum() + 1e-10)
                else:
                    doc_embedding = torch.zeros(hidden_dim, device=word_embeddings.device)
            else:
                doc_embedding = torch.zeros(hidden_dim, device=word_embeddings.device)

            doc_embeddings.append(doc_embedding)

        return torch.stack(doc_embeddings)

# CS Bias Prediction Model
class CSBiasPredictionModel(nn.Module):
    def __init__(self, bert_model_name, num_classes=3, dropout_rate=0.3, feature_dim=25):
        # Force num_classes to be exactly 3 for CS bias detection
        if num_classes != 3:
            print(f"WARNING: num_classes was {num_classes}, forcing to 3 for CS bias detection")
            num_classes = 3
        super(CSBiasPredictionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dimensions
        self.bert_dim = self.bert.config.hidden_size
        self.handcrafted_dim = feature_dim
        self.fusion_dim = 256
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(hidden_dim=self.bert_dim)
        
        # Feature fusion layer
        self.feature_fusion = FeatureFusionLayer(
            bert_dim=self.bert_dim, 
            handcrafted_dim=self.handcrafted_dim,
            fusion_dim=self.fusion_dim
        )
        
        # Enhanced classification layers with residual connections
        self.fc1 = nn.Linear(self.fusion_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # Better activation for transformers
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(128)
        
        # Residual connection for fusion dimension matching
        self.residual_proj = nn.Linear(self.fusion_dim, 128) if self.fusion_dim != 128 else nn.Identity()
        
    def forward(self, input_ids, attention_mask, handcrafted_features):
        batch_size = input_ids.size(0)
        
        # Reshape for BERT processing
        max_sections, max_sents, max_seq_length = input_ids.size(1), input_ids.size(2), input_ids.size(3)
        
        # Process each sentence with BERT
        word_embeddings = []
        for b in range(batch_size):
            section_embeddings = []
            for s in range(max_sections):
                sentence_embeddings = []
                for i in range(max_sents):
                    # Process single sentence through BERT
                    input_ids_sent = input_ids[b, s, i].unsqueeze(0)
                    attention_mask_sent = attention_mask[b, s, i].unsqueeze(0)
                    
                    # Skip processing empty sentences to save computation
                    if attention_mask_sent.sum() > 0:
                        outputs = self.bert(
                            input_ids=input_ids_sent, 
                            attention_mask=attention_mask_sent,
                            output_hidden_states=True
                        )
                        
                        # Get token embeddings from last layer
                        token_embeddings = outputs.last_hidden_state[0]
                    else:
                        # Create zero embeddings for empty sentences
                        token_embeddings = torch.zeros(
                            max_seq_length, self.bert_dim, device=input_ids.device)
                    
                    sentence_embeddings.append(token_embeddings)
                
                # Stack to get all sentence embeddings for this section
                section_embeddings.append(torch.stack(sentence_embeddings))
            
            # Stack to get all section embeddings for this document
            word_embeddings.append(torch.stack(section_embeddings))
        
        # Stack to get embeddings for the entire batch
        word_embeddings = torch.stack(word_embeddings)
        
        # Apply hierarchical attention
        doc_embeddings = []
        for b in range(batch_size):
            doc_embedding = self.hierarchical_attention(
                word_embeddings[b].unsqueeze(0),
                attention_mask[b].unsqueeze(0)
            )
            # Ensure doc_embedding is 1D (hidden_dim,)
            if doc_embedding.dim() > 1:
                doc_embedding = doc_embedding.squeeze()
            doc_embeddings.append(doc_embedding)
        
        # Stack to create [batch_size, hidden_dim]
        doc_embeddings = torch.stack(doc_embeddings)
        
        # Feature fusion
        fused_features = self.feature_fusion(doc_embeddings, handcrafted_features)
        
        # Enhanced classification with residual connections
        x1 = self.fc1(fused_features)
        x1 = self.gelu(x1)  # GELU works better with transformers
        x1 = self.layer_norm1(x1)
        x1 = self.dropout(x1)
        
        x2 = self.fc2(x1)
        x2 = self.gelu(x2)
        x2 = self.layer_norm2(x2)
        
        # Add residual connection
        residual = self.residual_proj(fused_features)
        x2 = x2 + residual
        
        x2 = self.dropout(x2)
        output = self.fc3(x2)
        
        return output

# Training function with gradient accumulation and class weighting
def train_cs_model(train_dataloader, val_dataloader, model, device, 
                   epochs=3, lr=4e-5, accumulation_steps=2, class_weights=None):
    """Train the CS bias prediction model with enhanced imbalance handling"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Optimized learning schedule for 3 epochs maximum
    warmup_steps = len(train_dataloader) // 4  # Shorter warmup for 3 epochs
    total_steps = epochs * len(train_dataloader)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Faster linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # More aggressive cosine annealing for 3 epochs
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))  # Don't go below 10% of max LR
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Use CrossEntropyLoss with class weights and label smoothing
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        print(f"Using CrossEntropyLoss with class weights and label smoothing (0.1)")
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_accuracy = 0
    best_model = None
    patience = 1  # Minimal patience for 3 epochs maximum
    patience_counter = 0
    min_improvement = 0.005  # Higher improvement threshold for aggressive training
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Aggressive memory management for GTX 1650
            if torch.cuda.is_available():
                if batch_idx % 3 == 0:  # Clear cache every 3 batches
                    torch.cuda.empty_cache()
                if batch_idx % 10 == 0:  # Synchronize every 10 batches
                    torch.cuda.synchronize()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            try:
                outputs = model(input_ids, attention_mask, handcrafted_features)
                loss = loss_fn(outputs, labels)
                loss = loss / accumulation_steps
                
                # Check for NaN/Inf before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Invalid loss detected at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                train_loss += loss.item() * accumulation_steps
                
                # Free intermediate tensors to save memory
                del outputs, loss
                
                # Clear GPU cache more aggressively
                if torch.cuda.is_available() and batch_idx % 2 == 0:
                    torch.cuda.empty_cache()
                
                # Accumulate gradients and update at intervals
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Clear cache after gradient updates
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}. Clearing cache and skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
            
            # More frequent progress reporting for debugging slow training
            if batch_idx % 5 == 0:  # Report every 5 batches instead of 20
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"Batch {batch_idx}/{len(train_dataloader)}: GPU Memory - Used: {memory_used:.2f}GB, Cached: {memory_cached:.2f}GB")
                else:
                    print(f"Batch {batch_idx}/{len(train_dataloader)}: Processing...")
        
        # Handle any remaining gradients at the end of epoch
        if len(train_dataloader) % accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
        
        train_loss /= len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_dataloader):
                # Memory management during validation
                if torch.cuda.is_available() and val_batch_idx % 3 == 0:
                    torch.cuda.empty_cache()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                handcrafted_features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)
                
                try:
                    outputs = model(input_ids, attention_mask, handcrafted_features)
                    loss = loss_fn(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Free memory
                    del outputs, loss, predicted
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM during validation batch {val_batch_idx}. Skipping...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        
        # Calculate per-class metrics for better monitoring (fixed to 3 classes)
        class_correct = torch.zeros(3).to(device)  # Fixed to 3 classes
        class_total = torch.zeros(3).to(device)    # Fixed to 3 classes
        
        with torch.no_grad():
            for pc_batch_idx, batch in enumerate(val_dataloader):
                # Memory management for per-class calculation
                if torch.cuda.is_available() and pc_batch_idx % 2 == 0:
                    torch.cuda.empty_cache()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                handcrafted_features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)
                
                try:
                    outputs = model(input_ids, attention_mask, handcrafted_features)
                    _, predicted = torch.max(outputs, 1)
                    
                    for i in range(len(labels)):
                        label = labels[i]
                        class_total[label] += 1
                        if predicted[i] == label:
                            class_correct[label] += 1
                    
                    # Free memory
                    del outputs, predicted
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM during per-class calculation batch {pc_batch_idx}. Skipping...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Calculate per-class accuracy (avoid division by zero)
        class_accuracies = []
        for i in range(len(class_correct)):
            if class_total[i] > 0:
                acc = (class_correct[i] / class_total[i]).item()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0.0)
        
        # Use balanced accuracy (mean of per-class accuracies) as main metric
        balanced_accuracy = np.mean(class_accuracies)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss = {train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}, '
              f'Balanced Acc = {balanced_accuracy:.4f}, '
              f'LR = {scheduler.get_last_lr()[0]:.6f}')
        print(f'Per-class accuracies: {[f"{acc:.3f}" for acc in class_accuracies]}')
        
        # Check if model is stuck predicting only one class
        non_zero_classes = sum(1 for acc in class_accuracies if acc > 0.001)
        print(f'Classes with non-zero accuracy: {non_zero_classes}/{len(class_accuracies)}')
        
        # Save model based on balanced accuracy (better for imbalanced datasets)
        # Relax condition to allow single-class models initially
        if balanced_accuracy > best_val_accuracy + min_improvement:
            improvement = balanced_accuracy - best_val_accuracy
            best_val_accuracy = balanced_accuracy
            best_model = model.state_dict().copy()
            torch.save(best_model, 'best_cs_bias_model.pt')
            print(f'New best model saved with balanced accuracy: {balanced_accuracy:.4f} (improvement: +{improvement:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1
            if non_zero_classes <= 1:
                print(f'WARNING: Model is only predicting {non_zero_classes} class(es)! This indicates severe overfitting to majority class.')
            print(f'No significant improvement for {patience_counter} epochs (current: {balanced_accuracy:.4f}, best: {best_val_accuracy:.4f})')
            
        # Early stopping based on patience
        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break
    
    # Load best model (handle case where no model was saved)
    if best_model is not None:
        model.load_state_dict(best_model)
        print("Loaded best model from training")
    else:
        print("No best model was saved (no improvement detected), using final model state")
    return model

# Load CS papers from JSON
def load_cs_papers_from_json(json_file_path):
    """Load CS research papers from JSON file with dynamic bias mapping"""
    
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found.")
        return pd.DataFrame(columns=['text', 'label', 'reason']), {}
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        bias_types = set()
        
        for paper in data:
            if not isinstance(paper, dict):
                continue
            
            # Extract text (Body field)
            text = paper.get('Body', '')
            if not text:
                continue
            
            # Extract bias label (handle both "Overall Bias" and "OverallBias" field names)
            bias_label = paper.get('Overall Bias') or paper.get('OverallBias')
            if not bias_label:
                print(f"Warning: Paper missing both 'Overall Bias' and 'OverallBias' fields: {paper.get('Title', 'Unknown')}")
                continue
            bias_types.add(bias_label)
            
            # Extract reason
            reason = paper.get('Reason', '')
            
            papers.append({
                'text': text,
                'bias_label': bias_label,
                'reason': reason
            })
        
        # Use fixed label mapping for the 3 expected classes
        label_mapping = {
            'No Bias': 0,
            'Cognitive Bias': 1, 
            'Publication Bias': 2
        }
        
        # Filter out any unexpected labels
        expected_labels = set(label_mapping.keys())
        unique_bias_types = sorted(list(bias_types.intersection(expected_labels)))
        
        print(f"Expected labels: {expected_labels}")
        print(f"Found labels: {bias_types}")
        
        # Warn about unexpected labels
        unexpected_labels = bias_types - expected_labels
        if unexpected_labels:
            print(f"WARNING: Found unexpected labels that will be filtered out: {unexpected_labels}")
        
        # Convert to DataFrame with numeric labels
        df = pd.DataFrame(papers)
        df['label'] = df['bias_label'].map(label_mapping)
        
        # Remove papers with unmapped labels (NaN values)
        original_count = len(df)
        df = df.dropna(subset=['label'])
        filtered_count = len(df)
        
        if original_count != filtered_count:
            print(f"Filtered out {original_count - filtered_count} papers with unexpected labels")
        
        # Convert labels to integers
        df['label'] = df['label'].astype(int)
        
        print(f"Successfully loaded {len(df)} CS papers from {json_file_path}")
        print(f"Bias types found: {unique_bias_types}")
        print(f"Label mapping: {label_mapping}")
        
        # Print class distribution
        print("Full dataset class distribution:")
        print(df['bias_label'].value_counts())
        
        return df, label_mapping
        
    except Exception as e:
        print(f"Error loading papers from {json_file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label', 'reason']), {}

def main():
    """
    CS Bias Detection Training - Single Dataset Approach
    
    Uses computer_science_papers.json as main dataset with 80/20 train/validation split.
    No separate test files needed - validation accuracy shows true performance.
    
    IMPORTANT: All data processing happens in-memory only. Original files never changed.
    
    Optimizations for 3 epochs:
    - Single dataset with clean train/validation split
    - Enhanced learning rate scheduling with warmup + cosine annealing
    - Improved model architecture with residual connections and GELU activation
    - SMOTE balancing (in-memory only, never saves to files)
    - Optimized batch size and gradient accumulation
    """
    # Set up device with memory management for GTX 1650
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
        
        # Clear cache and set memory management
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        
        # Enable memory-efficient settings for GTX 1650
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
    
    # Feature names for the CS-specific extractor
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'significance_stars_count',
        'performance_metrics_count', 'hedge_ratio', 'certainty_ratio', 'theory_term_ratio',
        'jargon_term_ratio', 'cs_method_count', 'abstract_claim_ratio',
        'results_metric_density', 'limitations_mentioned', 'experimental_rigor_ratio',
        'claim_consistency', 'figure_mentions', 'table_mentions', 'citation_count', 
        'self_citation_count', 'fake_peer_review', 'data_issues', 'paper_mill',
        'authorship_issues', 'duplication_issues', 'reproducibility_mentions'
    ]
    
    # Load and process CS paper data from main dataset
    papers_df, label_mapping = load_cs_papers_from_json('computer_science_papers.json')
    
    if len(papers_df) == 0:
        print("Error: No papers loaded. Please check the data file.")
        return
    
    # Extract handcrafted features with CS-specific extractor
    print("Extracting CS-specific handcrafted features...")
    extractor = CSBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'], row['reason'])
        handcrafted_features.append(features)
    
    # Convert to numpy arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Split into train and validation from main dataset
    # Use 80/20 split for train/validation (no separate test file needed)
    train_texts, val_texts, train_features, val_features, train_labels, val_labels = train_test_split(
        papers_df['text'].tolist(), features_array, labels_array,
        test_size=0.2, random_state=42, stratify=labels_array
    )
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    
    # Show training set distribution before any SMOTE
    print("\n" + "="*60)
    print("📊 TRAINING SET DISTRIBUTION (80% of main dataset):")
    print("="*60)
    train_counts = Counter(train_labels)
    train_total = len(train_labels)
    for label_idx in sorted(train_counts.keys()):
        count = train_counts[label_idx]
        percentage = (count / train_total) * 100
        label_names = {0: 'No Bias', 1: 'Cognitive Bias', 2: 'Publication Bias'}
        label_name = label_names.get(label_idx, f'Class {label_idx}')
        print(f"  {label_name:<18}: {count:>5} samples ({percentage:>5.1f}%)")
    print("="*60)
    
    # Check if SMOTE should be applied for class balancing
    print("Checking class balancing strategy...")
    try:
        # Get original class distribution
        original_counts = Counter(train_labels)
        print("\n" + "="*60)
        print("📊 REAL CLASS DISTRIBUTION (Before any correction):")
        print("="*60)
        total_samples = sum(original_counts.values())
        for label_idx in sorted(original_counts.keys()):
            count = original_counts[label_idx]
            percentage = (count / total_samples) * 100
            # Map back to label names for clarity
            label_names = {0: 'No Bias', 1: 'Cognitive Bias', 2: 'Publication Bias'}
            label_name = label_names.get(label_idx, f'Class {label_idx}')
            print(f"  {label_name:<18}: {count:>5} samples ({percentage:>5.1f}%)")
        print("="*60)
        
        # Calculate balanced class weights (less extreme than inverse frequency)
        total_samples = sum(original_counts.values())
        num_classes = len(original_counts)
        class_weights = []
        
        # Use square root of inverse frequency for less extreme weights
        for i in range(num_classes):
            if i in original_counts:
                # Balanced weighting based on actual data distribution
                raw_weight = total_samples / (num_classes * original_counts[i])
                
                # Aggressive weighting for 85% accuracy target in 3 epochs
                if original_counts[i] < 300:  # Cognitive Bias (smallest - 5.1%)
                    balanced_weight = min(raw_weight * 2.5, 25.0)  # Strong boost for smallest
                elif original_counts[i] < 600:  # No Bias (medium - 8.2%)
                    balanced_weight = min(raw_weight * 1.8, 12.0)  # Good boost for medium
                else:  # Publication Bias (majority - 86.7%)
                    balanced_weight = min(raw_weight ** 0.7, 3.0)  # Reasonable weight for majority
                class_weights.append(balanced_weight)
            else:
                class_weights.append(1.0)
        
        print(f"Calculated class weights: {[f'{w:.2f}' for w in class_weights]}")
        
        # Check if we have enough samples for SMOTE
        min_samples_per_class = min(original_counts.values())
        
        # Disable SMOTE to learn real data distribution
        use_smote = False  # SMOTE was over-correcting the imbalance
        
        if use_smote and min_samples_per_class >= 6:
            # Very aggressive SMOTE for 85% accuracy target
            max_samples = max(original_counts.values())
            target_samples = min(max_samples, 1000)  # Higher target for 85% accuracy
            
            # Create aggressive sampling strategy for 3-epoch training
            sampling_strategy = {}
            for label, count in original_counts.items():
                if count < 800:  # Boost all underrepresented classes
                    if count < 200:  # Cognitive Bias (154 samples)
                        sampling_strategy[label] = min(target_samples, count * 5)  # 5x boost for smallest
                    elif count < 500:  # No Bias if needed
                        sampling_strategy[label] = min(target_samples, count * 3)  # 3x boost for medium
                    else:
                        sampling_strategy[label] = min(target_samples, int(count * 1.5))  # 1.5x boost
            
            print(f"SMOTE sampling strategy: {sampling_strategy}")
            
            if sampling_strategy:  # Only apply SMOTE if needed
                k_neighbors = min(5, min_samples_per_class - 1)
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
                resampled_features, resampled_labels = smote.fit_resample(train_features, train_labels)
            else:
                resampled_features, resampled_labels = train_features, train_labels
            
            # Update training data with resampled features only (keep original texts)
            # NEVER modify the original dataset files - only work with in-memory data
            train_features = resampled_features
            train_labels = resampled_labels
            
            # For synthetic samples, duplicate existing texts from same class
            if len(resampled_labels) > len(train_texts):
                # Need to extend texts for synthetic samples
                extended_texts = list(train_texts)  # Copy original texts
                label_to_texts = {}
                for i, (text, label) in enumerate(zip(train_texts, train_labels[:len(train_texts)])):
                    if label not in label_to_texts:
                        label_to_texts[label] = []
                    label_to_texts[label].append(text)
                
                # Add texts for synthetic samples
                for i in range(len(train_texts), len(resampled_labels)):
                    label = resampled_labels[i]
                    if label in label_to_texts and label_to_texts[label]:
                        extended_texts.append(np.random.choice(label_to_texts[label]))
                    else:
                        # Fallback: use first text from training set
                        extended_texts.append(train_texts[0])
                
                train_texts = extended_texts
            
            print("\n" + "="*60)
            print("📊 CORRECTED CLASS DISTRIBUTION (After SMOTE):")
            print("="*60)
            corrected_counts = Counter(resampled_labels)
            corrected_total = sum(corrected_counts.values())
            for label_idx in sorted(corrected_counts.keys()):
                count = corrected_counts[label_idx]
                percentage = (count / corrected_total) * 100
                label_names = {0: 'No Bias', 1: 'Cognitive Bias', 2: 'Publication Bias'}
                label_name = label_names.get(label_idx, f'Class {label_idx}')
                print(f"  {label_name:<18}: {count:>5} samples ({percentage:>5.1f}%)")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("📊 NO CORRECTION APPLIED - Using real distribution:")
            print("="*60)
            if not use_smote:
                print("SMOTE disabled. Training on original class distribution.")
                print("Relying on class weights only to handle imbalance.")
            else:
                print(f"Not enough samples per class (min: {min_samples_per_class}). Skipping SMOTE.")
            print("="*60)
            # Still use class weights even without SMOTE
            
    except Exception as e:
        print(f"Error in SMOTE process: {e}")
        print("Proceeding with original imbalanced data")
        # Calculate class weights even if SMOTE fails
        original_counts = Counter(train_labels)
        total_samples = sum(original_counts.values())
        num_classes = len(original_counts)
        class_weights = []
        for i in range(num_classes):
            if i in original_counts:
                weight = total_samples / (num_classes * original_counts[i])
                class_weights.append(weight)
            else:
                class_weights.append(1.0)
    
    # Setup tokenizer - use bert-base-uncased like economics model
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create hierarchical datasets optimized for GTX 1650
    print("Creating hierarchical datasets optimized for GTX 1650...")
    
    # Much more aggressive reduction for faster training
    max_seq_length = 64   # Further reduced from 96
    max_sections = 3      # Further reduced from 4 
    max_sents = 4         # Further reduced from 6
    
    train_dataset = HierarchicalCSPaperDataset(
        train_texts, train_labels, tokenizer, train_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    val_dataset = HierarchicalCSPaperDataset(
        val_texts, val_labels, tokenizer, val_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Much smaller batch size to prevent 24-hour training issues
    if torch.cuda.is_available():
        # Use batch size 1 for GTX 1650 memory constraints
        batch_size = 1
        print("Using batch size of 1 to prevent memory issues and long training times")
    else:
        batch_size = 1
    
    if batch_size == 0:
        batch_size = 1
    print(f"Using batch size: {batch_size}")
    print(f"Model parameters: seq_len={max_seq_length}, sections={max_sections}, sents={max_sents}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize CS bias prediction model
    print("Initializing CS bias prediction model...")
    num_classes = 3  # Fixed: No Bias, Cognitive Bias, Publication Bias
    print(f"Model configured for {num_classes} classes")
    
    # Use bert-base-uncased like the successful economics model
    model = CSBiasPredictionModel(
        'bert-base-uncased', 
        num_classes=num_classes,
        feature_dim=len(feature_names)
    )
        
    model.to(device)
    
    # Train model with class weights for imbalance handling
    print("Training CS bias prediction model optimized for GTX 1650...")
    
    # Higher gradient accumulation to compensate for smaller batch size
    accumulation_steps = 8 if torch.cuda.is_available() else 4
    print(f"Using gradient accumulation steps: {accumulation_steps} (to compensate for batch_size=1)")
    
    model = train_cs_model(
        train_dataloader, val_dataloader, model, device, 
        epochs=3,  # Maximum 3 epochs due to CUDA constraints
        lr=4e-5,  # Higher learning rate for faster convergence in 3 epochs
        accumulation_steps=accumulation_steps,
        class_weights=class_weights
    )
    
    # Save the final model and metadata
    model_info = {
        'model_state_dict': model.state_dict(),
        'label_mapping': label_mapping,
        'feature_names': feature_names,
        'num_classes': num_classes
    }
    
    torch.save(model_info, 'cs_bias_model_complete.pt')
    
    # Save label mapping separately for easy access
    with open('cs_label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print("Training completed!")
    print(f"Model saved as 'cs_bias_model_complete.pt'")
    print(f"Label mapping saved as 'cs_label_mapping.json'")
    print(f"Label mapping: {label_mapping}")

if __name__ == "__main__":
    main()
