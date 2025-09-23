import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gc
import psutil

# Download NLTK resources if needed
nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Focal Loss for handling class imbalance and hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing for better generalization on large dataset
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            # Convert to one-hot for smooth targets
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = -(smooth_targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Enhanced dataset class to support hierarchical structure
class HierarchicalBusinessPaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, handcrafted_features, 
                 max_seq_length=512, max_sections=8, max_sents=16):
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
        """Extract sections from the business paper text"""
        # Business-specific section markers
        section_markers = [
            'abstract', 'executive summary', 'introduction', 'literature review', 
            'background', 'methodology', 'analysis', 'results', 'findings', 
            'discussion', 'implications', 'conclusion', 'recommendations', 'references'
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

# Enhanced feature extractor for business papers
class BusinessBiasFeatureExtractor:
    """Extract features that might indicate bias in business research papers"""
    
    def __init__(self):
        # Basic patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.0\d+'
        self.significance_stars = r'\*{1,3}\s*'
        self.correlation_pattern = r'(?:correlation|corr|r)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+'
        self.percentage_pattern = r'\d+(?:\.\d+)?%'
        
        # Business-specific hedge words
        self.hedge_words = [
            'may', 'might', 'could', 'possibly', 'potentially', 'suggests', 
            'appears', 'seems', 'likely', 'unlikely', 'perhaps', 'arguably',
            'tend to', 'tends to', 'tended to', 'indicate', 'indicates',
            'presumably', 'probably', 'apparently', 'supposedly'
        ]
        
        # Business-specific certainty words
        self.certainty_words = [
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'conclusively', 'absolutely', 'always', 'never', 'established',
            'proves', 'demonstrates', 'significant', 'substantial',
            'strong evidence', 'strongly supports', 'decisive', 'confirmed',
            'guaranteed', 'ensures', 'will result in', 'leads to'
        ]
        
        # Business theory and framework references
        self.theory_terms = [
            'porter', 'swot', 'balanced scorecard', 'lean', 'six sigma',
            'theory suggests', 'framework indicates', 'model predicts',
            'strategic management', 'organizational behavior', 'marketing theory',
            'financial theory', 'management theory', 'business model'
        ]
        
        # Business-specific claim words
        self.claim_terms = [
            'improve', 'increase', 'decrease', 'enhance', 'optimize', 'maximize',
            'minimize', 'performance', 'efficiency', 'productivity', 'profitability',
            'growth', 'revenue', 'cost reduction', 'competitive advantage',
            'market share', 'customer satisfaction', 'roi', 'return on investment'
        ]
        
        # Business jargon that might signal overconfidence
        self.business_jargon = [
            'synergies', 'leverage', 'paradigm shift', 'best practices',
            'core competencies', 'strategic alignment', 'value proposition',
            'scalable', 'disruptive', 'innovative', 'cutting-edge',
            'game-changer', 'transformational', 'revolutionary'
        ]
        
        # Methodology and validation terms
        self.method_terms = [
            'survey', 'interview', 'case study', 'experiment', 'observation',
            'questionnaire', 'focus group', 'analysis', 'regression',
            'correlation', 'statistical analysis', 'data analysis'
        ]
        
        # Business-specific validation patterns
        self.validation_patterns = [
            r'sample size', r'response rate', r'confidence interval',
            r'margin of error', r'validity', r'reliability',
            r'cronbach', r'factor analysis', r'regression analysis'
        ]
        
        # Stopwords for cleaning text
        self.stopwords = set(stopwords.words('english'))
        
    def extract_features(self, text):
        features = {}
        
        # Handle None or empty text
        if text is None or not text:
            # Return default values (22 features)
            return [0] * 22
        
        # Extract sections
        sections = self._extract_sections(text)
        
        # 1. Basic text statistics
        features['length'] = len(text)
        word_count = len(text.split()) + 1  # Add 1 to avoid division by zero
        features['avg_word_length'] = sum(len(w) for w in text.split()) / word_count
        
        # 2. Statistical reporting patterns
        features['p_value_count'] = len(re.findall(self.p_value_pattern, text))
        features['signif_stars_count'] = len(re.findall(self.significance_stars, text))
        features['correlation_count'] = len(re.findall(self.correlation_pattern, text))
        features['percentage_count'] = len(re.findall(self.percentage_pattern, text))
        
        # 3. Linguistic features
        # Hedging and certainty
        hedge_count = sum(text.lower().count(word) for word in self.hedge_words)
        certainty_count = sum(text.lower().count(word) for word in self.certainty_words)
        features['hedge_ratio'] = hedge_count / word_count * 1000  # per 1000 words
        features['certainty_ratio'] = certainty_count / word_count * 1000  # per 1000 words
        
        # 4. Business-specific patterns
        theory_count = sum(text.lower().count(term) for term in self.theory_terms)
        jargon_count = sum(text.lower().count(term) for term in self.business_jargon)
        features['theory_term_ratio'] = theory_count / word_count * 1000
        features['jargon_term_ratio'] = jargon_count / word_count * 1000
        
        # 5. Methodology patterns
        method_count = sum(text.lower().count(term) for term in self.method_terms)
        validation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in self.validation_patterns)
        features['method_term_count'] = method_count
        features['validation_pattern_count'] = validation_count
        
        # 6. Section-specific features
        # Abstract features (claims without evidence)
        if 'abstract' in sections:
            abstract = sections['abstract']
            abstract_words = len(abstract.split()) + 1
            abstract_claim_count = sum(abstract.lower().count(term) for term in self.claim_terms)
            features['abstract_claim_ratio'] = abstract_claim_count / abstract_words * 1000
        else:
            features['abstract_claim_ratio'] = 0
        
        # Results section features
        if 'results' in sections or 'findings' in sections:
            results_section = sections.get('results', sections.get('findings', ''))
            results_p_values = len(re.findall(self.p_value_pattern, results_section))
            results_words = len(results_section.split()) + 1
            features['results_stat_density'] = results_p_values / results_words * 1000
        else:
            features['results_stat_density'] = 0
        
        # Limitations acknowledgment
        features['limitations_mentioned'] = 1 if "limitation" in text.lower() or "limitations" in text.lower() else 0
        
        # 7. Business performance claims
        performance_terms = ['performance', 'improvement', 'success', 'effective', 'efficiency']
        performance_count = sum(text.lower().count(term) for term in performance_terms)
        features['performance_claim_ratio'] = performance_count / word_count * 1000
        
        # 8. Abstract vs conclusion claim consistency
        if 'abstract' in sections and ('conclusion' in sections or 'implications' in sections):
            abstract = sections['abstract']
            conclusion = sections.get('conclusion', sections.get('implications', ''))
            abstract_claims = self._extract_key_claims(abstract)
            conclusion_claims = self._extract_key_claims(conclusion)
            features['claim_consistency'] = self._compare_claims(abstract_claims, conclusion_claims)
        else:
            features['claim_consistency'] = 0
            
        # 9. Visual aids mentions
        features['figure_mentions'] = text.lower().count("figure") + text.lower().count("fig.")
        features['table_mentions'] = text.lower().count("table")
        features['chart_mentions'] = text.lower().count("chart") + text.lower().count("graph")
        
        # 10. Citation patterns indicating potential bias
        features['citation_but_count'] = (text.count("et al., but") + 
                                       text.count("et al. but") + 
                                       text.count("] but") +
                                       text.count(") but"))
        
        # 11. Self-reference patterns
        features['self_reference_count'] = (text.lower().count("our research") + 
                                        text.lower().count("our study") + 
                                        text.lower().count("our analysis") +
                                        text.lower().count("our findings") +
                                        text.lower().count("our company") +
                                        text.lower().count("our organization"))
        
        # Return feature values as a list in a consistent order
        feature_values = [
            features['length'], 
            features['avg_word_length'],
            features['p_value_count'], 
            features['signif_stars_count'],
            features['correlation_count'],
            features['percentage_count'],
            features['hedge_ratio'], 
            features['certainty_ratio'],
            features['theory_term_ratio'],
            features['jargon_term_ratio'],
            features['method_term_count'],
            features['validation_pattern_count'],
            features['abstract_claim_ratio'],
            features['results_stat_density'],
            features['limitations_mentioned'],
            features['performance_claim_ratio'],
            features['claim_consistency'],
            features['figure_mentions'],
            features['table_mentions'],
            features['chart_mentions'],
            features['citation_but_count'],
            features['self_reference_count']
        ]
        
        return feature_values
    
    def _extract_sections(self, text):
        """Extract sections from the business paper text"""
        section_dict = {}
        section_markers = {
            'abstract': ['abstract', 'executive summary'],
            'introduction': ['introduction', '1. introduction', 'i. introduction'],
            'literature': ['literature', 'literature review', 'related work', 'background'],
            'methods': ['method', 'methods', 'methodology', 'research design', 'data collection'],
            'analysis': ['analysis', 'data analysis', 'statistical analysis'],
            'results': ['results', 'findings', 'empirical results'],
            'discussion': ['discussion', 'implications', 'practical implications'],
            'conclusion': ['conclusion', 'conclusions', 'recommendations', 'concluding remarks']
        }
        
        text_lower = text.lower()
        
        for section_key, markers in section_markers.items():
            for marker in markers:
                # Try different formats of section headers
                patterns = [
                    f'\n{marker}\n',
                    f'\n{marker}.\n',
                    f'\n{marker}:\n',
                    f'\n{marker.title()}\n',
                    f'\n{marker.upper()}\n',
                    f'\n{marker.capitalize()}\n'
                ]
                
                for pattern in patterns:
                    start_pos = text_lower.find(pattern)
                    if start_pos != -1:
                        start_pos += len(pattern) - 1  # -1 to keep the last newline
                        
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
        """Extract key claims from text using business-specific terms"""
        claim_sentences = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in self.claim_terms):
                claim_sentences.append(sentence)
        
        return claim_sentences
    
    def _compare_claims(self, claims1, claims2):
        """Compare claims for consistency using business-specific terminology"""
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

class FeatureFusionLayer(nn.Module):
    """
    Learned feature fusion layer that combines BERT embeddings with handcrafted features
    """
    def __init__(self, bert_dim, handcrafted_dim, fusion_dim=256):
        super(FeatureFusionLayer, self).__init__()
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
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.3)  # Increased dropout
        
    def forward(self, bert_embedding, handcrafted_features):
        # Check if inputs are 2D or 3D, and project
        bert_proj = self.bert_projection(bert_embedding)            # [batch, fusion_dim] or [batch, seq, fusion_dim]
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)

        # Apply batch normalization if 2D
        if bert_proj.dim() == 2:
            bert_proj = self.bert_bn(bert_proj)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj)
            bert_proj = bert_proj.unsqueeze(1)  # Add seq_len = 1
            handcrafted_proj = handcrafted_proj.unsqueeze(1)
        elif bert_proj.dim() == 3:
            # For 3D tensors, apply batch norm to the last two dimensions
            batch_size, seq_len, dim = bert_proj.shape
            bert_proj = bert_proj.view(-1, dim)
            bert_proj = self.bert_bn(bert_proj)
            bert_proj = bert_proj.view(batch_size, seq_len, dim)
            
            handcrafted_proj = handcrafted_proj.view(-1, dim)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj)
            handcrafted_proj = handcrafted_proj.view(batch_size, seq_len, dim)

        # Apply attention (batch_first=True)
        attn_output, _ = self.attention(
            bert_proj, handcrafted_proj, handcrafted_proj
        )  # Output: [batch_size, seq_len, fusion_dim]

        # Remove seq_len dim if it is 1
        attn_output = attn_output.squeeze(1)
        bert_proj = bert_proj.squeeze(1)
        handcrafted_proj = handcrafted_proj.squeeze(1)

        # Gated fusion
        combined = torch.cat([bert_proj, handcrafted_proj], dim=-1)  # [batch_size, 2*fusion_dim]
        gate_values = self.gate(combined)
        fused = gate_values * bert_proj + (1 - gate_values) * handcrafted_proj
        output = self.layer_norm(fused)
        output = self.dropout(output)

        return output

class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention network for document classification
    Implements word-level, sentence-level, and section-level attention
    """
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
        """
        Args:
            word_embeddings: [batch_size, max_sections, max_sents, max_words, hidden_dim]
            attention_mask:  [batch_size, max_sections, max_sents, max_words]
        """
        batch_size, max_sections, max_sents, max_words, hidden_dim = word_embeddings.shape
        section_embeddings = []

        for b in range(batch_size):
            sentence_embeddings = []

            for s in range(max_sections):
                sent_embeddings = []

                for i in range(max_sents):
                    words = word_embeddings[b, s, i]           # [max_words, hidden_dim]
                    mask = attention_mask[b, s, i]             # [max_words]

                    if words.dim() == 3 and words.shape[0] == 1:
                        words = words.squeeze(0)
                    if mask.sum() > 0:
                        words = words.unsqueeze(1)  # [max_words, 1, hidden_dim]
                        words=words
                        key_padding_mask = (mask == 0).unsqueeze(0)  # [1, max_words]

                        attn_output, _ = self.word_attention(
                            words, words, words,
                            key_padding_mask=key_padding_mask
                        )
                        attn_output = attn_output.squeeze(1)  # [max_words, hidden_dim]

                        sent_embedding = (attn_output * mask.unsqueeze(-1)).sum(dim=0) / (mask.sum() + 1e-10)
                    else:
                        sent_embedding = torch.zeros(hidden_dim, device=words.device)

                    sent_embeddings.append(sent_embedding)

                if sent_embeddings:
                    section_sent_embeddings = torch.stack(sent_embeddings)  # [max_sents, hidden_dim]
                    sent_mask = (attention_mask[b, s].sum(dim=1) > 0).float()  # [max_sents]

                    if sent_mask.sum() > 0:
                        section_sent_embeddings = section_sent_embeddings.unsqueeze(1)  # [max_sents, 1, hidden_dim]
                        key_padding_mask = (sent_mask == 0).unsqueeze(0)  # [1, max_sents]

                        sent_attn_output, _ = self.sentence_attention(
                            section_sent_embeddings, section_sent_embeddings, section_sent_embeddings,
                            key_padding_mask=key_padding_mask
                        )
                        sent_attn_output = sent_attn_output.squeeze(1)

                        section_embedding = (sent_attn_output * sent_mask.unsqueeze(-1)).sum(dim=0) / (sent_mask.sum() + 1e-10)
                    else:
                        section_embedding = torch.zeros(hidden_dim, device=section_sent_embeddings.device)
                else:
                    section_embedding = torch.zeros(hidden_dim, device=word_embeddings.device)

                sentence_embeddings.append(section_embedding)

            if sentence_embeddings:
                doc_section_embeddings = torch.stack(sentence_embeddings)  # [max_sections, hidden_dim]
                section_mask = (attention_mask[b].sum(dim=(1, 2)) > 0).float()  # [max_sections]

                if section_mask.sum() > 0:
                    doc_section_embeddings = doc_section_embeddings.unsqueeze(1)  # [max_sections, 1, hidden_dim]
                    key_padding_mask = (section_mask == 0).unsqueeze(0)  # [1, max_sections]

                    section_attn_output, _ = self.section_attention(
                        doc_section_embeddings, doc_section_embeddings, doc_section_embeddings,
                        key_padding_mask=key_padding_mask
                    )
                    section_attn_output = section_attn_output.squeeze(1)

                    doc_embedding = (section_attn_output * section_mask.unsqueeze(-1)).sum(dim=0) / (section_mask.sum() + 1e-10)
                else:
                    doc_embedding = torch.zeros(hidden_dim, device=doc_section_embeddings.device)
            else:
                doc_embedding = torch.zeros(hidden_dim, device=word_embeddings.device)

            section_embeddings.append(doc_embedding)

        return torch.stack(section_embeddings)

# Add memory optimization for forward pass in HierarchicalBiasPredictionModel
def forward_with_memory_optimization(self, input_ids, attention_mask, handcrafted_features):
    batch_size = input_ids.size(0)
    
    # Reshape for BERT processing
    # Original shape: [batch_size, max_sections, max_sents, max_seq_length]
    max_sections, max_sents, max_seq_length = input_ids.size(1), input_ids.size(2), input_ids.size(3)
    
    # Process each sentence with BERT
    word_embeddings = []
    for b in range(batch_size):
        section_embeddings = []
        for s in range(max_sections):
            sentence_embeddings = []
            for i in range(max_sents):
                # Process single sentence through BERT
                input_ids_sent = input_ids[b, s, i].unsqueeze(0)  # [1, max_seq_length]
                attention_mask_sent = attention_mask[b, s, i].unsqueeze(0)  # [1, max_seq_length]
                
                # Skip processing empty sentences to save memory
                if attention_mask_sent.sum() > 0:
                    # Process with BERT but clear cache after each step
                    with torch.no_grad():  # Use no_grad for inference parts
                        outputs = self.bert(
                            input_ids=input_ids_sent, 
                            attention_mask=attention_mask_sent,
                            output_hidden_states=True
                        )
                        
                        # Get token embeddings from last layer
                        token_embeddings = outputs.last_hidden_state.squeeze(0).detach()  # [max_seq_length, hidden_size]
                        
                        # Explicitly delete to free memory
                        del outputs
                else:
                    # Create zero embeddings for empty sentences
                    token_embeddings = torch.zeros(
                        max_seq_length, self.bert_dim, device=input_ids.device)
                
                sentence_embeddings.append(token_embeddings)
            
            # Stack to get all sentence embeddings for this section
            if sentence_embeddings:
                section_embeddings.append(torch.stack(sentence_embeddings))
            else:
                # Create empty section if no sentences
                empty_section = torch.zeros(
                    max_sents, max_seq_length, self.bert_dim, device=input_ids.device)
                section_embeddings.append(empty_section)
        
        # Stack to get all section embeddings for this document
        if section_embeddings:
            word_embeddings.append(torch.stack(section_embeddings))
        else:
            # Create empty document if no sections
            empty_doc = torch.zeros(
                max_sections, max_sents, max_seq_length, self.bert_dim, device=input_ids.device)
            word_embeddings.append(empty_doc)
    
    # Stack to get embeddings for the entire batch
    word_embeddings = torch.stack(word_embeddings)  # [batch_size, max_sections, max_sents, max_seq_length, hidden_dim]
    
    # Apply hierarchical attention
    doc_embeddings = []
    for b in range(batch_size):
        doc_embedding = self.hierarchical_attention(
            word_embeddings[b].unsqueeze(0),  # Add batch dimension
            attention_mask[b].unsqueeze(0)
        )
        doc_embeddings.append(doc_embedding)
    
    # Free memory
    del word_embeddings
    
    doc_embeddings = torch.cat(doc_embeddings, dim=0)
    
    # Feature fusion
    fused_features = self.feature_fusion(doc_embeddings, handcrafted_features)
    
    # Free memory
    del doc_embeddings
    
    # Improved classification with residual connection
    x = self.fc1(fused_features)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.dropout(x)
    
    x2 = self.fc2(x)
    x2 = self.bn2(x2)
    x2 = self.relu(x2)
    
    # Residual connection
    shortcut = self.shortcut(fused_features)
    x2 = x2 + shortcut
    
    x2 = self.dropout(x2)
    output = self.fc3(x2)
    
    return output


# Modify the HierarchicalBiasPredictionModel class to use memory-optimized forward method
class HierarchicalBiasPredictionModel(nn.Module):
    def __init__(self, bert_model_name, num_classes=3, dropout_rate=0.3, feature_dim=22):
        super(HierarchicalBiasPredictionModel, self).__init__()
        
        print("Loading BERT with reduced complexity for CPU processing")
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bert_dim = self.bert.config.hidden_size  # 768 for standard BERT
        self.handcrafted_dim = feature_dim
        self.fusion_dim = 256  # Standard fusion dimension  
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(hidden_dim=self.bert_dim)
        
        # Feature fusion layer
        self.feature_fusion = FeatureFusionLayer(
            bert_dim=self.bert_dim, 
            handcrafted_dim=self.handcrafted_dim,
            fusion_dim=self.fusion_dim
        )
        
        # Classification layers - improved with batch normalization and residual connections
        self.fc1 = nn.Linear(self.fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Residual connection for fusion_dim -> 128 shortcut
        self.shortcut = nn.Linear(self.fusion_dim, 128) if self.fusion_dim != 128 else nn.Identity()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Increased dropout
        
        # Use memory-optimized forward
        self.forward = forward_with_memory_optimization.__get__(self)

# Enhanced model training function with advanced scheduling for large datasets
def train_hierarchical_model(train_dataloader, val_dataloader, model, device, 
                           epochs=20, lr=1e-4, accumulation_steps=2):
    """Train the hierarchical model with advanced learning rate scheduling"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    
    # Learning rate warmup + cosine annealing with restarts for large datasets
    total_steps = epochs * len(train_dataloader) // accumulation_steps
    warmup_steps = total_steps // 10  # 10% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Backup ReduceLROnPlateau for validation-based adjustment
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )
    
    # Calculate class weights for the balanced dataset
    train_labels = []
    for batch in train_dataloader:
        train_labels.extend(batch['label'].cpu().numpy())
    
    class_counts = torch.bincount(torch.tensor(train_labels))
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    
    print(f"Class distribution: {class_counts.cpu().numpy()}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Use Focal Loss with label smoothing for large balanced dataset
    loss_fn = FocalLoss(alpha=1, gamma=2, weight=class_weights, label_smoothing=0.1)
    
    best_val_accuracy = 0
    best_model = None
    patience = 7  # Increased patience for large dataset
    patience_counter = 0
    use_warmup_scheduler = True
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        # Add gradient clipping for stability
        max_grad_norm = 1.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, handcrafted_features)
            loss = loss_fn(outputs, labels)
            loss = loss / accumulation_steps  # Normalize loss
            
            loss.backward()
            train_loss += loss.item() * accumulation_steps  # Re-scale for logging
            
            # Accumulate gradients and update at intervals
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                # Use warmup scheduler for first half of training
                if use_warmup_scheduler:
                    scheduler.step()
                    
                optimizer.zero_grad()
        
        # Handle any remaining gradients at the end of epoch
        if len(train_dataloader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if use_warmup_scheduler:
                scheduler.step()
        
        train_loss /= len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                handcrafted_features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, handcrafted_features)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        
        # Switch to plateau scheduler after warmup phase (epoch 5)
        if epoch >= 5 and use_warmup_scheduler:
            use_warmup_scheduler = False
            print("Switching to plateau-based learning rate scheduling")
        
        # Update learning rate based on validation accuracy (after warmup)
        if not use_warmup_scheduler:
            plateau_scheduler.step(val_accuracy)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss = {train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}, '
              f'LR = {current_lr:.6f}')
        
        # Early stopping and best model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict().copy()
            torch.save(best_model, 'best_hierarchical_business_model.pt')
            patience_counter = 0
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Clear GPU cache and run garbage collection to avoid OOM on small GPUs
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    
    # Load best model
    model.load_state_dict(best_model)
    return model

# Enhanced evaluation function with business-specific metrics
def evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names=None):
    """Evaluate model on test set with business-specific analysis"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, handcrafted_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    unique_preds = np.unique(all_preds)
    unique_labels = np.unique(all_labels)
    print(f"Unique predictions: {unique_preds}")
    print(f"Unique labels in test set: {unique_labels}")
    
    # Use only the labels that are present in the data
    present_labels = sorted(set(np.concatenate([unique_preds, unique_labels])))
    filtered_label_names = [label_names[i] for i in present_labels] if present_labels else label_names[:1]
    
    try:
        # Classification report
        report = classification_report(all_labels, all_preds, 
                               target_names=filtered_label_names,
                               labels=present_labels,
                               output_dict=True)
        
        print(classification_report(all_labels, all_preds, 
                               target_names=filtered_label_names,
                               labels=present_labels))
        
        # Plot classification report as heatmap
        plt.figure(figsize=(10, 6))
        report_data = []
        for label in filtered_label_names:
            report_data.append([
                report[label]['precision'],
                report[label]['recall'],
                report[label]['f1-score']
            ])
        
        sns.heatmap(
            report_data, 
            annot=True,
            cmap='Blues',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=filtered_label_names
        )
        plt.title('Classification Performance by Business Research Bias Type')
        plt.tight_layout()
        plt.savefig('business_classification_performance.png')
        plt.close()
        
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        print("Falling back to basic accuracy calculation")
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0
        print(f"Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix if we have multiple classes
    if len(present_labels) > 1:
        cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Business Research Bias Detection - Confusion Matrix')
        plt.savefig('business_confusion_matrix.png')
        plt.close()
        
        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Business Research Bias Detection - Normalized Confusion Matrix')
        plt.savefig('business_normalized_confusion_matrix.png')
        plt.close()
    else:
        print("Not enough unique classes in test set to generate a meaningful confusion matrix")
    
    return all_preds, all_labels

def load_papers_from_json(json_file_path):
    """
    Load technology research papers from a JSON file.
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing paper data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing paper texts and bias labels
    """
    import pandas as pd
    import json
    import os
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found.")
        return pd.DataFrame(columns=['text', 'label'])
    
    try:
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different possible JSON structures
        papers = []
        
        # Case 1: List of paper objects
        if isinstance(data, list):
            for i, paper in enumerate(data):
                if not isinstance(paper, dict):
                    print(f"Warning: Item at index {i} is not a dictionary. Skipping.")
                    continue
                    
                # Extract text and label
                text = paper.get('Body', paper.get('text', paper.get('content', paper.get('abstract', ''))))
                
                # Extract label - prefer 'Overall Bias' or 'OverallBias' (case-insensitive), then fallbacks
                label = None
                # Try exact keys first
                for key in ['Overall Bias', 'OverallBias', 'Overall_Bias']:
                    if key in paper:
                        label = paper.get(key)
                        break
                # Try common alternative keys
                if label is None:
                    label = paper.get('OverallBias', paper.get('bias_label', paper.get('bias_type', 
                                  paper.get('Cognitive Bias', paper.get('Publication Bias', 
                                  paper.get('Selection Bias', paper.get('No Bias', 0)))))))
                # If still None, check case-insensitive keys
                if label is None:
                    for k, v in paper.items():
                        if k.replace('_', '').replace(' ', '').lower() == 'overallbias':
                            label = v
                            break
                
                # Convert string labels to integers if needed
                if isinstance(label, str):
                    # Normalize label strings
                    label_normal = label.strip().lower()
                    label_map = {
                        'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0, '0': 0,
                        'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1, '1': 1,
                        'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2, '2': 2,
                        'selection_bias': 2, 'selection': 2, 'selection bias': 2, 'selectionbias': 2,
                        'confirmation_bias': 1, 'confirmation': 1, 'confirmation bias': 1,
                        'survivorship_bias': 2, 'survivorship': 2, 'survivorship bias': 2
                    }
                    label = label_map.get(label_normal, None)
                    if label is None:
                        # Try to parse integers from strings like 'Overall Bias: 1'
                        m = re.search(r"(\d+)", label_normal)
                        if m:
                            label = int(m.group(1))
                        else:
                            label = 0
                
                # If label is float (like in sample data), convert to int
                if isinstance(label, float):
                    label = int(label)
                
                papers.append({'text': text, 'label': label})
                
        # Case 2: Dictionary with paper data
        elif isinstance(data, dict):
            # If it's a dictionary containing a list of papers
            if 'papers' in data and isinstance(data['papers'], list):
                papers_data = data['papers']
                for i, paper in enumerate(papers_data):
                    if not isinstance(paper, dict):
                        print(f"Warning: Item at index {i} in 'papers' is not a dictionary. Skipping.")
                        continue
                        
                    text = paper.get('text', paper.get('content', paper.get('abstract', '')))
                    # Prefer 'Overall Bias' keys
                    label = None
                    for key in ['Overall Bias', 'OverallBias', 'Overall_Bias']:
                        if key in paper:
                            label = paper.get(key)
                            break
                    if label is None:
                        label = paper.get('label', paper.get('bias_label', paper.get('bias_type', 
                                          paper.get('CognitiveBias', paper.get('PublicationBias', 
                                          paper.get('SelectionBias', paper.get('NoBias', 0)))))))

                    if label is None:
                        for k, v in paper.items():
                            if k.replace('_', '').replace(' ', '').lower() == 'overallbias':
                                label = v
                                break

                    # Normalize string labels
                    if isinstance(label, str):
                        label_normal = label.strip().lower()
                        label_map = {
                            'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0, '0': 0,
                            'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1, '1': 1,
                            'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2, '2': 2,
                            'selection_bias': 2, 'selection': 2, 'selection bias': 2, 'selectionbias': 2,
                            'confirmation_bias': 1, 'confirmation': 1, 'confirmation bias': 1,
                            'survivorship_bias': 2, 'survivorship': 2, 'survivorship bias': 2
                        }
                        label = label_map.get(label_normal, None)
                        if label is None:
                            m = re.search(r"(\d+)", label_normal)
                            if m:
                                label = int(m.group(1))
                            else:
                                label = 0

                    if isinstance(label, float):
                        label = int(label)

                    papers.append({'text': text, 'label': label})
            # If it's a dictionary with IDs as keys
            else:
                for paper_id, paper_data in data.items():
                    if isinstance(paper_data, dict):
                        text = paper_data.get('text', paper_data.get('content', paper_data.get('abstract', '')))
                        # Prefer 'Overall Bias' keys
                        label = None
                        for key in ['Overall Bias', 'OverallBias', 'Overall_Bias']:
                            if key in paper_data:
                                label = paper_data.get(key)
                                break
                        if label is None:
                            label = paper_data.get('label', paper_data.get('bias_label', paper_data.get('bias_type', 
                                              paper_data.get('CognitiveBias', paper_data.get('PublicationBias', 
                                              paper_data.get('SelectionBias', paper_data.get('NoBias', 0)))))))

                        if label is None:
                            for k, v in paper_data.items():
                                if k.replace('_', '').replace(' ', '').lower() == 'overallbias':
                                    label = v
                                    break

                        # Normalize string labels
                        if isinstance(label, str):
                            label_normal = label.strip().lower()
                            label_map = {
                                'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0, '0': 0,
                                'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1, '1': 1,
                                'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2, '2': 2,
                                'selection_bias': 2, 'selection': 2, 'selection bias': 2, 'selectionbias': 2,
                                'confirmation_bias': 1, 'confirmation': 1, 'confirmation bias': 1,
                                'survivorship_bias': 2, 'survivorship': 2, 'survivorship bias': 2
                            }
                            label = label_map.get(label_normal, None)
                            if label is None:
                                m = re.search(r"(\d+)", label_normal)
                                if m:
                                    label = int(m.group(1))
                                else:
                                    label = 0

                        if isinstance(label, float):
                            label = int(label)

                        papers.append({'text': text, 'label': label})
        
        # If no valid structure found
        else:
            print(f"Error: Unsupported JSON structure in {json_file_path}")
            return pd.DataFrame(columns=['text', 'label'])
        
        # Create DataFrame
        df = pd.DataFrame(papers)
        
        # Validate data
        if len(df) == 0:
            print(f"Warning: No valid paper data found in {json_file_path}")
            return pd.DataFrame(columns=['text', 'label'])
            
        if 'text' not in df.columns:
            print(f"Warning: 'text' column not found in the JSON data from {json_file_path}")
            df['text'] = ""
        
        if 'label' not in df.columns:
            print(f"Warning: 'label' column not found in the JSON data from {json_file_path}")
            df['label'] = 0
        
        # Ensure text is string and label is integer
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        
        # Remove any rows with empty text
        original_count = len(df)
        df = df[df['text'].str.strip().str.len() > 0].reset_index(drop=True)
        if len(df) < original_count:
            print(f"Info: Removed {original_count - len(df)} rows with empty text")
        
        # Validate label values
        if not all(df['label'].isin([0, 1, 2])):
            invalid_labels = df[~df['label'].isin([0, 1, 2])]['label'].unique()
            print(f"Warning: Found invalid label values: {invalid_labels}. Converting to 0.")
            df.loc[~df['label'].isin([0, 1, 2]), 'label'] = 0
        
        print(f"Successfully loaded {len(df)} technology papers from {json_file_path}")
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: {json_file_path} is not a valid JSON file: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"Error loading technology papers from {json_file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])

# Function to analyze business-specific feature importance
def analyze_business_feature_importance(model, feature_names):
    """
    Analyze which business-specific features are most important for bias detection
    Uses the weights from the feature fusion layer
    """
    # Extract feature fusion layer weights
    weights = model.feature_fusion.handcrafted_projection.weight.detach().cpu().numpy()
    
    # Calculate feature importance as the L2 norm of the weights for each feature
    importance = np.linalg.norm(weights, axis=0)
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Business-Specific Feature Importance for Bias Detection')
    plt.tight_layout()
    plt.savefig('business_feature_importance.png')
    plt.close()
    
    # Plot feature importance by category
    # Group features by category
    feature_categories = {
        'Basic Text': ['length', 'avg_word_length'],
        'Statistical': ['p_value_count', 'signif_stars_count', 'correlation_count', 
                       'percentage_count', 'results_stat_density'],
        'Linguistic': ['hedge_ratio', 'certainty_ratio'],
        'Business-Specific': ['theory_term_ratio', 'jargon_term_ratio', 'performance_claim_ratio'],
        'Methodology': ['method_term_count', 'validation_pattern_count'],
        'Section Analysis': ['abstract_claim_ratio', 'claim_consistency'],
        'Research Quality': ['limitations_mentioned'],
        'Visual/Tables': ['figure_mentions', 'table_mentions', 'chart_mentions'],
        'Citation': ['citation_but_count', 'self_reference_count']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        indices = [feature_names.index(f) for f in features if f in feature_names]
        if indices:
            category_importance[category] = importance[indices].mean()
    
    # Create DataFrame for category importance
    category_df = pd.DataFrame({
        'Category': list(category_importance.keys()),
        'Average Importance': list(category_importance.values())
    }).sort_values('Average Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Average Importance', y='Category', data=category_df)
    plt.title('Feature Category Importance for Business Research Bias Detection')
    plt.tight_layout()
    plt.savefig('business_feature_category_importance.png')
    plt.close()
    
    return importance_df, category_df

# Add a utility function to monitor memory usage
def print_memory_usage():
    # Get CPU memory info
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
    
    print(f"CPU Memory Usage: {mem_mb:.2f} MB")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Prefer GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature names for the business-specific extractor (updated for 22 features)
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'correlation_count', 'percentage_count', 'hedge_ratio', 'certainty_ratio', 
        'theory_term_ratio', 'jargon_term_ratio', 'method_term_count',
        'validation_pattern_count', 'abstract_claim_ratio', 'results_stat_density',
        'limitations_mentioned', 'performance_claim_ratio', 'claim_consistency',
        'figure_mentions', 'table_mentions', 'chart_mentions', 
        'citation_but_count', 'self_reference_count'
    ]
    
    # Load and process paper data from JSON
    try:
        # Specify the path to your business papers JSON file
        papers_df = load_papers_from_json('business_papers.json')
        print(f"Loaded {len(papers_df)} business papers")
        
        # Check class distribution
        print("Class distribution:")
        class_counts = papers_df['label'].value_counts()
        print(class_counts)
        
        # If there's insufficient data, create some dummy data for testing
        if len(papers_df) <= 3:  # Need at least 3 for train/val/test
            print("Warning: Not enough data found. Creating dummy data for testing.")
            # Create dummy business data
            papers_df = pd.DataFrame({
                'text': ['This is a sample business research paper with significant performance improvements and ROI of 25%.'] * 10,
                'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Dummy labels
            })
    except Exception as e:
        print(f"Error loading papers: {e}")
        # Generate dummy data for testing
        papers_df = pd.DataFrame({
            'text': ['This is a sample business research paper with significant performance improvements and ROI of 25%.'] * 10,
            'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Dummy labels
        })
    
    # Extract handcrafted features with business-specific extractor
    print("Extracting business-specific handcrafted features...")
    extractor = BusinessBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    # Convert to numpy arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Normalize features using StandardScaler for better training stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    print("Features normalized using StandardScaler")
    
    # Get indices for each class
    class_indices = {}
    for class_label in np.unique(labels_array):
        class_indices[class_label] = np.where(labels_array == class_label)[0]
    
    # Create test set with ~20% of data, ensuring at least one sample per class if available
    test_indices = []
    for class_label, indices in class_indices.items():
        # If a class has only 1 member, we'll put it in the training set
        if len(indices) > 1:
            # Take ~20% of samples from this class for testing
            n_test = max(1, int(0.2 * len(indices)))
            test_indices.extend(np.random.choice(indices, n_test, replace=False))
    
    # Get remaining indices for training/validation
    all_indices = set(range(len(labels_array)))
    train_val_indices = list(all_indices - set(test_indices))
    
    # Split train_val into train and validation (80/20 split)
    n_val = max(1, int(0.2 * len(train_val_indices)))
    val_indices = np.random.choice(train_val_indices, n_val, replace=False)
    train_indices = list(set(train_val_indices) - set(val_indices))
    
    # Create train, validation, and test sets
    X_train = [(papers_df['text'].iloc[i], features_array[i]) for i in train_indices]
    y_train = labels_array[train_indices]
    
    X_val = [(papers_df['text'].iloc[i], features_array[i]) for i in val_indices]
    y_val = labels_array[val_indices]
    
    X_test = [(papers_df['text'].iloc[i], features_array[i]) for i in test_indices]
    y_test = labels_array[test_indices]
    
    # Separate text and features
    train_texts, train_features = zip(*X_train) if X_train else ([], [])
    val_texts, val_features = zip(*X_val) if X_val else ([], [])
    test_texts, test_features = zip(*X_test) if X_test else ([], [])
    
    # Convert to numpy arrays for SMOTE
    train_features = np.array(train_features)
    train_labels = np.array(y_train)
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Apply SMOTE to the training data - keep original SMOTE implementation intact
    print("Applying SMOTE to balance classes...")
    try:
        # Get original class distribution
        original_counts = Counter(train_labels)
        print("Original class distribution:", original_counts)
        
        # Check if any class has very few samples
        min_samples_per_class = min(original_counts.values()) if original_counts else 0
        
        if min_samples_per_class >= 6:
            # Regular SMOTE can be applied
            smote = SMOTE(random_state=42)
            resampled_features, resampled_labels = smote.fit_resample(train_features, train_labels)
        else:
            # Custom approach for very small classes
            print(f"Found class with only {min_samples_per_class} samples. Using custom balancing approach.")
            
            # Find majority class count for reference
            max_count = max(original_counts.values()) if original_counts else 0
            target_count = max_count  # We'll aim for this many samples per class
            
            resampled_features = []
            resampled_labels = []
            
            # Process each class
            for class_label in np.unique(train_labels):
                class_indices = np.where(train_labels == class_label)[0]
                class_count = len(class_indices)
                
                if class_count >= 5:
                    # If we have enough samples, use SMOTE for this class specifically
                    try:
                        class_features = train_features[class_indices]
                        # Use SMOTE with fewer neighbors
                        k_neighbors = min(class_count - 1, 5)  # Max neighbors possible
                        if k_neighbors >= 1:  # Need at least 1 neighbor
                            smote_single = SMOTE(k_neighbors=k_neighbors, random_state=42)
                            
                            # Create a temporary binary classification problem
                            temp_labels = np.zeros(len(train_labels))
                            temp_labels[class_indices] = 1
                            
                            # Only apply SMOTE to samples of this class and a similar number of other samples
                            other_indices = np.random.choice(
                                np.where(temp_labels == 0)[0], 
                                min(len(class_indices) * 2, len(train_labels) - len(class_indices)),
                                replace=False
                            )
                            
                            temp_indices = np.concatenate([class_indices, other_indices])
                            temp_features = train_features[temp_indices]
                            temp_class_labels = temp_labels[temp_indices]
                            
                            # Apply SMOTE to this subset
                            resampled_temp_features, resampled_temp_labels = smote_single.fit_resample(
                                temp_features, temp_class_labels)
                            
                            # Extract only the synthetic samples of our target class
                            synthetic_indices = np.where(
                                (resampled_temp_labels == 1) & 
                                (np.arange(len(resampled_temp_labels)) >= len(temp_features))
                            )[0]
                            
                            synthetic_features = resampled_temp_features[synthetic_indices]
                            
                            # Calculate how many samples we need
                            samples_needed = target_count - class_count
                            
                            if samples_needed > 0 and len(synthetic_features) > 0:
                                # Sample with replacement if we need more than we generated
                                if samples_needed > len(synthetic_features):
                                    indices = np.random.choice(
                                        len(synthetic_features), samples_needed, replace=True)
                                else:
                                    indices = np.random.choice(
                                        len(synthetic_features), samples_needed, replace=False)
                                
                                selected_synthetic_features = synthetic_features[indices]
                                
                                # Add original and synthetic samples
                                resampled_features.extend(train_features[class_indices])
                                resampled_features.extend(selected_synthetic_features)
                                resampled_labels.extend([class_label] * class_count)
                                resampled_labels.extend([class_label] * samples_needed)
                            else:
                                # Just add original samples
                                resampled_features.extend(train_features[class_indices])
                                resampled_labels.extend([class_label] * class_count)
                        else:
                            # Not enough neighbors, just duplicate
                            resampled_features.extend(train_features[class_indices])
                            resampled_labels.extend([class_label] * class_count)
                            
                            # Duplicate to reach target count
                            samples_needed = target_count - class_count
                            if samples_needed > 0:
                                # Random sampling with replacement from this class
                                indices = np.random.choice(class_count, samples_needed, replace=True)
                                selected_features = train_features[class_indices][indices]
                                resampled_features.extend(selected_features)
                                resampled_labels.extend([class_label] * samples_needed)
                    
                    except Exception as e:
                        print(f"Error applying SMOTE to class {class_label}: {e}")
                        # Fallback to simple duplication
                        resampled_features.extend(train_features[class_indices])
                        resampled_labels.extend([class_label] * class_count)
                else:
                    # For very small classes, use simple duplication
                    resampled_features.extend(train_features[class_indices])
                    resampled_labels.extend([class_label] * class_count)
                    
                    # Duplicate to reach target count
                    samples_needed = target_count - class_count
                    if samples_needed > 0:
                        # Random sampling with replacement from this class
                        indices = np.random.choice(class_count, samples_needed, replace=True)
                        selected_features = train_features[class_indices][indices]
                        resampled_features.extend(selected_features)
                        resampled_labels.extend([class_label] * samples_needed)
        
            # Convert lists to numpy arrays
            resampled_features = np.array(resampled_features)
            resampled_labels = np.array(resampled_labels)
        
        # Generate synthetic texts for new samples
        # First, map original labels to their texts
        label_to_texts = {}
        for text, label in zip(train_texts, train_labels):
            if label not in label_to_texts:
                label_to_texts[label] = []
            label_to_texts[label].append(text)
        
        # Count new samples per class
        new_counts = Counter(resampled_labels)
        
        # Generate texts for all samples (including synthetic ones)
        synthetic_texts = []
        for i, label in enumerate(resampled_labels):
            if i < len(train_texts):  # Original sample
                synthetic_texts.append(train_texts[i])
            else:  # Synthetic sample
                # Randomly select a text from the same class
                if label in label_to_texts and label_to_texts[label]:
                    synthetic_texts.append(np.random.choice(label_to_texts[label]))
                else:
                    # Fallback text if somehow we don't have any text for this label
                    synthetic_texts.append(f"Synthetic business research paper text for class {label}")
        
        # Update train_texts and train_features with resampled data
        train_texts = synthetic_texts
        train_features = resampled_features
        train_labels = resampled_labels
        
        print("After balancing class distribution:", new_counts)
        
    except Exception as e:
        print(f"Error in data balancing process: {e}")
        print("Proceeding with original imbalanced data")
    
    # Memory optimization - clear unnecessary variables
    del papers_df, handcrafted_features, features_array, labels_array
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    try:
        # Use SciBERT which is better for scientific papers - with CPU settings
        model_name = 'allenai/scibert_scivocab_uncased'
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Reduce model complexity to save memory for the available device
    print(f"Using reduced model complexity for device: {device}")
    max_seq_length = 64   # Reduced from 128
    max_sections = 2      # Reduced from 4
    max_sents = 4         # Reduced from 8
    
    # Create hierarchical datasets with reduced complexity
    print("Creating hierarchical datasets...")
    train_dataset = HierarchicalBusinessPaperDataset(
        train_texts, train_labels, tokenizer, train_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    val_dataset = HierarchicalBusinessPaperDataset(
        val_texts, y_val, tokenizer, val_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    test_dataset = HierarchicalBusinessPaperDataset(
        test_texts, y_test, tokenizer, test_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Use larger batch size for the balanced large dataset (11,709 samples)
    batch_size = 8  # Increased from 1 for better gradient estimates
    print(f"Using batch size: {batch_size} for large balanced dataset ({len(train_texts)} samples)")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize hierarchical model - force CPU execution
    print("Initializing hierarchical model for CPU execution...")
    try:
        print("Using CPU memory optimization settings")
        model = HierarchicalBiasPredictionModel(
            'allenai/scibert_scivocab_uncased', 
            num_classes=3,
            feature_dim=len(feature_names),  # 22 features for business
            dropout_rate=0.5  # Increased dropout rate for regularization
        )
    except Exception as e:
        print(f"Error loading SciBERT for hierarchical model: {e}")
        print("Falling back to bert-base-uncased")
        model = HierarchicalBiasPredictionModel(
            'bert-base-uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.5
        )
    
    # Ensure model is on CPU
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Train hierarchical model optimized for large balanced dataset
    print("Training hierarchical model...")
    model = train_hierarchical_model(
        train_dataloader, val_dataloader, model, device, 
        epochs=20,  # More epochs needed for large dataset
        accumulation_steps=2,  # Reduced since we have larger batch size
        lr=1e-4  # Higher learning rate for large balanced dataset
    )
    
    # Evaluate hierarchical model
    print("Evaluating hierarchical model...")
    label_names = ['No Bias', 'Cognitive Bias', 'Selection/Publication Bias']
    evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names)
    
    # Analyze business-specific feature importance
    print("Analyzing business-specific feature importance...")
    analyze_business_feature_importance(model, feature_names)
    
    print("Business bias detection analysis complete!")

if __name__ == "__main__":
    # Set the Python multiprocessing method to avoid issues on Windows
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Run the main function
    main()