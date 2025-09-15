import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import re
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

# Enhanced dataset class to support hierarchical structure
class HierarchicalResearchPaperDataset(Dataset):
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
        """Extract sections from the paper text"""
        # Simple section extraction - in practice, you might want more sophisticated parsing
        section_markers = [
            'abstract', 'introduction', 'literature review', 'methodology', 
            'data', 'results', 'discussion', 'conclusion', 'references'
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

# Enhanced feature extractor for computer science papers
class ComputerScienceBiasFeatureExtractor:
    """Extract features that might indicate bias in computer science research papers"""
    
    def __init__(self):
        # Basic patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.0\d+'
        self.significance_stars = r'\*{1,3}\s*'
        self.performance_pattern = r'(?:accuracy|precision|recall|f1|auc|rmse|mae)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+'
        
        # Computer science-specific hedge words
        self.hedge_words = [
            'may', 'might', 'could', 'possibly', 'potentially', 'suggests', 
            'appears', 'seems', 'likely', 'unlikely', 'perhaps', 'arguably',
            'tend to', 'tends to', 'tended to', 'indicate', 'indicates',
            'approximately', 'roughly', 'around', 'about'
        ]
        
        # Computer science-specific certainty words
        self.certainty_words = [
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'conclusively', 'absolutely', 'always', 'never', 'established',
            'proves', 'demonstrates', 'robust', 'significant', 'substantial',
            'strong evidence', 'strongly supports', 'decisive', 'optimal',
            'best', 'superior', 'outperforms', 'state-of-the-art'
        ]
        
        # Computer science theory and method references
        self.theory_terms = [
            'machine learning', 'deep learning', 'neural network', 'algorithm',
            'theoretical analysis', 'complexity analysis', 'big o', 'np-hard',
            'polynomial time', 'exponential time', 'optimization', 'heuristic'
        ]
        
        # Computer science-specific claim words
        self.claim_terms = [
            'performance', 'efficiency', 'accuracy', 'speed', 'scalability',
            'improvement', 'enhancement', 'optimization', 'breakthrough',
            'novel', 'innovative', 'cutting-edge', 'state-of-the-art',
            'faster', 'better', 'superior', 'outperforms'
        ]
        
        # Computer science jargon that might signal overconfidence
        self.cs_jargon = [
            'big data', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural networks', 'convolutional', 'recurrent', 'transformer',
            'gradient descent', 'backpropagation', 'overfitting', 'underfitting',
            'cross-validation', 'hyperparameter', 'feature engineering'
        ]
        
        # Evaluation and validation terms
        self.evaluation_terms = [
            'evaluation', 'validation', 'testing', 'benchmark', 'baseline',
            'comparison', 'ablation study', 'cross-validation', 'holdout',
            'train-test split', 'k-fold', 'statistical significance'
        ]
        
        # Computer science-specific method patterns
        self.cs_method_patterns = [
            r'CNN', r'RNN', r'LSTM', r'GRU', r'SVM', r'Random Forest',
            r'Gradient Boosting', r'K-means', r'PCA', r't-SNE',
            r'BERT', r'GPT', r'Transformer', r'ResNet', r'VGG'
        ]
        
        # Stopwords for cleaning text
        self.stopwords = set(stopwords.words('english'))
        
    def extract_features(self, text):
        features = {}
        
        # Handle None or empty text
        if text is None or not text:
            # Return default values (20 features)
            return [0] * 20
        
        # Extract sections (abstract, introduction, methods, results, discussion, conclusion)
        sections = self._extract_sections(text)
        
        # 1. Basic text statistics
        features['length'] = len(text)
        word_count = len(text.split()) + 1  # Add 1 to avoid division by zero
        features['avg_word_length'] = sum(len(w) for w in text.split()) / word_count
        
        # 2. Statistical reporting patterns
        features['p_value_count'] = len(re.findall(self.p_value_pattern, text))
        features['signif_stars_count'] = len(re.findall(self.significance_stars, text))
        features['performance_count'] = len(re.findall(self.performance_pattern, text))
        
        # 3. Linguistic features
        # Hedging and certainty
        hedge_count = sum(text.lower().count(word) for word in self.hedge_words)
        certainty_count = sum(text.lower().count(word) for word in self.certainty_words)
        features['hedge_ratio'] = hedge_count / word_count * 1000  # per 1000 words
        features['certainty_ratio'] = certainty_count / word_count * 1000  # per 1000 words
        
        # 4. Computer science-specific patterns
        theory_count = sum(text.lower().count(term) for term in self.theory_terms)
        jargon_count = sum(text.lower().count(term) for term in self.cs_jargon)
        features['theory_term_ratio'] = theory_count / word_count * 1000
        features['jargon_term_ratio'] = jargon_count / word_count * 1000
        
        # 5. Computer science method patterns
        cs_method_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in self.cs_method_patterns)
        features['cs_method_count'] = cs_method_count
        
        # 6. Section-specific features
        # Abstract features (claims without evidence)
        if 'abstract' in sections:
            abstract = sections['abstract']
            abstract_words = len(abstract.split()) + 1
            abstract_claim_count = sum(abstract.lower().count(term) for term in self.claim_terms)
            features['abstract_claim_ratio'] = abstract_claim_count / abstract_words * 1000
        else:
            features['abstract_claim_ratio'] = 0
        
        # Results section features (performance reporting)
        if 'results' in sections:
            results = sections['results']
            results_performance = len(re.findall(self.performance_pattern, results))
            results_words = len(results.split()) + 1
            features['results_performance_density'] = results_performance / results_words * 1000
        else:
            features['results_performance_density'] = 0
        
        # Limitations acknowledgment
        features['limitations_mentioned'] = 1 if "limitation" in text.lower() or "limitations" in text.lower() else 0
        
        # 7. Evaluation checks
        evaluation_count = sum(text.lower().count(term) for term in self.evaluation_terms)
        features['evaluation_ratio'] = evaluation_count / word_count * 1000
        
        # 8. Abstract vs conclusion claim consistency
        if 'abstract' in sections and 'conclusion' in sections:
            abstract = sections['abstract']
            conclusion = sections['conclusion']
            abstract_claims = self._extract_key_claims(abstract)
            conclusion_claims = self._extract_key_claims(conclusion)
            features['claim_consistency'] = self._compare_claims(abstract_claims, conclusion_claims)
        else:
            features['claim_consistency'] = 0
            
        # 9. Figure and table mentions
        features['figure_mentions'] = text.lower().count("figure") + text.lower().count("fig.")
        features['table_mentions'] = text.lower().count("table")
        
        # 10. Comparison patterns
        features['comparison_count'] = (text.count("compared to") + 
                                     text.count("versus") + 
                                     text.count("vs.") +
                                     text.count("outperforms") +
                                     text.count("better than"))
        
        # 11. Self-citation patterns
        features['self_reference_count'] = (text.lower().count("our method") + 
                                        text.lower().count("our approach") + 
                                        text.lower().count("our algorithm") +
                                        text.lower().count("our model") +
                                        text.lower().count("our system"))
        
        # Return feature values as a list in a consistent order
        feature_values = [
            features['length'], 
            features['avg_word_length'],
            features['p_value_count'], 
            features['signif_stars_count'],
            features['performance_count'],
            features['hedge_ratio'], 
            features['certainty_ratio'],
            features['theory_term_ratio'],
            features['jargon_term_ratio'],
            features['cs_method_count'],
            features['abstract_claim_ratio'],
            features['results_performance_density'],
            features['limitations_mentioned'],
            features['evaluation_ratio'],
            features['claim_consistency'],
            features['figure_mentions'],
            features['table_mentions'],
            features['comparison_count'],
            features['self_reference_count'],
            features.get('method_limitation_ratio', 0)  # Include with default if not set
        ]
        
        return feature_values
    
    def _extract_sections(self, text):
        """Extract sections from the paper text"""
        section_dict = {}
        section_markers = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', 'i. introduction'],
            'literature': ['literature', 'literature review', 'related work', 'related literature'],
            'methods': ['method', 'methods', 'methodology', 'approach', 'algorithm'],
            'data': ['data', 'dataset', 'experimental setup', 'experiments'],
            'results': ['results', 'findings', 'experimental results', 'evaluation'],
            'discussion': ['discussion'],
            'conclusion': ['conclusion', 'conclusions', 'concluding remarks']
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
        """Extract key claims from text using computer science-specific terms"""
        claim_sentences = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in self.claim_terms):
                claim_sentences.append(sentence)
        
        return claim_sentences
    
    def _compare_claims(self, claims1, claims2):
        """Compare claims for consistency using computer science-specific terminology"""
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
        # Check if inputs are 2D or 3D, and project
        bert_proj = self.bert_projection(bert_embedding)            # [batch, fusion_dim] or [batch, seq, fusion_dim]
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)

        # Ensure 3D shape: [batch_size, seq_len, fusion_dim]
        if bert_proj.dim() == 2:
            bert_proj = bert_proj.unsqueeze(1)  # Add seq_len = 1
        if handcrafted_proj.dim() == 2:
            handcrafted_proj = handcrafted_proj.unsqueeze(1)

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


# Main HierarchicalBiasPredictionModel class for GPU execution


class HierarchicalBiasPredictionModel(nn.Module):
    def __init__(self, bert_model_name, num_classes=3, dropout_rate=0.3, feature_dim=20):
        super(HierarchicalBiasPredictionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Enable gradient checkpointing to save memory with explicit reentrant setting
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
            # Set reentrant to False to avoid the warning
            if hasattr(self.bert.config, 'use_reentrant'):
                self.bert.config.use_reentrant = False
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dimensions optimized for 4GB GPU
        self.bert_dim = self.bert.config.hidden_size
        self.handcrafted_dim = feature_dim
        self.fusion_dim = 256  # Balanced fusion dimension for learning
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(hidden_dim=self.bert_dim)
        
        # Feature fusion layer
        self.feature_fusion = FeatureFusionLayer(
            bert_dim=self.bert_dim, 
            handcrafted_dim=self.handcrafted_dim,
            fusion_dim=self.fusion_dim
        )
        
        # Classification layers - balanced for learning
        self.fc1 = nn.Linear(self.fusion_dim, 128)  # Better capacity for learning
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(128)
        
    def forward(self, input_ids, attention_mask, handcrafted_features):
        batch_size = input_ids.size(0)
        
        # Reshape for BERT processing
        # Original shape: [batch_size, max_sections, max_sents, max_seq_length]
        max_sections, max_sents, max_seq_length = input_ids.size(1), input_ids.size(2), input_ids.size(3)
        
        # OPTIMIZED: Batch process all sentences at once instead of nested loops
        # Reshape to process all sentences in a single BERT call
        total_sentences = batch_size * max_sections * max_sents
        
        # Flatten input tensors
        flat_input_ids = input_ids.view(total_sentences, max_seq_length)
        flat_attention_mask = attention_mask.view(total_sentences, max_seq_length)
        
        # Find non-empty sentences to avoid processing padding
        non_empty_mask = flat_attention_mask.sum(dim=1) > 0
        
        if non_empty_mask.sum() > 0:
            # Process only non-empty sentences
            non_empty_input_ids = flat_input_ids[non_empty_mask]
            non_empty_attention_mask = flat_attention_mask[non_empty_mask]
            
            # Single BERT forward pass for all non-empty sentences
            outputs = self.bert(
                input_ids=non_empty_input_ids,
                attention_mask=non_empty_attention_mask,
                output_hidden_states=False  # Don't need hidden states, just last layer
            )
            
            # Get sentence-level embeddings using [CLS] token
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # [num_non_empty, hidden_size]
        else:
            sentence_embeddings = torch.empty(0, self.bert_dim, device=input_ids.device)
        
        # Create full embedding tensor and fill in non-empty sentences
        all_sentence_embeddings = torch.zeros(total_sentences, self.bert_dim, device=input_ids.device)
        if non_empty_mask.sum() > 0:
            all_sentence_embeddings[non_empty_mask] = sentence_embeddings
        
        # Reshape back to hierarchical structure
        doc_embeddings = all_sentence_embeddings.view(batch_size, max_sections, max_sents, self.bert_dim)
        
        # Simple aggregation instead of complex hierarchical attention for speed
        # Average over sentences and sections
        section_embeddings = doc_embeddings.mean(dim=2)  # Average sentences within sections
        final_doc_embeddings = section_embeddings.mean(dim=1)  # Average sections within documents
        
        # Feature fusion
        fused_features = self.feature_fusion(final_doc_embeddings, handcrafted_features)
        
        # Final classification
        x = self.fc1(fused_features)
        x = self.relu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output
    
# Enhanced model training function with gradient accumulation and mixed precision
def train_hierarchical_model(train_dataloader, val_dataloader, model, device, 
                           epochs=5, lr=2e-5, accumulation_steps=2):
    """Train the hierarchical model with gradient accumulation and mixed precision"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_dataloader))
    loss_fn = nn.CrossEntropyLoss()
    
    # Enable mixed precision training for GPU
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training for GPU acceleration")
    
    best_val_accuracy = 0
    best_model = None
    
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch+1}/{epochs}...")
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        print(f"Processing {len(train_dataloader)} training batches...")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Processing batch {batch_idx+1}/{len(train_dataloader)}...")
            
            # Clear cache less frequently for speed
            if device.type == 'cuda' and batch_idx % 25 == 0:  # Clear every 25 batches
                clear_gpu_cache(f"Batch {batch_idx}")
            
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                handcrafted_features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)
                
                # Debug: Print tensor shapes
                if batch_idx == 0:
                    print(f"    Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
                    print(f"    Features shape: {handcrafted_features.shape}, Labels shape: {labels.shape}")
            except Exception as e:
                print(f"Error loading batch {batch_idx}: {e}")
                continue
            
            # Mixed precision forward pass
            if use_amp:
                try:
                    with torch.cuda.amp.autocast():
                        if batch_idx == 0:
                            print(f"    Running forward pass...")
                        outputs = model(input_ids, attention_mask, handcrafted_features)
                        if batch_idx == 0:
                            print(f"    Forward pass completed, output shape: {outputs.shape}")
                        loss = loss_fn(outputs, labels)
                        loss = loss / accumulation_steps  # Normalize loss
                except Exception as e:
                    print(f"Error in forward pass (batch {batch_idx}): {e}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                
                scaler.scale(loss).backward()
                train_loss += loss.item() * accumulation_steps  # Re-scale for logging
                
                # Accumulate gradients and update at intervals
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    # Clear cache after optimizer step for 4GB GPU
                    if device.type == 'cuda':
                        clear_gpu_cache("After Optimizer Step")
            else:
                try:
                    if batch_idx == 0:
                        print(f"    Running forward pass (CPU mode)...")
                    outputs = model(input_ids, attention_mask, handcrafted_features)
                    if batch_idx == 0:
                        print(f"    Forward pass completed, output shape: {outputs.shape}")
                    loss = loss_fn(outputs, labels)
                    loss = loss / accumulation_steps  # Normalize loss
                except Exception as e:
                    print(f"Error in forward pass (batch {batch_idx}): {e}")
                    continue
                
                loss.backward()
                train_loss += loss.item() * accumulation_steps  # Re-scale for logging
                
                # Accumulate gradients and update at intervals
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # Clear cache after optimizer step for 4GB GPU
                    if device.type == 'cuda':
                        clear_gpu_cache("After Optimizer Step")
        
        # Handle any remaining gradients at the end of epoch
        if len(train_dataloader) % accumulation_steps != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
        
        train_loss /= len(train_dataloader)
        
        print(f"\nStarting validation for epoch {epoch+1}...")
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_dataloader):
                # Clear cache before validation batch for 4GB GPU
                if device.type == 'cuda' and val_batch_idx % 10 == 0:  # Clear every 10 validation batches
                    clear_gpu_cache(f"Validation Batch {val_batch_idx}")
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    handcrafted_features = batch['handcrafted_features'].to(device)
                    labels = batch['label'].to(device)
                except Exception as e:
                    print(f"Error loading validation batch {val_batch_idx}: {e}")
                    continue
                
                try:
                    outputs = model(input_ids, attention_mask, handcrafted_features)
                    loss = loss_fn(outputs, labels)
                except Exception as e:
                    print(f"Error in validation forward pass (batch {val_batch_idx}): {e}")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_dataloader)
        val_accuracy = correct / total
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss = {train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}, '
              f'LR = {scheduler.get_last_lr()[0]:.6f}')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict().copy()
            torch.save(best_model, 'best_hierarchical_model.pt')
    
    # Load best model
    model.load_state_dict(best_model)
    return model

# Enhanced evaluation function with computer science-specific metrics
def evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names=None):
    """Evaluate model on test set with computer science-specific analysis"""
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
        plt.title('Classification Performance by Bias Type')
        plt.tight_layout()
        plt.savefig('classification_performance.png')
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
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.savefig('normalized_confusion_matrix.png')
        plt.close()
    else:
        print("Not enough unique classes in test set to generate a meaningful confusion matrix")
    
    # Analyze feature importance if feature names are provided
    if feature_names is not None:
        # Feature analysis will be implemented in a separate function
        pass
    
    return all_preds, all_labels

def load_papers_from_json(json_file_path):
    """
    Load computer science research papers from a JSON file.
    
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
                
                # Extract label - use 'Overall Bias' (with space, as in CS data)
                label = paper.get('Overall Bias', paper.get('OverallBias', paper.get('bias_label', 
                                  paper.get('bias_type', None))))
                
                # Skip papers with missing or empty bias labels (but not numeric 0 which is valid)
                if label is None or label == "":
                    print(f"Warning: Skipping paper at index {i} - missing or empty bias label")
                    continue
                
                # Convert string labels to integers if needed
                if isinstance(label, str):
                    # Handle exact matches first (CS data format)
                    if label == 'No Bias':
                        label = 0
                    elif label == 'Cognitive Bias':
                        label = 1
                    elif label == 'Publication Bias':
                        label = 2
                    else:
                        # Fallback to case-insensitive mapping
                        label_map = {
                            'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0,
                            'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1,
                            'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2
                        }
                        mapped_label = label_map.get(label.strip().lower(), None)
                        if mapped_label is None:
                            print(f"Warning: Skipping paper at index {i} - unknown bias label '{label}'")
                            continue
                        label = mapped_label
                
                # If label is float (like in sample data), convert to int
                elif isinstance(label, float):
                    if label in [0.0, 1.0, 2.0]:
                        label = int(label)
                    else:
                        print(f"Warning: Skipping paper at index {i} - invalid float label '{label}'")
                        continue
                
                # Skip papers with empty text
                if not text or text.strip() == "":
                    print(f"Warning: Skipping paper at index {i} - empty text content")
                    continue
                
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
                    label = paper.get('label', paper.get('bias_label', paper.get('bias_type', 
                                      paper.get('CognitiveBias', paper.get('PublicationBias', 
                                      paper.get('NoBias', 0))))))
                    
                    # Convert string labels if needed
                    if isinstance(label, str):
                        label_map = {
                            'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0,
                            'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1,
                            'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2
                        }
                        label = label_map.get(label.lower(), 0)
                    
                    # If label is float, convert to int
                    if isinstance(label, float):
                        label = int(label)
                    
                    papers.append({'text': text, 'label': label})
            # If it's a dictionary with IDs as keys
            else:
                for paper_id, paper_data in data.items():
                    if isinstance(paper_data, dict):
                        text = paper_data.get('text', paper_data.get('content', paper_data.get('abstract', '')))
                        label = paper_data.get('label', paper_data.get('bias_label', paper_data.get('bias_type', 
                                          paper_data.get('CognitiveBias', paper_data.get('PublicationBias', 
                                          paper_data.get('NoBias', 0))))))
                        
                        # Convert string labels if needed
                        if isinstance(label, str):
                            label_map = {
                                'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0,
                                'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1,
                                'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2
                            }
                            label = label_map.get(label.lower(), 0)
                        
                        # If label is float, convert to int
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
        
        # Validate label values - should only be 0, 1, 2 at this point
        invalid_labels = df[~df['label'].isin([0, 1, 2])]
        if len(invalid_labels) > 0:
            print(f"Error: Found {len(invalid_labels)} papers with invalid labels after filtering:")
            for idx, row in invalid_labels.iterrows():
                print(f"  Index {idx}: label = {row['label']}")
            # Remove invalid entries rather than converting
            df = df[df['label'].isin([0, 1, 2])].reset_index(drop=True)
            print(f"Removed {len(invalid_labels)} invalid entries")
        
        # Final summary
        final_counts = df['label'].value_counts().sort_index()
        print(f"Successfully loaded {len(df)} papers from {json_file_path}")
        print(f"Final label distribution:")
        for label_num, count in final_counts.items():
            label_names = {0: 'No Bias', 1: 'Cognitive Bias', 2: 'Publication Bias'}
            print(f"  {label_names[label_num]} ({label_num}): {count} papers")
        
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: {json_file_path} is not a valid JSON file: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"Error loading papers from {json_file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])

# Function to analyze computer science-specific feature importance
def analyze_cs_feature_importance(model, feature_names):
    """
    Analyze which computer science-specific features are most important for bias detection
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
    plt.title('Computer Science-Specific Feature Importance for Bias Detection')
    plt.tight_layout()
    plt.savefig('cs_feature_importance.png')
    plt.close()
    
    # Plot feature importance by category
    # Group features by category
    feature_categories = {
        'Basic Text': ['length', 'avg_word_length'],
        'Statistical': ['p_value_count', 'signif_stars_count', 'performance_count', 'results_performance_density'],
        'Linguistic': ['hedge_ratio', 'certainty_ratio'],
        'CS-Specific': ['theory_term_ratio', 'jargon_term_ratio', 'cs_method_count'],
        'Section Analysis': ['abstract_claim_ratio', 'claim_consistency'],
        'Rigor': ['limitations_mentioned', 'evaluation_ratio'],
        'Visual/Tables': ['figure_mentions', 'table_mentions'],
        'Comparison': ['comparison_count', 'self_reference_count']
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
    plt.title('Feature Category Importance for Computer Science Research Bias Detection')
    plt.tight_layout()
    plt.savefig('cs_feature_category_importance.png')
    plt.close()
    
    return importance_df, category_df

# Add a utility function to monitor memory usage
def print_memory_usage(stage=""):
    # Get CPU memory info
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB
    
    stage_prefix = f"[{stage}] " if stage else ""
    print(f"{stage_prefix}CPU Memory Usage: {mem_mb:.2f} MB")
    
    # GPU memory info if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        print(f"{stage_prefix}GPU Memory Allocated: {gpu_memory:.2f} MB")
        print(f"{stage_prefix}GPU Memory Reserved: {gpu_memory_cached:.2f} MB")
        print(f"{stage_prefix}GPU Memory Peak: {gpu_memory_peak:.2f} MB")
        print(f"{stage_prefix}GPU Memory Usage: {(gpu_memory_cached/total_gpu_memory)*100:.1f}% of {total_gpu_memory:.0f} MB")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_gpu_cache(stage=""):
    """Comprehensive GPU cache clearing with status reporting"""
    if torch.cuda.is_available():
        stage_prefix = f"[{stage}] " if stage else ""
        print(f"{stage_prefix}Clearing GPU cache...")
        
        # Get memory before clearing
        before_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        before_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Get memory after clearing
        after_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        after_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        freed_allocated = before_allocated - after_allocated
        freed_reserved = before_reserved - after_reserved
        
        if freed_allocated > 0 or freed_reserved > 0:
            print(f"{stage_prefix}Freed {freed_allocated:.1f} MB allocated, {freed_reserved:.1f} MB reserved")
        else:
            print(f"{stage_prefix}Cache already clean")

def main():
    # Clear GPU cache at startup for clean memory state
    if torch.cuda.is_available():
        print("Clearing GPU cache from previous runs...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Force garbage collection
        gc.collect()
        print("GPU cache cleared successfully!")
    
    # Configure CUDA memory management for 4GB GPU
    if torch.cuda.is_available():
        # Set memory fraction to avoid fragmentation
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        # Clear any existing cache again after configuration
        torch.cuda.empty_cache()
        # Set memory allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total_memory:.1f} GB")
        if total_memory < 6.0:  # Less than 6GB
            print("WARNING: Low GPU memory detected. Using memory-optimized settings.")
    
    # Feature names for the computer science-specific extractor
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'performance_count', 'hedge_ratio', 'certainty_ratio', 'theory_term_ratio',
        'jargon_term_ratio', 'cs_method_count', 'abstract_claim_ratio',
        'results_performance_density', 'limitations_mentioned', 'evaluation_ratio',
        'claim_consistency', 'figure_mentions', 'table_mentions', 'comparison_count', 
        'self_reference_count', 'method_limitation_ratio'
    ]
    
    # Load and process paper data from JSON
    try:
        # Specify the path to your papers JSON file
        papers_df = load_papers_from_json('cs_train.json')
        print(f"Loaded {len(papers_df)} papers")
        
        # Check class distribution
        print("Class distribution:")
        class_counts = papers_df['label'].value_counts()
        print(class_counts)
        
        # If there's insufficient data, create some dummy data for testing
        if len(papers_df) <= 3:  # Need at least 3 for train/val/test
            print("Warning: Not enough data found. Creating dummy data for testing.")
            # Create dummy data
            papers_df = pd.DataFrame({
                'text': ['This is a sample computer science research paper with accuracy = 0.95 and significant results.'] * 10,
                'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Dummy labels
            })
    except Exception as e:
        print(f"Error loading papers: {e}")
        # Generate dummy data for testing
        papers_df = pd.DataFrame({
            'text': ['This is a sample computer science research paper with accuracy = 0.95 and significant results.'] * 10,
            'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Dummy labels
        })
    
    # Extract handcrafted features with computer science-specific extractor
    print("Extracting computer science-specific handcrafted features...")
    extractor = ComputerScienceBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    # Convert to numpy arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    print(f"\nTotal papers loaded from cs_train.json: {len(papers_df)}")
    print(f"Class distribution in cs_train.json:")
    for i, count in enumerate(papers_df['label'].value_counts().sort_index()):
        class_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
        print(f"  {class_names[i]}: {count} papers ({count/len(papers_df)*100:.1f}%)")
    
    # ⚠️ CRITICAL CHECK: Verify we have balanced classes
    unique_labels = papers_df['label'].unique()
    print(f"Unique labels found: {sorted(unique_labels)}")
    if len(unique_labels) == 1:
        print("🚨 CRITICAL ISSUE: All labels are the same! Model will only learn one class.")
        print("This explains the 57% accuracy issue.")
    elif len(unique_labels) < 3:
        print(f"⚠️ WARNING: Only {len(unique_labels)} classes found. Missing some bias types.")
    else:
        print("✅ Good: All 3 classes present.")
    
    # Use entire cs_train.json for training and validation only
    # Split into training (80%) and validation (20%) sets
    
    # Get indices for each class for stratified splitting
    class_indices = {}
    for class_label in np.unique(labels_array):
        class_indices[class_label] = np.where(labels_array == class_label)[0]
    
    # Create stratified train/validation split (80/20)
    train_indices = []
    val_indices = []
    
    for class_label, indices in class_indices.items():
        np.random.shuffle(indices)  # Randomize order
        
        # Calculate split point (80% for training, 20% for validation)
        n_train = int(0.8 * len(indices))
        n_train = max(1, n_train)  # Ensure at least 1 sample for training
        
        # If class has only 1 sample, put it in training
        if len(indices) == 1:
            train_indices.extend(indices)
        else:
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])
    
    # Shuffle the final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Create train and validation sets (no test set - using separate cs_test.json)
    X_train = [(papers_df['text'].iloc[i], features_array[i]) for i in train_indices]
    y_train = labels_array[train_indices]
    
    X_val = [(papers_df['text'].iloc[i], features_array[i]) for i in val_indices]
    y_val = labels_array[val_indices]
    
    # Separate text and features
    train_texts, train_features = zip(*X_train) if X_train else ([], [])
    val_texts, val_features = zip(*X_val) if X_val else ([], [])
    
    # Convert to numpy arrays for SMOTE
    train_features = np.array(train_features)
    train_labels = np.array(y_train)
    
    print(f"Using entire cs_train.json for model training:")
    print(f"Training set size: {len(train_texts)} ({len(train_texts)/len(papers_df)*100:.1f}% of total)")
    print(f"Validation set size: {len(val_texts)} ({len(val_texts)/len(papers_df)*100:.1f}% of total)")
    print(f"Total papers used: {len(train_texts) + len(val_texts)} (100% of cs_train.json)")
    
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
                    synthetic_texts.append(f"Synthetic computer science research paper text for class {label}")
        
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
    
    # Use balanced model complexity for 4GB GPU - not too aggressive
    if device.type == 'cuda':
        print("Using balanced model complexity for 4GB GPU")
        max_seq_length = 64   # Reduced for speed
        max_sections = 2      # Reduced for speed  
        max_sents = 3         # Reduced for speed
        print(f"GPU settings: seq_len={max_seq_length}, sections={max_sections}, sents={max_sents}")
    else:
        print("Using reduced model complexity for CPU execution")
        max_seq_length = 128  # Better than 64 for learning
        max_sections = 3      # Better than 2 for learning
        max_sents = 6         # Better than 4 for learning
    
    # Create hierarchical datasets with reduced complexity
    print("Creating hierarchical datasets...")
    train_dataset = HierarchicalResearchPaperDataset(
        train_texts, train_labels, tokenizer, train_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    val_dataset = HierarchicalResearchPaperDataset(
        val_texts, y_val, tokenizer, np.array(val_features),
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # No test dataset - using separate cs_test.json for testing
    
    # Use memory-optimized batch size for 4GB GPU
    if device.type == 'cuda':
        batch_size = 4  # Larger batch size for better throughput
        print(f"Using batch size: {batch_size} for 4GB GPU (speed optimized)")
    else:
        batch_size = 2  # Smaller batch size for CPU
        print(f"Using batch size: {batch_size} for CPU processing")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # No test dataloader - using separate testing script
    
    # Initialize hierarchical model - force CPU execution
    print("Initializing hierarchical model for CPU execution...")
    try:
        print("Loading full hierarchical model")
        model = HierarchicalBiasPredictionModel(
            'allenai/scibert_scivocab_uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.3  # Standard dropout rate
        )
    except Exception as e:
        print(f"Error loading SciBERT for hierarchical model: {e}")
        print("Falling back to bert-base-uncased")
        model = HierarchicalBiasPredictionModel(
            'bert-base-uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.3
        )
    
    # Move model to device (GPU or CPU)
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Print memory usage after model loading
    print_memory_usage("After Model Loading")
    
    # Test a single forward pass to check memory
    if len(train_dataloader) > 0:
        print("\nTesting single forward pass...")
        model.eval()
        with torch.no_grad():
            try:
                test_batch = next(iter(train_dataloader))
                test_input_ids = test_batch['input_ids'].to(device)
                test_attention_mask = test_batch['attention_mask'].to(device)
                test_handcrafted_features = test_batch['handcrafted_features'].to(device)
                
                print(f"Test batch shapes: {test_input_ids.shape}, {test_attention_mask.shape}")
                
                # Try forward pass
                test_outputs = model(test_input_ids, test_attention_mask, test_handcrafted_features)
                print(f"Test forward pass successful! Output shape: {test_outputs.shape}")
                
                # Clear test tensors
                del test_batch, test_input_ids, test_attention_mask, test_handcrafted_features, test_outputs
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"ERROR in test forward pass: {e}")
                print("Model may be too large for available memory!")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        model.train()  # Back to training mode
    
    # Additional memory optimization for 4GB GPU
    if device.type == 'cuda':
        # Set memory growth to avoid pre-allocation
        torch.backends.cudnn.benchmark = False  # Disable for memory consistency
        torch.backends.cudnn.deterministic = True  # Enable for reproducibility
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    # Train hierarchical model
    print("Training hierarchical model...")
    try:
        if device.type == 'cuda':
            # GPU training parameters (optimized for 4GB GPU)
            print("Attempting GPU training with memory optimization...")
            model = train_hierarchical_model(
                train_dataloader, val_dataloader, model, device, 
                epochs=2,  # Reduced epochs for faster training
                accumulation_steps=8,  # Higher accumulation for effective larger batch size
                lr=2e-5  # Better learning rate
            )
        else:
            # CPU training parameters (fallback)
            print("Using CPU training...")
            model = train_hierarchical_model(
                train_dataloader, val_dataloader, model, device, 
                epochs=2,  # Reduced epochs for faster training
                accumulation_steps=4,  # Balanced accumulation for CPU
                lr=2e-5  # Better learning rate for CPU
            )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n{'='*50}")
            print("GPU OUT OF MEMORY ERROR DETECTED!")
            print(f"{'='*50}")
            print("Falling back to CPU training...")
            
            # Clear GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                del model
                torch.cuda.empty_cache()
            
            # Switch to CPU
            device = torch.device('cpu')
            
            # Recreate model on CPU
            try:
                model = HierarchicalBiasPredictionModel(
                    'allenai/scibert_scivocab_uncased', 
                    num_classes=3,
                    feature_dim=len(feature_names),
                    dropout_rate=0.3
                )
            except:
                model = HierarchicalBiasPredictionModel(
                    'bert-base-uncased', 
                    num_classes=3,
                    feature_dim=len(feature_names),
                    dropout_rate=0.3
                )
            
            model.to(device)
            print(f"Model moved to CPU. Retrying training...")
            
            # Retry training on CPU
            model = train_hierarchical_model(
                train_dataloader, val_dataloader, model, device, 
                epochs=2,  # Even fewer epochs for CPU fallback
                accumulation_steps=2,  # Smaller accumulation for CPU
                lr=1e-5
            )
        else:
            raise e  # Re-raise if it's not a memory error
    
    # Model training completed - saved as 'best_hierarchical_model.pt'
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Best model saved as: 'best_hierarchical_model.pt'")
    print(f"Model trained on {len(train_texts) + len(val_texts)} papers from cs_train.json")
    print(f"Training papers: {len(train_texts)}")
    print(f"Validation papers: {len(val_texts)}")
    print("\nUse your separate testing script with cs_test.json to evaluate the model.")
    print("="*50)

if __name__ == "__main__":
    # Set the Python multiprocessing method to avoid issues on Windows
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Initial GPU cleanup before any operations
    print("=" * 60)
    print("STARTING HIERARCHICAL BIAS PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("Performing initial GPU cleanup...")
        # Reset all GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # Wait for all operations to complete
        
        # Set environment variables for memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,roundup_power2_divisions:16'
        
        # Force garbage collection
        gc.collect()
        
        # Show initial GPU memory state
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print("GPU cleanup completed!")
    else:
        print("No GPU detected - will use CPU training")
    
    print("\nStarting main training process...\n")
    
    # Run the main function
    main()
