import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing for better generalization
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
class HierarchicalHumanitiesPaperDataset(Dataset):
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
        token_ids = torch.stack(token_ids)
        attention_masks = torch.stack(attention_masks)
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_masks,
            'handcrafted_features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _extract_sections(self, text):
        """Extract sections from the humanities paper text"""
        # Humanities-specific section markers
        section_markers = [
            'abstract', 'introduction', 'literature review', 'theoretical framework',
            'historical context', 'analysis', 'interpretation', 'discussion', 
            'methodology', 'findings', 'implications', 'conclusion', 'references'
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

# Enhanced feature extractor for humanities papers
class HumanitiesBiasFeatureExtractor:
    """Extract features that might indicate bias in humanities research papers"""
    
    def __init__(self):
        # Humanities-specific hedge words
        self.hedge_words = [
            'may', 'might', 'could', 'possibly', 'potentially', 'suggests', 
            'appears', 'seems', 'likely', 'unlikely', 'perhaps', 'arguably',
            'presumably', 'probably', 'apparently', 'supposedly', 'tend to',
            'tends to', 'tended to', 'indicate', 'indicates', 'implies'
        ]
        
        # Humanities-specific certainty words
        self.certainty_words = [
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'conclusively', 'absolutely', 'always', 'never', 'established',
            'proves', 'demonstrates', 'evident', 'unquestionably', 'indisputably',
            'undeniably', 'categorically', 'incontrovertibly'
        ]
        
        # Interpretive and subjective language
        self.interpretive_terms = [
            'interpretation', 'reading', 'perspective', 'lens', 'view',
            'understanding', 'approach', 'framework', 'paradigm', 'worldview',
            'ideology', 'discourse', 'narrative', 'story', 'account'
        ]
        
        # Critical theory and philosophical references
        self.theory_terms = [
            'foucault', 'derrida', 'butler', 'said', 'marxist', 'feminist',
            'postcolonial', 'postmodern', 'structuralist', 'poststructuralist',
            'hermeneutic', 'phenomenological', 'existentialist', 'critical theory',
            'cultural studies', 'discourse analysis', 'deconstruction'
        ]
        
        # Value-laden and normative language
        self.normative_terms = [
            'should', 'ought', 'must', 'need to', 'necessary', 'essential',
            'important', 'crucial', 'vital', 'imperative', 'obligation',
            'right', 'wrong', 'good', 'bad', 'better', 'worse', 'ideal'
        ]
        
        # Emotional and evaluative language
        self.emotional_terms = [
            'powerful', 'profound', 'disturbing', 'troubling', 'fascinating',
            'remarkable', 'striking', 'surprising', 'disappointing', 'impressive',
            'significant', 'meaningful', 'tragic', 'triumphant', 'compelling'
        ]
        
        # Citation and authority patterns
        self.authority_terms = [
            'according to', 'as noted by', 'as argued by', 'following',
            'building on', 'drawing on', 'influenced by', 'in line with',
            'contrary to', 'challenging', 'critiquing', 'extending'
        ]
        
        # Historiographical and source terms
        self.source_terms = [
            'archive', 'manuscript', 'document', 'text', 'artifact',
            'primary source', 'secondary source', 'evidence', 'testimony',
            'record', 'account', 'chronicle', 'inscription'
        ]
        
        # Comparative and contextual terms
        self.comparative_terms = [
            'similar', 'different', 'contrast', 'comparison', 'parallel',
            'analogous', 'distinct', 'unique', 'common', 'shared',
            'divergent', 'convergent', 'related', 'connected'
        ]
        
        # Methodological terms specific to humanities
        self.method_terms = [
            'close reading', 'textual analysis', 'discourse analysis',
            'historiography', 'ethnography', 'case study', 'archival research',
            'oral history', 'literary analysis', 'cultural analysis',
            'comparative method', 'hermeneutics', 'genealogy'
        ]
        
        # Stopwords for cleaning text
        self.stopwords = set(stopwords.words('english'))
        
    def extract_features(self, text):
        features = {}
        
        # Handle None or empty text
        if text is None or not text:
            # Return default values (25 features)
            return [0] * 25
        
        # Extract sections
        sections = self._extract_sections(text)
        
        # 1. Basic text statistics
        features['length'] = len(text)
        word_count = len(text.split()) + 1  # Add 1 to avoid division by zero
        features['avg_word_length'] = sum(len(w) for w in text.split()) / word_count
        
        # 2. Linguistic features - hedging and certainty
        hedge_count = sum(text.lower().count(word) for word in self.hedge_words)
        certainty_count = sum(text.lower().count(word) for word in self.certainty_words)
        features['hedge_ratio'] = hedge_count / word_count * 1000  # per 1000 words
        features['certainty_ratio'] = certainty_count / word_count * 1000
        
        # 3. Interpretive language
        interpretive_count = sum(text.lower().count(term) for term in self.interpretive_terms)
        features['interpretive_ratio'] = interpretive_count / word_count * 1000
        
        # 4. Theoretical and philosophical references
        theory_count = sum(text.lower().count(term) for term in self.theory_terms)
        features['theory_term_ratio'] = theory_count / word_count * 1000
        
        # 5. Normative language
        normative_count = sum(text.lower().count(term) for term in self.normative_terms)
        features['normative_ratio'] = normative_count / word_count * 1000
        
        # 6. Emotional and evaluative language
        emotional_count = sum(text.lower().count(term) for term in self.emotional_terms)
        features['emotional_ratio'] = emotional_count / word_count * 1000
        
        # 7. Citation and authority patterns
        authority_count = sum(text.lower().count(term) for term in self.authority_terms)
        features['authority_ratio'] = authority_count / word_count * 1000
        
        # 8. Source and evidence mentions
        source_count = sum(text.lower().count(term) for term in self.source_terms)
        features['source_ratio'] = source_count / word_count * 1000
        
        # 9. Comparative language
        comparative_count = sum(text.lower().count(term) for term in self.comparative_terms)
        features['comparative_ratio'] = comparative_count / word_count * 1000
        
        # 10. Methodological terms
        method_count = sum(text.lower().count(term) for term in self.method_terms)
        features['method_ratio'] = method_count / word_count * 1000
        
        # 11. First-person usage (subjectivity indicator)
        first_person_count = (text.lower().count(' i ') + text.lower().count(' my ') + 
                            text.lower().count(' we ') + text.lower().count(' our '))
        features['first_person_ratio'] = first_person_count / word_count * 1000
        
        # 12. Quotation usage (primary source engagement)
        quote_count = text.count('"') + text.count('"') + text.count('"')
        features['quotation_density'] = quote_count / word_count * 1000
        
        # 13. Section-specific features - Abstract
        if 'abstract' in sections:
            abstract = sections['abstract']
            abstract_words = len(abstract.split()) + 1
            abstract_interpretive = sum(abstract.lower().count(term) for term in self.interpretive_terms)
            features['abstract_interpretive_ratio'] = abstract_interpretive / abstract_words * 1000
        else:
            features['abstract_interpretive_ratio'] = 0
        
        # 14. Analysis section features
        if 'analysis' in sections or 'interpretation' in sections:
            analysis_section = sections.get('analysis', sections.get('interpretation', ''))
            analysis_words = len(analysis_section.split()) + 1
            analysis_certainty = sum(analysis_section.lower().count(word) for word in self.certainty_words)
            features['analysis_certainty_ratio'] = analysis_certainty / analysis_words * 1000
        else:
            features['analysis_certainty_ratio'] = 0
        
        # 15. Limitations acknowledgment
        features['limitations_mentioned'] = 1 if "limitation" in text.lower() or "limitations" in text.lower() else 0
        
        # 16. Multiple perspectives mentioned
        perspective_indicators = ['however', 'on the other hand', 'alternatively', 
                                'by contrast', 'conversely', 'different perspective']
        perspective_count = sum(text.lower().count(term) for term in perspective_indicators)
        features['perspective_diversity'] = perspective_count / word_count * 1000
        
        # 17. Claim-evidence balance
        claim_words = ['argues', 'claims', 'asserts', 'maintains', 'contends', 'proposes']
        evidence_words = ['evidence', 'demonstrates', 'shows', 'indicates', 'reveals', 'suggests']
        claim_count = sum(text.lower().count(word) for word in claim_words)
        evidence_count = sum(text.lower().count(word) for word in evidence_words)
        features['claim_count'] = claim_count
        features['evidence_count'] = evidence_count
        
        # 18. Citation density (et al., year patterns)
        citation_pattern = r'\(\d{4}\)|\(\w+,?\s+\d{4}\)|\[\d+\]'
        citation_count = len(re.findall(citation_pattern, text))
        features['citation_density'] = citation_count / word_count * 1000
        
        # 19. Temporal language (historical context)
        temporal_terms = ['century', 'period', 'era', 'age', 'epoch', 'time', 
                         'historical', 'contemporary', 'modern', 'ancient']
        temporal_count = sum(text.lower().count(term) for term in temporal_terms)
        features['temporal_ratio'] = temporal_count / word_count * 1000
        
        # 20. Abstract vs conclusion consistency
        if 'abstract' in sections and 'conclusion' in sections:
            abstract = sections['abstract']
            conclusion = sections['conclusion']
            abstract_claims = self._extract_key_claims(abstract)
            conclusion_claims = self._extract_key_claims(conclusion)
            features['claim_consistency'] = self._compare_claims(abstract_claims, conclusion_claims)
        else:
            features['claim_consistency'] = 0
        
        # 21. Self-reference patterns
        features['self_reference_count'] = (text.lower().count("my research") + 
                                        text.lower().count("my work") + 
                                        text.lower().count("my analysis") +
                                        text.lower().count("my interpretation") +
                                        text.lower().count("my argument"))
        
        # 22. Passive vs active voice (passive as objectivity indicator)
        passive_indicators = [' was ', ' were ', ' been ', ' being ', 'is considered', 'are regarded']
        passive_count = sum(text.lower().count(ind) for ind in passive_indicators)
        features['passive_voice_ratio'] = passive_count / word_count * 1000
        
        # 23. Interdisciplinary references
        discipline_terms = ['psychology', 'sociology', 'anthropology', 'philosophy',
                          'history', 'literature', 'linguistics', 'art', 'religion']
        interdisciplinary_count = sum(text.lower().count(term) for term in discipline_terms)
        features['interdisciplinary_ratio'] = interdisciplinary_count / word_count * 1000
        
        # 24. Footnote/endnote indicators
        footnote_count = text.count('[') + text.count('†') + text.count('‡')
        features['footnote_density'] = footnote_count / word_count * 1000
        
        # Return feature values as a list in a consistent order
        feature_values = [
            features['length'], 
            features['avg_word_length'],
            features['hedge_ratio'], 
            features['certainty_ratio'],
            features['interpretive_ratio'],
            features['theory_term_ratio'],
            features['normative_ratio'],
            features['emotional_ratio'],
            features['authority_ratio'],
            features['source_ratio'],
            features['comparative_ratio'],
            features['method_ratio'],
            features['first_person_ratio'],
            features['quotation_density'],
            features['abstract_interpretive_ratio'],
            features['analysis_certainty_ratio'],
            features['limitations_mentioned'],
            features['perspective_diversity'],
            features['claim_count'],
            features['evidence_count'],
            features['citation_density'],
            features['temporal_ratio'],
            features['claim_consistency'],
            features['self_reference_count'],
            features['passive_voice_ratio'],
            features['interdisciplinary_ratio'],
            features['footnote_density']
        ]
        
        return feature_values[:25]  # Ensure exactly 25 features
    
    def _extract_sections(self, text):
        """Extract sections from the humanities paper text"""
        section_dict = {}
        section_markers = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', 'i. introduction'],
            'literature': ['literature', 'literature review', 'related work', 'scholarship'],
            'theoretical': ['theoretical framework', 'theory', 'theoretical background'],
            'historical': ['historical context', 'background', 'history'],
            'methods': ['method', 'methods', 'methodology', 'approach'],
            'analysis': ['analysis', 'interpretation', 'reading', 'examination'],
            'discussion': ['discussion', 'implications'],
            'conclusion': ['conclusion', 'conclusions', 'final thoughts', 'concluding remarks']
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
        """Extract key claims from text"""
        claim_sentences = []
        sentences = sent_tokenize(text)
        
        claim_indicators = ['argues', 'claims', 'asserts', 'maintains', 'contends', 
                          'proposes', 'suggests', 'demonstrates', 'shows']
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in claim_indicators):
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
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, bert_embedding, handcrafted_features):
        # Project inputs
        bert_proj = self.bert_projection(bert_embedding)
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)

        # Apply batch normalization if 2D
        if bert_proj.dim() == 2:
            bert_proj = self.bert_bn(bert_proj)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj)
            bert_proj = bert_proj.unsqueeze(1)
            handcrafted_proj = handcrafted_proj.unsqueeze(1)
        elif bert_proj.dim() == 3:
            batch_size, seq_len, dim = bert_proj.shape
            bert_proj = bert_proj.view(-1, dim)
            bert_proj = self.bert_bn(bert_proj)
            bert_proj = bert_proj.view(batch_size, seq_len, dim)
            
            handcrafted_proj = handcrafted_proj.view(-1, dim)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj)
            handcrafted_proj = handcrafted_proj.view(batch_size, seq_len, dim)

        # Apply attention
        attn_output, _ = self.attention(
            bert_proj, handcrafted_proj, handcrafted_proj
        )

        # Remove seq_len dim if it is 1
        attn_output = attn_output.squeeze(1)
        bert_proj = bert_proj.squeeze(1)
        handcrafted_proj = handcrafted_proj.squeeze(1)

        # Gated fusion
        combined = torch.cat([bert_proj, handcrafted_proj], dim=-1)
        gate_values = self.gate(combined)
        fused = gate_values * bert_proj + (1 - gate_values) * handcrafted_proj
        output = self.layer_norm(fused)
        output = self.dropout(output)

        return output

class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention network for document classification
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
                    words = word_embeddings[b, s, i]
                    mask = attention_mask[b, s, i]

                    if words.dim() == 3 and words.shape[0] == 1:
                        words = words.squeeze(0)
                    if mask.sum() > 0:
                        words = words.unsqueeze(1)
                        key_padding_mask = (mask == 0).unsqueeze(0)

                        attn_output, _ = self.word_attention(
                            words, words, words,
                            key_padding_mask=key_padding_mask
                        )
                        attn_output = attn_output.squeeze(1)

                        sent_embedding = (attn_output * mask.unsqueeze(-1)).sum(dim=0) / (mask.sum() + 1e-10)
                    else:
                        sent_embedding = torch.zeros(hidden_dim, device=words.device)

                    sent_embeddings.append(sent_embedding)

                if sent_embeddings:
                    section_sent_embeddings = torch.stack(sent_embeddings)
                    sent_mask = (attention_mask[b, s].sum(dim=1) > 0).float()

                    if sent_mask.sum() > 0:
                        section_sent_embeddings = section_sent_embeddings.unsqueeze(1)
                        key_padding_mask = (sent_mask == 0).unsqueeze(0)

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
                doc_section_embeddings = torch.stack(sentence_embeddings)
                section_mask = (attention_mask[b].sum(dim=(1, 2)) > 0).float()

                if section_mask.sum() > 0:
                    doc_section_embeddings = doc_section_embeddings.unsqueeze(1)
                    key_padding_mask = (section_mask == 0).unsqueeze(0)

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

# Memory-optimized forward pass
def forward_with_memory_optimization(self, input_ids, attention_mask, handcrafted_features):
    batch_size = input_ids.size(0)
    max_sections, max_sents, max_seq_length = input_ids.size(1), input_ids.size(2), input_ids.size(3)
    
    # Process each sentence with BERT
    word_embeddings = []
    for b in range(batch_size):
        section_embeddings = []
        for s in range(max_sections):
            sentence_embeddings = []
            for i in range(max_sents):
                input_ids_sent = input_ids[b, s, i].unsqueeze(0)
                attention_mask_sent = attention_mask[b, s, i].unsqueeze(0)
                
                if attention_mask_sent.sum() > 0:
                    with torch.no_grad():
                        outputs = self.bert(
                            input_ids=input_ids_sent, 
                            attention_mask=attention_mask_sent,
                            output_hidden_states=True
                        )
                        token_embeddings = outputs.last_hidden_state.squeeze(0).detach()
                        del outputs
                else:
                    token_embeddings = torch.zeros(
                        max_seq_length, self.bert_dim, device=input_ids.device)
                
                sentence_embeddings.append(token_embeddings)
            
            if sentence_embeddings:
                section_embeddings.append(torch.stack(sentence_embeddings))
            else:
                empty_section = torch.zeros(
                    max_sents, max_seq_length, self.bert_dim, device=input_ids.device)
                section_embeddings.append(empty_section)
        
        if section_embeddings:
            word_embeddings.append(torch.stack(section_embeddings))
        else:
            empty_doc = torch.zeros(
                max_sections, max_sents, max_seq_length, self.bert_dim, device=input_ids.device)
            word_embeddings.append(empty_doc)
    
    word_embeddings = torch.stack(word_embeddings)
    
    # Apply hierarchical attention
    doc_embeddings = []
    for b in range(batch_size):
        doc_embedding = self.hierarchical_attention(
            word_embeddings[b].unsqueeze(0),
            attention_mask[b].unsqueeze(0)
        )
        doc_embeddings.append(doc_embedding)
    
    del word_embeddings
    doc_embeddings = torch.cat(doc_embeddings, dim=0)
    
    # Feature fusion
    fused_features = self.feature_fusion(doc_embeddings, handcrafted_features)
    del doc_embeddings
    
    # Classification with residual connection
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

# Hierarchical Bias Prediction Model
class HierarchicalBiasPredictionModel(nn.Module):
    def __init__(self, bert_model_name, num_classes=3, dropout_rate=0.4, feature_dim=25):
        super(HierarchicalBiasPredictionModel, self).__init__()
        
        print("Loading BERT model...")
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        self.dropout = nn.Dropout(dropout_rate)
        
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
        
        # Classification layers with batch normalization
        self.fc1 = nn.Linear(self.fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Residual connection
        self.shortcut = nn.Linear(self.fusion_dim, 128) if self.fusion_dim != 128 else nn.Identity()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Higher dropout for small dataset
        
        # Use memory-optimized forward
        self.forward = forward_with_memory_optimization.__get__(self)

# Training function optimized for small dataset (371 papers)
def train_hierarchical_model(train_dataloader, val_dataloader, model, device, 
                           epochs=10, lr=5e-5, accumulation_steps=4):
    """Train the hierarchical model - optimized for small dataset"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    
    # Cosine annealing with warm restarts for small dataset
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )
    
    # Backup plateau scheduler
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Calculate class weights
    train_labels = []
    for batch in train_dataloader:
        train_labels.extend(batch['label'].cpu().numpy())
    
    class_counts = torch.bincount(torch.tensor(train_labels))
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    
    print(f"Class distribution: {class_counts.cpu().numpy()}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Use Focal Loss without label smoothing for small dataset
    loss_fn = FocalLoss(alpha=1, gamma=2, weight=class_weights, label_smoothing=0.0)
    
    best_val_accuracy = 0
    best_model = None
    patience = 10  # Higher patience for small dataset
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        max_grad_norm = 1.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            handcrafted_features = batch['handcrafted_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, handcrafted_features)
            loss = loss_fn(outputs, labels)
            loss = loss / accumulation_steps
            
            loss.backward()
            train_loss += loss.item() * accumulation_steps
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        if len(train_dataloader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
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
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss = {train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}, '
              f'LR = {current_lr:.6f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict().copy()
            torch.save(best_model, 'best_hierarchical_humanities_model.pt')
            patience_counter = 0
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Clear cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    
    # Load best model
    model.load_state_dict(best_model)
    return model

# Evaluation function
def evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names=None):
    """Evaluate model on test set"""
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
    
    present_labels = sorted(set(np.concatenate([unique_preds, unique_labels])))
    filtered_label_names = [label_names[i] for i in present_labels] if present_labels else label_names[:1]
    
    try:
        report = classification_report(all_labels, all_preds, 
                               target_names=filtered_label_names,
                               labels=present_labels,
                               output_dict=True)
        
        print(classification_report(all_labels, all_preds, 
                               target_names=filtered_label_names,
                               labels=present_labels))
        
        # Plot classification report
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
        plt.title('Humanities Research Bias Detection Performance')
        plt.tight_layout()
        plt.savefig('humanities_classification_performance.png')
        plt.close()
        
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0
        print(f"Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    if len(present_labels) > 1:
        cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Humanities Research Bias Detection - Confusion Matrix')
        plt.savefig('humanities_confusion_matrix.png')
        plt.close()
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Humanities Research Bias Detection - Normalized Confusion Matrix')
        plt.savefig('humanities_normalized_confusion_matrix.png')
        plt.close()
    
    return all_preds, all_labels

def load_papers_from_json(json_file_path):
    """Load humanities research papers from a JSON file"""
    import pandas as pd
    import json
    import os
    
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found.")
        return pd.DataFrame(columns=['text', 'label'])
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = []
        
        if isinstance(data, list):
            for i, paper in enumerate(data):
                if not isinstance(paper, dict):
                    continue
                    
                text = paper.get('Body', paper.get('text', paper.get('content', paper.get('abstract', ''))))
                
                label = None
                for key in ['Overall Bias', 'OverallBias', 'Overall_Bias']:
                    if key in paper:
                        label = paper.get(key)
                        break
                if label is None:
                    label = paper.get('label', paper.get('bias_label', paper.get('bias_type', 0)))
                
                if label is None:
                    for k, v in paper.items():
                        if k.replace('_', '').replace(' ', '').lower() == 'overallbias':
                            label = v
                            break
                
                if isinstance(label, str):
                    label_normal = label.strip().lower()
                    label_map = {
                        'no_bias': 0, 'none': 0, 'no bias': 0, 'nobias': 0, '0': 0,
                        'cognitive_bias': 1, 'cognitive': 1, 'cognitive bias': 1, 'cognitivebias': 1, '1': 1,
                        'publication_bias': 2, 'publication': 2, 'publication bias': 2, 'publicationbias': 2, '2': 2,
                        'selection_bias': 2, 'selection': 2, 'selection bias': 2
                    }
                    label = label_map.get(label_normal, 0)
                
                if isinstance(label, float):
                    label = int(label)
                
                papers.append({'text': text, 'label': label})
                
        elif isinstance(data, dict):
            if 'papers' in data and isinstance(data['papers'], list):
                papers_data = data['papers']
                for paper in papers_data:
                    if not isinstance(paper, dict):
                        continue
                    text = paper.get('text', paper.get('content', ''))
                    label = paper.get('label', paper.get('Overall Bias', 0))
                    
                    if isinstance(label, str):
                        label_map = {
                            'no_bias': 0, 'none': 0, '0': 0,
                            'cognitive_bias': 1, 'cognitive': 1, '1': 1,
                            'publication_bias': 2, 'publication': 2, '2': 2
                        }
                        label = label_map.get(label.strip().lower(), 0)
                    
                    if isinstance(label, float):
                        label = int(label)
                    
                    papers.append({'text': text, 'label': label})
            else:
                for paper_id, paper_data in data.items():
                    if isinstance(paper_data, dict):
                        text = paper_data.get('text', '')
                        label = paper_data.get('label', 0)
                        papers.append({'text': text, 'label': label})
        
        df = pd.DataFrame(papers)
        
        if len(df) == 0:
            return pd.DataFrame(columns=['text', 'label'])
        
        if 'text' not in df.columns:
            df['text'] = ""
        if 'label' not in df.columns:
            df['label'] = 0
        
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        
        original_count = len(df)
        df = df[df['text'].str.strip().str.len() > 0].reset_index(drop=True)
        if len(df) < original_count:
            print(f"Info: Removed {original_count - len(df)} rows with empty text")
        
        if not all(df['label'].isin([0, 1, 2])):
            invalid_labels = df[~df['label'].isin([0, 1, 2])]['label'].unique()
            print(f"Warning: Found invalid label values: {invalid_labels}. Converting to 0.")
            df.loc[~df['label'].isin([0, 1, 2]), 'label'] = 0
        
        print(f"Successfully loaded {len(df)} humanities papers from {json_file_path}")
        return df
        
    except Exception as e:
        print(f"Error loading papers: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])

# Feature importance analysis
def analyze_humanities_feature_importance(model, feature_names):
    """Analyze which humanities-specific features are most important"""
    weights = model.feature_fusion.handcrafted_projection.weight.detach().cpu().numpy()
    importance = np.linalg.norm(weights, axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Humanities-Specific Feature Importance for Bias Detection')
    plt.tight_layout()
    plt.savefig('humanities_feature_importance.png')
    plt.close()
    
    # Category importance
    feature_categories = {
        'Basic Text': ['length', 'avg_word_length'],
        'Linguistic': ['hedge_ratio', 'certainty_ratio', 'passive_voice_ratio'],
        'Interpretive': ['interpretive_ratio', 'first_person_ratio'],
        'Theoretical': ['theory_term_ratio', 'interdisciplinary_ratio'],
        'Normative': ['normative_ratio', 'emotional_ratio'],
        'Citation': ['authority_ratio', 'citation_density', 'footnote_density'],
        'Evidence': ['source_ratio', 'quotation_density', 'claim_count', 'evidence_count'],
        'Methodology': ['method_ratio', 'comparative_ratio'],
        'Quality': ['limitations_mentioned', 'perspective_diversity', 'claim_consistency'],
        'Context': ['temporal_ratio', 'self_reference_count']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        indices = [feature_names.index(f) for f in features if f in feature_names]
        if indices:
            category_importance[category] = importance[indices].mean()
    
    category_df = pd.DataFrame({
        'Category': list(category_importance.keys()),
        'Average Importance': list(category_importance.values())
    }).sort_values('Average Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Average Importance', y='Category', data=category_df)
    plt.title('Feature Category Importance for Humanities Research Bias Detection')
    plt.tight_layout()
    plt.savefig('humanities_feature_category_importance.png')
    plt.close()
    
    return importance_df, category_df

def main():
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature names for humanities-specific extractor (25 features)
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
    
    # Add interdisciplinary_ratio and footnote_density to make 27, then trim to 25
    feature_names = feature_names[:23] + ['interdisciplinary_ratio', 'footnote_density']
    
    # Load papers
    try:
        papers_df = load_papers_from_json('/kaggle/input/humanities/humanities_papers.json')
        print(f"Loaded {len(papers_df)} humanities papers")
        
        print("Class distribution:")
        class_counts = papers_df['label'].value_counts()
        print(class_counts)
        
        if len(papers_df) <= 3:
            print("Warning: Not enough data. Creating dummy data.")
            papers_df = pd.DataFrame({
                'text': ['This is a sample humanities research paper analyzing historical texts.'] * 10,
                'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
            })
    except Exception as e:
        print(f"Error loading papers: {e}")
        papers_df = pd.DataFrame({
            'text': ['This is a sample humanities research paper analyzing historical texts.'] * 10,
            'label': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        })
    
    # Extract features
    print("Extracting humanities-specific handcrafted features...")
    extractor = HumanitiesBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    print("Features normalized using StandardScaler")
    
    # Split data
    class_indices = {}
    for class_label in np.unique(labels_array):
        class_indices[class_label] = np.where(labels_array == class_label)[0]
    
    test_indices = []
    for class_label, indices in class_indices.items():
        if len(indices) > 1:
            n_test = max(1, int(0.2 * len(indices)))
            test_indices.extend(np.random.choice(indices, n_test, replace=False))
    
    all_indices = set(range(len(labels_array)))
    train_val_indices = list(all_indices - set(test_indices))
    
    n_val = max(1, int(0.2 * len(train_val_indices)))
    val_indices = np.random.choice(train_val_indices, n_val, replace=False)
    train_indices = list(set(train_val_indices) - set(val_indices))
    
    X_train = [(papers_df['text'].iloc[i], features_array[i]) for i in train_indices]
    y_train = labels_array[train_indices]
    
    X_val = [(papers_df['text'].iloc[i], features_array[i]) for i in val_indices]
    y_val = labels_array[val_indices]
    
    X_test = [(papers_df['text'].iloc[i], features_array[i]) for i in test_indices]
    y_test = labels_array[test_indices]
    
    train_texts, train_features = zip(*X_train) if X_train else ([], [])
    val_texts, val_features = zip(*X_val) if X_val else ([], [])
    test_texts, test_features = zip(*X_test) if X_test else ([], [])
    
    train_features = np.array(train_features)
    train_labels = np.array(y_train)
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Apply SMOTE (keeping original implementation)
    print("Applying SMOTE to balance classes...")
    try:
        original_counts = Counter(train_labels)
        print("Original class distribution:", original_counts)
        
        min_samples_per_class = min(original_counts.values()) if original_counts else 0
        
        if min_samples_per_class >= 6:
            smote = SMOTE(random_state=42)
            resampled_features, resampled_labels = smote.fit_resample(train_features, train_labels)
        else:
            print(f"Found class with only {min_samples_per_class} samples. Using custom balancing.")
            
            max_count = max(original_counts.values()) if original_counts else 0
            target_count = max_count
            
            resampled_features = []
            resampled_labels = []
            
            for class_label in np.unique(train_labels):
                class_indices = np.where(train_labels == class_label)[0]
                class_count = len(class_indices)
                
                if class_count >= 5:
                    try:
                        k_neighbors = min(class_count - 1, 5)
                        if k_neighbors >= 1:
                            smote_single = SMOTE(k_neighbors=k_neighbors, random_state=42)
                            
                            temp_labels = np.zeros(len(train_labels))
                            temp_labels[class_indices] = 1
                            
                            other_indices = np.random.choice(
                                np.where(temp_labels == 0)[0], 
                                min(len(class_indices) * 2, len(train_labels) - len(class_indices)),
                                replace=False
                            )
                            
                            temp_indices = np.concatenate([class_indices, other_indices])
                            temp_features = train_features[temp_indices]
                            temp_class_labels = temp_labels[temp_indices]
                            
                            resampled_temp_features, resampled_temp_labels = smote_single.fit_resample(
                                temp_features, temp_class_labels)
                            
                            synthetic_indices = np.where(
                                (resampled_temp_labels == 1) & 
                                (np.arange(len(resampled_temp_labels)) >= len(temp_features))
                            )[0]
                            
                            synthetic_features = resampled_temp_features[synthetic_indices]
                            samples_needed = target_count - class_count
                            
                            if samples_needed > 0 and len(synthetic_features) > 0:
                                if samples_needed > len(synthetic_features):
                                    indices = np.random.choice(
                                        len(synthetic_features), samples_needed, replace=True)
                                else:
                                    indices = np.random.choice(
                                        len(synthetic_features), samples_needed, replace=False)
                                
                                selected_synthetic_features = synthetic_features[indices]
                                
                                resampled_features.extend(train_features[class_indices])
                                resampled_features.extend(selected_synthetic_features)
                                resampled_labels.extend([class_label] * class_count)
                                resampled_labels.extend([class_label] * samples_needed)
                            else:
                                resampled_features.extend(train_features[class_indices])
                                resampled_labels.extend([class_label] * class_count)
                        else:
                            resampled_features.extend(train_features[class_indices])
                            resampled_labels.extend([class_label] * class_count)
                            
                            samples_needed = target_count - class_count
                            if samples_needed > 0:
                                indices = np.random.choice(class_count, samples_needed, replace=True)
                                selected_features = train_features[class_indices][indices]
                                resampled_features.extend(selected_features)
                                resampled_labels.extend([class_label] * samples_needed)
                    
                    except Exception as e:
                        print(f"Error applying SMOTE to class {class_label}: {e}")
                        resampled_features.extend(train_features[class_indices])
                        resampled_labels.extend([class_label] * class_count)
                else:
                    resampled_features.extend(train_features[class_indices])
                    resampled_labels.extend([class_label] * class_count)
                    
                    samples_needed = target_count - class_count
                    if samples_needed > 0:
                        indices = np.random.choice(class_count, samples_needed, replace=True)
                        selected_features = train_features[class_indices][indices]
                        resampled_features.extend(selected_features)
                        resampled_labels.extend([class_label] * samples_needed)
        
            resampled_features = np.array(resampled_features)
            resampled_labels = np.array(resampled_labels)
        
        # Generate synthetic texts
        label_to_texts = {}
        for text, label in zip(train_texts, train_labels):
            if label not in label_to_texts:
                label_to_texts[label] = []
            label_to_texts[label].append(text)
        
        new_counts = Counter(resampled_labels)
        
        synthetic_texts = []
        for i, label in enumerate(resampled_labels):
            if i < len(train_texts):
                synthetic_texts.append(train_texts[i])
            else:
                if label in label_to_texts and label_to_texts[label]:
                    synthetic_texts.append(np.random.choice(label_to_texts[label]))
                else:
                    synthetic_texts.append(f"Synthetic humanities research paper text for class {label}")
        
        train_texts = synthetic_texts
        train_features = resampled_features
        train_labels = resampled_labels
        
        print("After balancing class distribution:", new_counts)
        
    except Exception as e:
        print(f"Error in data balancing process: {e}")
        print("Proceeding with original imbalanced data")
    
    # Clear memory
    del papers_df, handcrafted_features, features_array, labels_array
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    try:
        model_name = 'allenai/scibert_scivocab_uncased'
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Reduced complexity for small dataset
    print(f"Using reduced model complexity for small dataset")
    max_seq_length = 64
    max_sections = 2
    max_sents = 4
    
    # Create datasets
    print("Creating hierarchical datasets...")
    train_dataset = HierarchicalHumanitiesPaperDataset(
        train_texts, train_labels, tokenizer, train_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    val_dataset = HierarchicalHumanitiesPaperDataset(
        val_texts, y_val, tokenizer, val_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    test_dataset = HierarchicalHumanitiesPaperDataset(
        test_texts, y_test, tokenizer, test_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Small batch size for small dataset
    batch_size = 4
    print(f"Using batch size: {batch_size} for small dataset ({len(train_texts)} samples)")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    print("Initializing hierarchical model...")
    try:
        model = HierarchicalBiasPredictionModel(
            'allenai/scibert_scivocab_uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.5
        )
    except Exception as e:
        print(f"Error loading SciBERT: {e}")
        print("Falling back to bert-base-uncased")
        model = HierarchicalBiasPredictionModel(
            'bert-base-uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.5
        )
    
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Train model - optimized for small dataset
    print("Training hierarchical model...")
    model = train_hierarchical_model(
        train_dataloader, val_dataloader, model, device, 
        epochs=10,  # More epochs for small dataset
        accumulation_steps=4,  # Higher accumulation for stability
        lr=2e-4  # Lower learning rate for small dataset
    )
    
    # Evaluate model
    print("Evaluating hierarchical model...")
    label_names = ['No Bias', 'Cognitive Bias', 'Selection/Publication Bias']
    evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names)
    
    # Analyze feature importance
    print("Analyzing humanities-specific feature importance...")
    analyze_humanities_feature_importance(model, feature_names)
    
    print("Humanities bias detection analysis complete!")

if __name__ == "__main__":
    # Set multiprocessing method
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Run main
    main()
