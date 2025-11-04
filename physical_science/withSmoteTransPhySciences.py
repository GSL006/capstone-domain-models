import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
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
import textstat
from tqdm.auto import tqdm

# Download NLTK resources if needed
def download_nltk_data():
    """Download NLTK resources if they are not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK 'stopwords'...")
        nltk.download('stopwords')
    try:
        # This resource is sometimes needed by sent_tokenize for certain languages/edge cases
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK 'punkt_tab'...")
        nltk.download('punkt_tab')

# Focal Loss for handling class imbalance and hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Can be a tensor of class weights
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)

        # Apply class-specific alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            else:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Centralized function for section extraction
def extract_sections(text):
    """
    Extracts sections from a research paper text.
    Returns a dictionary mapping section keys to their content.
    """
    section_dict = {}
    section_markers = {
        'abstract': ['abstract', 'executive summary', 'summary'],
        'introduction': ['introduction', '1. introduction', 'i. introduction', 'background'],
        'literature': ['literature review', 'theoretical framework', 'related work'],
        'methods': ['methodology', 'methods', 'research design', 'data and methods', 'materials and methods'],
        'analysis': ['analysis', 'data analysis', 'statistical analysis'],
        'results': ['results', 'findings', 'empirical results'],
        'discussion': ['discussion', 'implications', 'practical implications'],
        'conclusion': ['conclusion', 'conclusions', 'recommendations', 'concluding remarks']
    }

    text_lower = text.lower()
    
    # Use regex to split the text by section headers. This is more robust.
    # The pattern looks for a newline, followed by a marker, then optional punctuation and another newline.
    all_markers = [item for sublist in section_markers.values() for item in sublist]
    # Sort by length to match longer markers first (e.g., 'literature review' before 'literature')
    all_markers.sort(key=len, reverse=True)
    
    pattern = r'\n(' + '|'.join(re.escape(m) for m in all_markers) + r')[:.\s]*\n'
    
    # Split the text by the pattern
    content_splits = re.split(pattern, text_lower, flags=re.IGNORECASE)
    
    # The first item is the text before any sections (usually empty or part of intro)
    # The subsequent items are pairs of (marker, content)
    if len(content_splits) > 1:
        # Assign content to the correct section key
        for i in range(1, len(content_splits), 2):
            marker_found = content_splits[i].lower()
            content = content_splits[i+1]
            for key, marker_list in section_markers.items():
                if marker_found in marker_list:
                    section_dict[key] = content.strip()
                    break


    # If no sections were found, treat the entire text as one section under 'introduction'
    if not section_dict:
        section_dict['introduction'] = text

    return section_dict
# Enhanced dataset class for hierarchical text processing
class HierarchicalPaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, handcrafted_features, 
                 max_seq_length=256, max_sections=8, max_sents=12):
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
        
        # Use the centralized section extractor
        sections_dict = extract_sections(text)
        # We only need the text content for tokenization
        sections = list(sections_dict.values())

        # Truncate/pad sections to max_sections
        sections = sections[:self.max_sections]
        
        all_sentences = []
        for section in sections:
            sents = sent_tokenize(section)
            # Truncate/pad sentences within each section
            sents = sents[:self.max_sents]
            sents += [""] * (self.max_sents - len(sents))
            all_sentences.extend(sents)

        # Pad sections if there were fewer than max_sections
        num_pad_sections = self.max_sections - len(sections)
        if num_pad_sections > 0:
            all_sentences.extend([""] * (num_pad_sections * self.max_sents))

        # Tokenize all sentences in one batch - this is much faster
        encoding = self.tokenizer(
            all_sentences,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Reshape into hierarchical structure: [max_sections, max_sents, max_seq_length]
        token_ids = encoding['input_ids'].view(self.max_sections, self.max_sents, self.max_seq_length)
        attention_masks = encoding['attention_mask'].view(self.max_sections, self.max_sents, self.max_seq_length)
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_masks,
            'handcrafted_features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Enhanced feature extractor for physical science papers
class PhysicalScienceFeatureExtractor:
    """Extract features that might indicate bias in physical science research papers"""
    
    def __init__(self):
        # Basic patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.0\d+'
        self.significance_stars = r'\*{1,3}\s*'
        self.correlation_pattern = r'(?:correlation|corr|r)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+'
        self.percentage_pattern = r'\d+(?:\.\d+)?%'
        
        # General hedge words (applicable across sciences)
        self.hedge_words = [ # Kept from original
            'may', 'might', 'could', 'possibly', 'potentially', 'suggests', 
            'appears', 'seems', 'likely', 'unlikely', 'perhaps', 'arguably',
            'tend to', 'tends to', 'tended to', 'indicate', 'indicates',
            'presumably', 'probably', 'apparently', 'supposedly'
        ]
        
        # Social science-specific certainty words
        self.certainty_words = [ # Kept from original
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'conclusively', 'absolutely', 'always', 'never', 'established',
            'proves', 'demonstrates', 'significant', 'substantial',
            'strong evidence', 'strongly supports', 'decisive', 'confirmed',
            'will result in', 'leads to'
        ]
        
        # Physical science theory and framework references
        self.theory_terms = [
            'quantum mechanics', 'general relativity', 'special relativity', 'newtonian',
            'thermodynamics', 'electromagnetism', 'standard model', 'string theory',
            'big bang', 'inflation', 'dark matter', 'dark energy', 'wave-particle duality',
            'uncertainty principle', 'schrödinger equation', 'density functional theory', 'dft'
        ]
        
        # Physical science-specific claim words
        self.claim_terms = [
            'impact', 'effect', 'influence', 'relationship', 'correlation',
            'causation', 'implication', 'mechanism', 'property', 'constant',
            'law', 'principle', 'model', 'simulation', 'measurement', 'discovery',
            'observation', 'evidence'
        ]
        
        # Overly confident or promotional terms (replaces business jargon)
        self.promotional_terms = [
            'paradigm shift', 'breakthrough', 'revolutionary', 'game-changer',
            'transformational', 'cutting-edge', 'state-of-the-art', 'novel',
            'unprecedented', 'first-ever', 'world-first'
        ]
        
        # Methodology and validation terms for physical sciences
        self.method_terms = [
            'spectroscopy', 'chromatography', 'x-ray diffraction', 'xrd', 'nmr',
            'mass spectrometry', 'microscopy', 'sem', 'tem', 'afm',
            'simulation', 'monte carlo', 'finite element', 'computational',
            'experiment', 'observation', 'analysis', 'regression', 'calibration',
            'quantitative'
        ]
        
        # Physical science-specific validation patterns
        self.validation_patterns = [
            r'confidence interval', r'margin of error', r'uncertainty',
            r'error analysis', r'error propagation', r'signal-to-noise', r's/n',
            r'calibration', r'reproducibility', r'validation', 'benchmark'
        ]
        
        # Stopwords for cleaning text
        self.stopwords = set(stopwords.words('english'))
        
    def extract_features(self, text):
        features = {}
        
        # Handle None or empty text
        if text is None or not text:
            # Return default zero values for all features
            return [0] * 29  # Updated feature count with new features
        
        # Extract sections
        sections = extract_sections(text)
        
        # 1. Basic text statistics
        features['length'] = len(text)
        word_count = len(text.split()) + 1  # Add 1 to avoid division by zero
        sentences = sent_tokenize(text)
        sentence_count = len(sentences) + 1
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
        
        # 4. Field-specific patterns
        theory_count = sum(text.lower().count(term) for term in self.theory_terms)
        promo_count = sum(text.lower().count(term) for term in self.promotional_terms)
        features['theory_term_ratio'] = theory_count / word_count * 1000
        features['promo_term_ratio'] = promo_count / word_count * 1000
        
        # 5. Methodology patterns
        method_count = sum(text.lower().count(term) for term in self.method_terms)
        validation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in self.validation_patterns)
        features['method_term_count'] = method_count
        features['validation_pattern_count'] = validation_count
        
        # 6. Section-specific features
        # Abstract features (claims without evidence)
        abstract = sections.get('abstract', '')
        abstract_words = len(abstract.split()) + 1
        abstract_claim_count = sum(abstract.lower().count(term) for term in self.claim_terms)
        features['abstract_claim_ratio'] = abstract_claim_count / abstract_words * 1000
        
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
        
        # 7. Measurement/efficiency claims
        efficiency_terms = ['efficiency', 'yield', 'performance', 'improvement', 'enhancement', 'optimization']
        efficiency_count = sum(text.lower().count(term) for term in efficiency_terms)
        features['efficiency_claim_ratio'] = efficiency_count / word_count * 1000
        
        # 8. Abstract vs conclusion claim consistency
        abstract_text = sections.get('abstract', '')
        conclusion_text = sections.get('conclusion', sections.get('discussion', ''))
        abstract_claims = self._extract_key_claims(abstract_text)
        conclusion_claims = self._extract_key_claims(conclusion_text)
        features['claim_consistency'] = self._compare_claims(abstract_claims, conclusion_claims)
            
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
        
        # NEW FEATURES (23-29): Readability and complexity metrics
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        except:
            features['flesch_reading_ease'] = 50.0  # Default mid-range value
            features['flesch_kincaid_grade'] = 10.0

        # Sentence length variance (complexity indicator)
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_length_variance'] = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        features['avg_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0

        # Citation density
        citation_pattern = r'\[\d+\]|\(\d{4}\)|et al\.'
        citation_count = len(re.findall(citation_pattern, text))
        features['citation_density'] = citation_count / word_count * 1000

        # Methodology section strength (more detailed methods = better quality)
        methods_section = sections.get('methods', '')
        methods_words = len(methods_section.split())
        features['methodology_detail_ratio'] = methods_words / word_count if methods_words > 0 else 0

        # Data availability statement
        features['data_availability'] = 1 if any(phrase in text.lower() for phrase in 
                                                 ['data available', 'data are available', 
                                                  'available upon request', 'supplementary data']) else 0
        
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
            features['promo_term_ratio'],
            features['method_term_count'],
            features['validation_pattern_count'],
            features['abstract_claim_ratio'],
            features['results_stat_density'],
            features['limitations_mentioned'],
            features['efficiency_claim_ratio'],
            features['claim_consistency'],
            features['figure_mentions'],
            features['table_mentions'],
            features['chart_mentions'],
            features['citation_but_count'],
            features['self_reference_count'],
            # New features
            features['flesch_reading_ease'],
            features['flesch_kincaid_grade'],
            features['sentence_length_variance'],
            features['avg_sentence_length'],
            features['citation_density'],
            features['methodology_detail_ratio'],
            features['data_availability']
        ]
        
        return feature_values
    
    def _extract_key_claims(self, text):
        """Extract key claims from text using field-specific terms"""
        claim_sentences = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences: # Iterate through sentences
            if any(term in sentence.lower() for term in self.claim_terms):
                claim_sentences.append(sentence)
        
        return claim_sentences
    
    def _compare_claims(self, claims1, claims2):
        """Compare claims for consistency using shared terminology"""
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
        
        # Use batch_first=True for easier tensor manipulation
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(fusion_dim * 2) # Adjusted for concatenation
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, bert_embedding, handcrafted_features):
        # Project both embeddings to the fusion dimension
        bert_proj = self.bert_projection(bert_embedding)
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)
        
        # Apply batch norm if batch size is greater than 1
        if bert_proj.shape[0] > 1:
            bert_proj = self.bert_bn(bert_proj)
            handcrafted_proj = self.handcrafted_bn(handcrafted_proj)

        # Add a sequence dimension for attention
        bert_proj_seq = bert_proj.unsqueeze(1)
        handcrafted_proj_seq = handcrafted_proj.unsqueeze(1)

        # Cross-attention: text queries features, and features query text
        # This allows for a richer interaction between the two modalities
        attn_output_1, _ = self.attention(query=bert_proj_seq, key=handcrafted_proj_seq, value=handcrafted_proj_seq)
        attn_output_2, _ = self.attention(query=handcrafted_proj_seq, key=bert_proj_seq, value=bert_proj_seq)

        # Concatenate the original projected features with their attended counterparts
        fused_1 = torch.cat([bert_proj, attn_output_1.squeeze(1)], dim=-1)
        fused_2 = torch.cat([handcrafted_proj, attn_output_2.squeeze(1)], dim=-1)

        # Combine the two fusion paths
        output = self.layer_norm(fused_1 + fused_2)
        output = self.dropout(output)
        
        return output

# Modify the HierarchicalBiasPredictionModel class to use memory-optimized forward method
class HierarchicalBiasPredictionModel(nn.Module):
    def __init__(self, bert_model_name, num_classes=3, dropout_rate=0.2, feature_dim=29):
        super(HierarchicalBiasPredictionModel, self).__init__()

        print("Loading BERT model...")
        self.bert = AutoModel.from_pretrained(bert_model_name, add_pooling_layer=False, use_safetensors=True)
        self.bert.gradient_checkpointing_enable()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.bert_dim = self.bert.config.hidden_size  # 768 for standard BERT
        self.handcrafted_dim = feature_dim
        self.fusion_dim = 512  # Increased fusion dimension for more capacity
        
        # Attention layers for sentence and section aggregation
        self.sent_attention_projection = nn.Linear(self.bert_dim, self.bert_dim)
        self.sent_attention_vector = nn.Linear(self.bert_dim, 1, bias=False)
        
        self.sect_attention_projection = nn.Linear(self.bert_dim, self.bert_dim)
        self.sect_attention_vector = nn.Linear(self.bert_dim, 1, bias=False)
        
        # Feature fusion layer
        self.feature_fusion = FeatureFusionLayer(
            bert_dim=self.bert_dim, 
            handcrafted_dim=self.handcrafted_dim,
            fusion_dim=self.fusion_dim
        )
        
        # Classification layers - improved with batch normalization and residual connections
        self.fc1 = nn.Linear(self.fusion_dim * 2, 256) # Input is doubled from new fusion layer
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Residual connection for fusion_dim -> 128 shortcut
        self.shortcut = nn.Linear(self.fusion_dim * 2, 128)
        
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask, handcrafted_features):
        # input_ids: [batch, sections, sents, seq_len]
        batch_size, n_sections, n_sents, seq_len = input_ids.shape
        
        # 1. Reshape for batched BERT processing
        # Combine batch, sections, and sents dimensions
        input_ids_flat = input_ids.view(batch_size * n_sections * n_sents, seq_len)
        attention_mask_flat = attention_mask.view(batch_size * n_sections * n_sents, seq_len)
        
        # 2. Get BERT embeddings for all sentences in one pass
        # This is massively more efficient than looping
        bert_output = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        
        # Use the [CLS] token embedding for each sentence
        sent_embeddings = bert_output.last_hidden_state[:, 0, :] # [batch*sections*sents, hidden_dim]
        
        # 3. Reshape back to hierarchical structure
        sent_embeddings = sent_embeddings.view(batch_size, n_sections, n_sents, self.bert_dim)
        
        # 4. Hierarchical Attention
        # Create masks to ignore padded sentences/sections
        # A sentence is valid if its attention mask has any 1s
        sent_mask = (attention_mask.sum(dim=3) > 0).float() # [batch, sections, sents]
        
        # Sentence-level attention to get section vectors
        sent_proj = torch.tanh(self.sent_attention_projection(sent_embeddings))
        # Attention weights: [batch, sections, sents, 1]
        sent_att_weights = self.sent_attention_vector(sent_proj)

        sent_att_weights[sent_mask.unsqueeze(-1) == 0] = torch.finfo(sent_att_weights.dtype).min # Mask out padded sentences
        sent_att_weights = F.softmax(sent_att_weights, dim=2)
        
        # Weighted sum of sentence embeddings to get section embeddings
        # [batch, sections, hidden_dim]
        sect_embeddings = torch.einsum("bsnh,bsn->bsh", sent_embeddings, sent_att_weights.squeeze(-1))
        
        # Section-level attention to get document vector
        # A section is valid if any of its sentences are valid
        sect_mask = (sent_mask.sum(dim=2) > 0).float() # [batch, sections]
        
        sect_proj = torch.tanh(self.sect_attention_projection(sect_embeddings))
        # Attention weights: [batch, sections, 1]
        sect_att_weights = self.sect_attention_vector(sect_proj)

        sect_att_weights[sect_mask.unsqueeze(-1) == 0] = torch.finfo(sect_att_weights.dtype).min # Mask out padded sections
        sect_att_weights = F.softmax(sect_att_weights, dim=1)
        
        # Weighted sum of section embeddings to get document embedding
        # [batch, hidden_dim]
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
        x2 = self.relu(x2 + shortcut) # Apply activation after residual connection
        x2 = self.dropout(x2)
        output = self.fc3(x2)
        
        return output

# Enhanced model training function with advanced scheduling
def train_hierarchical_model(train_dataloader, val_dataloader, model, device, 
                           epochs=40, lr=3e-5, accumulation_steps=16):
    """
    Train the hierarchical model with gradual unfreezing and discriminative learning rates.
    
    Stage 1: Freeze BERT and train only the classifier head.
    Stage 2: Unfreeze all layers and fine-tune the entire model with different learning rates.
    """
    
    # --- Stage 1: Train Classifier Head ---
    print("\n--- Training Stage 1: Fine-tuning the classifier head ---")
    # Freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    # Create an optimizer for only the unfrozen classifier layers
    classifier_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stage1 = torch.optim.AdamW(classifier_params, lr=lr*2, weight_decay=0.02) # Higher LR for head
    
    # Train for a few epochs
    stage1_epochs = 3
    for epoch in range(stage1_epochs):
        _train_one_epoch(epoch, stage1_epochs, model, train_dataloader, optimizer_stage1, accumulation_steps, device, is_classifier_only=True)
        # A quick validation check after each head-tuning epoch
        _validate_one_epoch(model, val_dataloader, device)

    # --- Set up for Stage 2 ---
    # We will unfreeze layers and set up the optimizer inside the loop to avoid multiprocessing print issues
    optimizer_stage2 = None
    scheduler = None
    main_epochs = epochs - stage1_epochs
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_accuracy = 0
    best_model = model.state_dict().copy() # Initialize with the starting model state
    patience_counter = 0
    
    for epoch in range(main_epochs):
        # --- Set up Stage 2 on the first epoch ---
        if epoch == 0:
            print("\n--- Training Stage 2: Unfreezing and fine-tuning the entire model ---")
            # Unfreeze all layers
            for param in model.bert.parameters():
                param.requires_grad = True

            # --- Set up Layer-wise Learning Rate Decay (LLRD) for BERT ---
            print("Setting up Layer-wise Learning Rate Decay (LLRD)...")
            optimizer_parameters = []
            num_bert_layers = model.bert.config.num_hidden_layers
            lr_decay_rate = 0.95

            head_lr = lr * 2
            classifier_params = [
                model.feature_fusion, model.fc1, model.bn1, model.fc2, model.bn2,
                model.fc3, model.shortcut, model.sent_attention_projection,
                model.sent_attention_vector, model.sect_attention_projection,
                model.sect_attention_vector
            ]
            optimizer_parameters.append({'params': [p for model_part in classifier_params for p in model_part.parameters()], 'lr': head_lr})

            for i in range(num_bert_layers):
                layer_lr = lr * (lr_decay_rate ** (num_bert_layers - i))
                optimizer_parameters.append({'params': model.bert.encoder.layer[i].parameters(), 'lr': layer_lr})

            optimizer_parameters.append({'params': model.bert.embeddings.parameters(), 'lr': lr * (lr_decay_rate ** (num_bert_layers + 1))})

            optimizer_stage2 = torch.optim.AdamW(optimizer_parameters, weight_decay=0.01, eps=1e-6)

            num_training_steps = len(train_dataloader) * main_epochs // accumulation_steps
            num_warmup_steps = num_training_steps // 10
            scheduler = get_linear_schedule_with_warmup(
                optimizer_stage2,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

        # Use the helper function for training one epoch
        train_loss = _train_one_epoch(
            epoch, main_epochs, model, train_dataloader, optimizer_stage2, 
            accumulation_steps, device, is_classifier_only=False, 
            scheduler=scheduler, loss_fn=loss_fn
        )
        
        # Use the helper function for validation
        val_accuracy, val_loss = _validate_one_epoch(model, val_dataloader, device, loss_fn=loss_fn)
        
        current_lr = optimizer_stage2.param_groups[0]['lr'] # Get LR of the classifier head
        
        print(f'Epoch {epoch+1+stage1_epochs}/{epochs}: '
              f'Train Loss = {train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}, '
              f'Val Accuracy = {val_accuracy:.4f}, '
              f'LR = {current_lr:.6f}')

        patience = 12 # Increased patience for full fine-tuning
        # Early stopping and best model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict().copy()
            # Save the best model state
            save_path = 'best_hierarchical_physical_science_model.pt'
            torch.save(best_model, save_path)
            patience_counter = 0
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1+stage1_epochs} epochs")
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

def _train_one_epoch(epoch, total_epochs, model, dataloader, optimizer, 
                   accumulation_steps, device, is_classifier_only=False, 
                   scheduler=None, loss_fn=None):
    """Helper function to run a single training epoch."""
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    max_grad_norm = 1.0  # Gradient clipping

    # Use a scaler for mixed-precision training to prevent underflow
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Use provided loss function or default
    if loss_fn is None:
        # Default to CrossEntropy if no loss function is provided for some reason
        loss_fn = nn.CrossEntropyLoss()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        handcrafted_features = batch['handcrafted_features'].to(device)
        labels = batch['label'].to(device)
        
        if device.type == 'cuda':
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids, attention_mask, handcrafted_features)
                loss = loss_fn(outputs, labels)
        else:
            outputs = model(input_ids, attention_mask, handcrafted_features)
            loss = loss_fn(outputs, labels)
        
        # Scale loss for gradient accumulation
        scaled_loss = scaler.scale(loss / accumulation_steps)
        scaled_loss.backward()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer) # Update weights
            scaler.update()
            if scheduler:
                scheduler.step() # Update learning rate
            optimizer.zero_grad() # Reset gradients
        
        progress_bar.set_postfix(loss=loss.item())
            
    # Handle the final batch if it's not a multiple of accumulation_steps
    if (len(dataloader) % accumulation_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad() # Reset gradients

    train_loss /= len(dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    
    if is_classifier_only:
        print(f"Classifier Head Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
    
    return train_loss

def _validate_one_epoch(model, dataloader, device, loss_fn=None):
    """Helper function to run a single validation epoch."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # Use provided loss function or default
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
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
            
    val_loss /= len(dataloader)
    val_accuracy = correct / total
    
    return val_accuracy, val_loss







# Enhanced evaluation function with business-specific metrics
def evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names=None):
    """Evaluate model on test set with business-specific analysis"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating on Test Set"):
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
                               output_dict=True,
                               zero_division=0)
        
        print(classification_report(all_labels, all_preds, 
                               target_names=filtered_label_names,
                               labels=present_labels,
                               zero_division=0))
        
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
            cmap='YlGnBu',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=filtered_label_names
        )
        plt.title('Classification Performance by Physical Science Bias Type')
        plt.tight_layout()
        plt.savefig('physical_science_classification_performance.png')
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Physical Science Bias Detection - Confusion Matrix')
        plt.savefig('physical_science_confusion_matrix.png')
        plt.close()
        
        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu', 
                    xticklabels=[label_names[i] for i in present_labels],
                    yticklabels=[label_names[i] for i in present_labels])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Physical Science Bias Detection - Normalized Confusion Matrix')
        plt.savefig('physical_science_normalized_confusion_matrix.png')
        plt.close()
    else:
        print("Not enough unique classes in test set to generate a meaningful confusion matrix")
    
    return all_preds, all_labels

def load_papers_from_json(json_file_path):
    """
    Load physical science research papers from a JSON file.
    
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
    
    def _get_label(paper_dict):
        """Helper to extract and normalize bias labels from a paper dictionary."""
        # Case-insensitive search for 'overallbias'
        for k, v in paper_dict.items():
            if k.replace('_', '').replace(' ', '').lower() == 'overallbias':
                return v
        
        # Check common explicit keys
        for key in ['OverallBias', 'Overall Bias', 'Overall_Bias', 'bias_label', 'bias_type', 
                    'CognitiveBias', 'PublicationBias', 'Selection Bias', 'No Bias']:
            if key in paper_dict:
                return paper_dict[key]
        
        return 0 # Default to 'No Bias'

    def _normalize_label(label):
        """Helper to convert various label formats to integers."""
        if isinstance(label, (int, float)):
            return int(label)
        if isinstance(label, str):
            label_normal = label.strip().lower()
            label_map = {
                'no bias': 0, 'nobias': 0,
                'cognitive bias': 1, 'cognitivebias': 1, 'confirmation bias': 1,
                'publication bias': 2, 'publicationbias': 2,
                'selection bias': 2, 'selectionbias': 2, 'survivorship bias': 2
            }
            if label_normal in label_map:
                return label_map[label_normal]
            # Try to parse integers from strings like 'Overall Bias: 1'
            m = re.search(r"(\d+)", label_normal)
            return int(m.group(1)) if m else 0
        return 0

    try:
        with open(json_file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        
        papers = []
        papers_data = []

        # Handle common JSON structures: list of papers or dict of papers
        if isinstance(data, list):
            papers_data = data
        elif isinstance(data, dict):
            # Can be a dict with a 'papers' key, or a dict where values are papers
            if 'papers' in data and isinstance(data.get('papers'), list):
                papers_data = data['papers']
            else:
                papers_data = list(data.values())
        
        for paper_dict in papers_data:
            if not isinstance(paper_dict, dict):
                continue # Skip any non-dictionary items in the list
            
            text = paper_dict.get('Body', paper_dict.get('text', ''))
            raw_label = _get_label(paper_dict)
            normalized_label = _normalize_label(raw_label)
            papers.append({'text': text, 'label': normalized_label})

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
        
        print(f"Successfully loaded {len(df)} physical science papers from {json_file_path}")
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: '{json_file_path}' is not a valid JSON file: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"Error loading physical science papers from {json_file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])

# Function to analyze feature importance
def analyze_feature_importance(model, feature_names):
    """
    Analyze which handcrafted features are most important for bias detection
    by inspecting the weights of the feature fusion layer.
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
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='plasma', legend=False)
    plt.title('Physical Science Feature Importance for Bias Detection')
    plt.tight_layout()
    plt.savefig('physical_science_feature_importance.png')
    plt.close()
    
    # Plot feature importance by category
    # Group features by category for physical science
    feature_categories = {
        'Basic Text': ['length', 'avg_word_length'],
        'Statistical': ['p_value_count', 'signif_stars_count', 'correlation_count', 
                       'percentage_count', 'results_stat_density'],
        'Linguistic': ['hedge_ratio', 'certainty_ratio'],
        'Field-Specific': ['theory_term_ratio', 'promo_term_ratio', 'efficiency_claim_ratio'],
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
    sns.barplot(x='Average Importance', y='Category', data=category_df, hue='Category', palette='plasma', legend=False)
    plt.title('Feature Category Importance for Physical Science Bias Detection')
    plt.tight_layout()
    plt.savefig('physical_science_feature_category_importance.png')
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
    # --- PyTorch/CUDA Diagnostics ---
    print("--- PyTorch/CUDA Diagnostics ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Active GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available to PyTorch. The script will run on the CPU.")
    print("------------------------------------")

    # Prefer GPU if available, otherwise CPU
    # To force CPU, comment the line below and uncomment the next one.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Download NLTK data if needed
    download_nltk_data()
    
    # Feature names for the physical science-specific extractor
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'correlation_count', 'percentage_count', 'hedge_ratio', 'certainty_ratio', 
        'theory_term_ratio', 'promo_term_ratio', 'method_term_count', 'validation_pattern_count',
        'abstract_claim_ratio', 'results_stat_density', 'limitations_mentioned',
        'efficiency_claim_ratio', 'claim_consistency', 'figure_mentions', 'table_mentions',
        'chart_mentions', 'citation_but_count', 'self_reference_count',
        'flesch_reading_ease', 'flesch_kincaid_grade', 'sentence_length_variance',
        'avg_sentence_length', 'citation_density', 'methodology_detail_ratio', 'data_availability'
    ]
    
    # Load and process paper data from JSON
    try:
        # Specify the path to your physical science papers JSON file
        papers_df = load_papers_from_json('physical_sciences_papers.json')
        print(f"Loaded {len(papers_df)} physical science papers")
        
        # Check class distribution
        print("Class distribution:")
        class_counts = papers_df['label'].value_counts()
        print(class_counts)

        # If there's insufficient data, create some dummy data for testing
        if len(papers_df) <= 3:  # Need at least 3 for train/val/test
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! WARNING: FAILED TO LOAD REAL DATA.                        !!!")
            print("!!! USING A SMALL, REPETITIVE DUMMY DATASET FOR TESTING.      !!!")
            print("!!! RESULTS WILL NOT BE MEANINGFUL.                           !!!")
            print("!!! PLEASE FIX 'physical_sciences_papers.json' TO GET ACCURACY.  !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            # Create more diverse dummy physical science data
            dummy_texts = [
                "This paper presents a measurement of the gravitational constant. We found no significant bias.", # No Bias
                "Our revolutionary experiment on cold fusion clearly proves our initial hypothesis.", # Cognitive Bias
                "Analysis of astronomical surveys shows a strong selection bias towards detecting bright galaxies.", # Selection/Pub Bias
                "A simulation was conducted to model particle interactions. The results were neutral.", # No Bias
                "We argue that the interpretation of this quantum state is flawed and demonstrates confirmation bias.", # Cognitive Bias
                "The exclusion of noisy data from the analysis introduces a selection bias.", # Selection/Pub Bias
            ] * 2 # Repeat to get 12 samples
            papers_df = pd.DataFrame({
                'text': dummy_texts,
                'label': [0, 1, 2, 0, 1, 2] * 2
            })
    except Exception as e:
        print(f"Error loading papers: {e}")
        # Generate dummy data for testing
        # This block is a fallback, but the primary dummy data generation is above.
        # For consistency, we can reuse the logic, but it's unlikely to be hit.
        raise e # Re-raise the exception to make it clear something went wrong.
    
    # Extract handcrafted features with physical science-specific extractor
    print("Extracting physical science-specific handcrafted features...")
    
    # --- Parallelize Feature Extraction ---
    # This is a major bottleneck. We use multiprocessing to speed it up.
    from functools import partial
    from multiprocessing import Pool

    extractor = PhysicalScienceFeatureExtractor()

    # Use a Pool of workers to process texts in parallel
    # Use os.cpu_count() - 1 to leave one core for other tasks, or 1 if only one core is available.
    with Pool(processes=os.cpu_count() - 1 or 1) as pool:
        # Use partial to pass the extractor instance to the worker function
        worker_func = partial(_extract_features_worker, extractor_instance=extractor)
        handcrafted_features = list(tqdm(pool.imap(worker_func, papers_df['text']), total=len(papers_df), desc="Extracting Features"))

    # Convert to numpy arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Normalize features using StandardScaler for better training stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    print("Features normalized using StandardScaler")
    
    # Stratified split to ensure all sets have representation of each class
    # Using StratifiedShuffleSplit for robustness with small classes.
    from sklearn.model_selection import StratifiedShuffleSplit
    
    all_indices = np.arange(len(labels_array))
    
    # Split into train+validation (80%) and test (20%)
    split_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_val_idx_gen, test_idx_gen = next(split_test.split(all_indices, labels_array))
        train_val_indices = train_val_idx_gen
        test_indices = test_idx_gen
    except ValueError: # Fallback for extremely small datasets where stratification fails
        print("Warning: Stratified split failed. Falling back to regular train_test_split.")
        train_val_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    # Split train+validation into train (60% of total) and validation (20% of total)
    # This means the validation set is 25% of the train_val set (0.2 / 0.8 = 0.25)
    split_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    try:
        train_idx_gen, val_idx_gen = next(split_val.split(train_val_indices, labels_array[train_val_indices]))
        train_indices = train_val_indices[train_idx_gen]
        val_indices = train_val_indices[val_idx_gen]
    except ValueError:
        print("Warning: Stratified split for validation failed. Falling back to regular split.")
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, random_state=42)

    
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
    original_train_features = train_features.copy() # Keep a copy for nearest neighbor search
    train_labels = np.array(y_train)
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Apply SMOTE to the training data
    print("Applying SMOTE to balance classes...")
    try:
        # Get original class distribution
        original_counts = Counter(train_labels)
        print("Original class distribution:", original_counts)

        # --- Simplified and Robust Oversampling Logic ---
        # SMOTE requires k_neighbors < number of samples in the smallest class to be oversampled.
        # We dynamically adjust k_neighbors to be safe.
        minority_counts = [c for c in original_counts.values() if c < max(original_counts.values())]
        
        if not minority_counts:
            print("Classes are already balanced or only one class is present. Skipping SMOTE.")
            resampled_features, resampled_labels = train_features, train_labels
        else:
            # Set k_neighbors to be one less than the smallest minority class count, but at least 1.
            min_minority_size = min(minority_counts)
            k_neighbors = max(1, min_minority_size - 1)
            
            print(f"Smallest minority class has {min_minority_size} samples. Setting k_neighbors for SMOTE to {k_neighbors}.")
            
            # We can use SMOTE for all minority classes with this adjusted k_neighbors.
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
            resampled_features, resampled_labels = smote.fit_resample(train_features, train_labels)
        
        # Generate synthetic texts for new samples
        new_counts = Counter(resampled_labels)
        
        num_original_samples = len(train_texts)
        
        if len(resampled_features) > num_original_samples:
            from sklearn.neighbors import NearestNeighbors
            print("Generating text for synthetic samples by finding nearest neighbors...")
            nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(original_train_features)
            # Find nearest original text for all synthetic feature vectors at once
            synthetic_features = resampled_features[num_original_samples:]
            _, indices = nn.kneighbors(synthetic_features)
            synthetic_texts_to_add = [train_texts[i[0]] for i in indices]
            synthetic_texts = list(train_texts) + synthetic_texts_to_add
        else: # No new samples were added
            synthetic_texts = list(train_texts)
        
        # Update train_texts and train_features with resampled data
        train_texts = synthetic_texts
        train_features = resampled_features
        train_labels = resampled_labels
        
        print("After balancing class distribution:", new_counts)
        
    except Exception as e:
        print(f"Error in data balancing process: {e}")
        print("Proceeding with original imbalanced data")
    
    # Initialize hierarchical model - force CPU execution
    print("Initializing hierarchical model...")
    # For faster CPU training, use a distilled model. For best accuracy (on GPU), use the scientific model.
    model_name = 'allenai/scibert_scivocab_uncased'
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_safetensors=True)
    model = HierarchicalBiasPredictionModel(
        model_name, 
        num_classes=3,
        feature_dim=len(feature_names),
        dropout_rate=0.2 # Reduced dropout for better learning
    )
    
    # Memory optimization - clear unnecessary variables
    del papers_df, handcrafted_features, features_array, labels_array
    gc.collect()
    
    # Use increased context size
    print(f"Using increased context size for model")
    max_seq_length = 128  # Reduced to save memory
    max_sections = 8      # Increased to analyze more paper sections
    max_sents = 8         # Reduced to save memory
    
    # Create hierarchical datasets with increased context
    print("Creating hierarchical datasets...")
    train_dataset = HierarchicalPaperDataset(
        list(train_texts), train_labels, tokenizer, train_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    val_dataset = HierarchicalPaperDataset(
        val_texts, y_val, tokenizer, val_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    test_dataset = HierarchicalPaperDataset(
        test_texts, y_test, tokenizer, test_features,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Use a batch size appropriate for the dataset and hardware
    batch_size = 2  # Reduced to fit into GPU memory
    print(f"Using batch size: {batch_size} for large balanced dataset ({len(list(train_texts))} samples)")
    
    # Use multiple workers for data loading to speed up training.
    # `pin_memory=True` works with `num_workers > 0` to speed up CPU-to-GPU data transfer.
    num_workers = 2 if os.name == 'nt' else 4 # A good starting point, can be tuned
    print(f"Using {num_workers} workers for data loading.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                                num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False,
                                 num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    
    # Ensure model is on the correct device
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # JIT compile the model for a significant speedup (PyTorch 2.0+).
    # 'max-autotune' mode spends more time tuning to find the fastest kernels.
    # NOTE: Triton, the backend for inductor on GPU, is not officially supported on Windows.
    # We will skip compilation on Windows when using CUDA to avoid errors.
    if device.type == 'cuda' and os.name == 'nt':
        print("Skipping torch.compile() on Windows with CUDA due to lack of official Triton support.")
    elif torch.__version__.startswith("2."):
        try:
            model = torch.compile(model, mode="max-autotune", dynamic=True)
            print("Model successfully compiled for optimized performance.")
        except Exception as e:
            print(f"Could not compile model: {e}. Proceeding without compilation.")
    else:
        print("PyTorch version is not 2.x. Skipping model compilation.")


    # Train hierarchical model optimized for large balanced dataset
    print("Training hierarchical model...")
    model = train_hierarchical_model(
        train_dataloader, val_dataloader, model, device, 
        epochs=40,  # Increased epochs for convergence
        accumulation_steps=32,  # Effective batch size = 2 * 32 = 64
        lr=2e-5  # A slightly smaller LR is often more stable with smaller batches
    )
    
    # Evaluate hierarchical model
    print("Evaluating hierarchical model...")
    label_names = ['No Bias', 'Cognitive Bias', 'Selection/Publication Bias']
    evaluate_hierarchical_model(test_dataloader, model, device, label_names, feature_names=feature_names)
    
    # Analyze physical science-specific feature importance
    print("Analyzing physical science-specific feature importance...")
    analyze_feature_importance(model, feature_names)
    
    print("Physical science bias detection analysis complete!")

# Define the worker function at the top level of the module for multiprocessing
def _extract_features_worker(text, extractor_instance):
    """Helper function for parallel feature extraction."""
    return extractor_instance.extract_features(text)

if __name__ == "__main__":
    # Set the Python multiprocessing method to avoid issues on Windows
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Run the main function
    main()