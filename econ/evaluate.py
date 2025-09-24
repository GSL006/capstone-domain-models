from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import random
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
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

# Import all the necessary classes from the original training script
# (Copy the same classes to ensure exact preprocessing)

class EconomicsBiasFeatureExtractor:
    """Extract features that might indicate bias in economics research papers"""
    
    def __init__(self):
        # Basic patterns
        self.p_value_pattern = r'p\s*[<>=]\s*0\.0\d+'
        self.significance_stars = r'\*{1,3}\s*'
        self.coefficient_pattern = r'(?:coefficient|coef|β|beta)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+'
        
        # Economics-specific hedge words
        self.hedge_words = [
            'may', 'might', 'could', 'possibly', 'potentially', 'suggests', 
            'appears', 'seems', 'likely', 'unlikely', 'perhaps', 'arguably',
            'tend to', 'tends to', 'tended to', 'indicate', 'indicates'
        ]
        
        # Economics-specific certainty words
        self.certainty_words = [
            'clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
            'conclusively', 'absolutely', 'always', 'never', 'established',
            'proves', 'demonstrates', 'robust', 'significant', 'substantial',
            'strong evidence', 'strongly supports', 'decisive'
        ]
        
        # Economic theory references
        self.theory_terms = [
            'neoclassical', 'keynesian', 'austrian', 'marxist', 'monetarist',
            'theory predicts', 'consistent with theory', 'theoretical predictions',
            'standard model', 'economic theory'
        ]
        
        # Economics-specific claim words
        self.claim_terms = [
            'impact', 'effect', 'affect', 'influence', 'determine', 'cause',
            'increase', 'decrease', 'improve', 'worsen', 'optimal', 'efficiency',
            'welfare', 'growth', 'productivity', 'policy implications'
        ]
        
        # Economics jargon that might signal overconfidence
        self.econ_jargon = [
            'ceteris paribus', 'exogenous', 'endogenous', 'instrumental variable',
            'causal effect', 'causal identification', 'identification strategy',
            'natural experiment', 'quasi-experimental', 'control for'
        ]
        
        # Robustness check words
        self.robustness_terms = [
            'robust', 'robustness', 'sensitivity', 'alternative specification',
            'subsample', 'validation', 'check', 'falsification test'
        ]
        
        # Economics-specific variable patterns
        self.econometric_patterns = [
            r'OLS', r'2SLS', r'IV', r'GMM', r'fixed[ -]effects', r'random[ -]effects',
            r'diff[- ]in[- ]diff', r'regression discontinuity', r'matching',
            r'instrumental variable', r'propensity score'
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
        features['coefficient_count'] = len(re.findall(self.coefficient_pattern, text))
        
        # 3. Linguistic features
        # Hedging and certainty
        hedge_count = sum(text.lower().count(word) for word in self.hedge_words)
        certainty_count = sum(text.lower().count(word) for word in self.certainty_words)
        features['hedge_ratio'] = hedge_count / word_count * 1000  # per 1000 words
        features['certainty_ratio'] = certainty_count / word_count * 1000  # per 1000 words
        
        # 4. Economic-specific patterns
        theory_count = sum(text.lower().count(term) for term in self.theory_terms)
        jargon_count = sum(text.lower().count(term) for term in self.econ_jargon)
        features['theory_term_ratio'] = theory_count / word_count * 1000
        features['jargon_term_ratio'] = jargon_count / word_count * 1000
        
        # 5. Economics method patterns
        econometric_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in self.econometric_patterns)
        features['econometric_method_count'] = econometric_count
        
        # 6. Section-specific features
        # Abstract features (claims without evidence)
        if 'abstract' in sections:
            abstract = sections['abstract']
            abstract_words = len(abstract.split()) + 1
            abstract_claim_count = sum(abstract.lower().count(term) for term in self.claim_terms)
            features['abstract_claim_ratio'] = abstract_claim_count / abstract_words * 1000
        else:
            features['abstract_claim_ratio'] = 0
        
        # Results section features (p-hacking indicators)
        if 'results' in sections:
            results = sections['results']
            results_p_values = len(re.findall(self.p_value_pattern, results))
            results_words = len(results.split()) + 1
            features['results_p_value_density'] = results_p_values / results_words * 1000
        else:
            features['results_p_value_density'] = 0
        
        # Limitations acknowledgment
        features['limitations_mentioned'] = 1 if "limitation" in text.lower() or "limitations" in text.lower() else 0
        
        # 7. Robustness checks
        robustness_count = sum(text.lower().count(term) for term in self.robustness_terms)
        features['robustness_check_ratio'] = robustness_count / word_count * 1000
        
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
        
        # 10. One-sided citation patterns
        features['citation_but_count'] = (text.count("et al., but") + 
                                       text.count("et al. but") + 
                                       text.count("] but") +
                                       text.count(") but"))
        
        # 11. Self-citation patterns
        features['self_reference_count'] = (text.lower().count("our research") + 
                                        text.lower().count("our study") + 
                                        text.lower().count("our analysis") +
                                        text.lower().count("our results") +
                                        text.lower().count("our findings"))
        
        # Return feature values as a list in a consistent order (NO method_limitation_ratio calculation - use 0 like training)
        feature_values = [
            features['length'], 
            features['avg_word_length'],
            features['p_value_count'], 
            features['signif_stars_count'],
            features['coefficient_count'],
            features['hedge_ratio'], 
            features['certainty_ratio'],
            features['theory_term_ratio'],
            features['jargon_term_ratio'],
            features['econometric_method_count'],
            features['abstract_claim_ratio'],
            features['results_p_value_density'],
            features['limitations_mentioned'],
            features['robustness_check_ratio'],
            features['claim_consistency'],
            features['figure_mentions'],
            features['table_mentions'],
            features['citation_but_count'],
            features['self_reference_count'],
            features.get('method_limitation_ratio', 0)  # Include with default if not set (SAME AS TRAINING)
        ]
        
        return feature_values
    
    def _extract_sections(self, text):
        """Extract sections from the economics paper text"""
        section_dict = {}
        section_markers = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', 'i. introduction'],
            'literature': ['literature', 'literature review', 'related work', 'related literature'],
            'methods': ['method', 'methods', 'methodology', 'model', 'data and methods'],
            'data': ['data', 'data description', 'empirical data'],
            'results': ['results', 'findings', 'empirical results', 'analysis'],
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
        """Extract key claims from text using economics-specific terms"""
        claim_sentences = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in self.claim_terms):
                claim_sentences.append(sentence)
        
        return claim_sentences
    
    def _compare_claims(self, claims1, claims2):
        """Compare claims for consistency using economics-specific terminology"""
        if not claims1 or not claims2:
            return 0
        
        # Tokenize and remove stopwords
        words1 = []
        for claim in claims1:
            words = [w.lower() for w in word_tokenize(claim) if w.lower() not in self.stopwords]
            words1.extend(words)
        
        words2 = []
        for claim in claims2:
            words = [w.lower() for w in word_tokenize(claim) if w.lower() not in self.stopwords]
            words2.extend(words)
        
        # Calculate overlap
        common_words = set(words1).intersection(set(words2))
        all_words = set(words1).union(set(words2))
        
        if not all_words:
            return 0
            
        return len(common_words) / len(all_words)


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
            sentences = sent_tokenize(section)
            section_token_ids = []
            section_attention_masks = []
            
            for sentence in sentences[:self.max_sents]:
                encoding = self.tokenizer(
                    sentence,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                section_token_ids.append(encoding['input_ids'].squeeze(0))
                section_attention_masks.append(encoding['attention_mask'].squeeze(0))
            
            # Pad sentences to max_sents
            while len(section_token_ids) < self.max_sents:
                section_token_ids.append(torch.zeros(self.max_seq_length, dtype=torch.long))
                section_attention_masks.append(torch.zeros(self.max_seq_length, dtype=torch.long))
            
            token_ids.append(torch.stack(section_token_ids))
            attention_masks.append(torch.stack(section_attention_masks))
        
        # Pad sections to max_sections
        while len(token_ids) < self.max_sections:
            token_ids.append(torch.zeros(self.max_sents, self.max_seq_length, dtype=torch.long))
            attention_masks.append(torch.zeros(self.max_sents, self.max_seq_length, dtype=torch.long))
        
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


def load_papers_from_json(json_file_path):
    """
    Load economics research papers from a JSON file.
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing paper data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing paper texts and bias labels
    """
    print(f"Loading papers from {json_file_path}...")
    
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
            for item in data:
                if isinstance(item, dict):
                    # Extract text content
                    text_content = ""
                    
                    # Try different field names for text content
                    text_fields = ['Body', 'text', 'content', 'abstract', 'full_text', 'paper_text']
                    for field in text_fields:
                        if field in item and item[field]:
                            text_content = str(item[field])
                            break
                    
                    if not text_content:
                        continue
                    
                    # Extract bias label
                    label = None
                    
                    # Try different field names for labels
                    if 'OverallBias' in item:
                        bias_str = str(item['OverallBias']).strip()
                        if bias_str == "No Bias":
                            label = 0
                        elif bias_str in ["Cognitive Bias", "CognitiveBias"]:
                            label = 1
                        elif bias_str in ["Publication Bias", "PublicationBias", "Selection Bias", "SelectionBias"]:
                            label = 2
                    elif 'Overall Bias' in item:
                        bias_str = str(item['Overall Bias']).strip()
                        if bias_str == "No Bias":
                            label = 0
                        elif bias_str in ["Cognitive Bias", "CognitiveBias"]:
                            label = 1
                        elif bias_str in ["Publication Bias", "PublicationBias", "Selection Bias", "SelectionBias"]:
                            label = 2
                    elif 'label' in item:
                        try:
                            label = int(item['label'])
                            if label not in [0, 1, 2]:
                                continue
                        except (ValueError, TypeError):
                            continue
                    elif 'bias_type' in item:
                        bias_str = str(item['bias_type']).strip().lower()
                        if bias_str in ["no bias", "none", "0"]:
                            label = 0
                        elif bias_str in ["cognitive bias", "cognitive", "1"]:
                            label = 1
                        elif bias_str in ["publication bias", "selection bias", "publication", "selection", "2"]:
                            label = 2
                    # Use numeric bias fields if available
                    elif 'NoBias' in item and 'CognitiveBias' in item and 'PublicationBias' in item:
                        try:
                            no_bias = float(item['NoBias'])
                            cognitive_bias = float(item['CognitiveBias'])
                            publication_bias = float(item['PublicationBias'])
                            
                            # Find the maximum value
                            max_val = max(no_bias, cognitive_bias, publication_bias)
                            if max_val == no_bias:
                                label = 0
                            elif max_val == cognitive_bias:
                                label = 1
                            else:
                                label = 2
                        except (ValueError, TypeError):
                            continue
                    
                    if label is not None:
                        papers.append({
                            'text': text_content,
                            'label': label
                        })
        
        # Case 2: Dictionary with papers as values
        elif isinstance(data, dict):
            # Check if it's a dictionary of papers
            for key, item in data.items():
                if isinstance(item, dict):
                    # Extract text content
                    text_content = ""
                    
                    # Try different field names for text content
                    text_fields = ['Body', 'text', 'content', 'abstract', 'full_text', 'paper_text']
                    for field in text_fields:
                        if field in item and item[field]:
                            text_content = str(item[field])
                            break
                    
                    if not text_content:
                        continue
                    
                    # Extract bias label
                    label = None
                    
                    # Try different field names for labels
                    if 'OverallBias' in item:
                        bias_str = str(item['OverallBias']).strip()
                        if bias_str == "No Bias":
                            label = 0
                        elif bias_str in ["Cognitive Bias", "CognitiveBias"]:
                            label = 1
                        elif bias_str in ["Publication Bias", "PublicationBias", "Selection Bias", "SelectionBias"]:
                            label = 2
                    elif 'Overall Bias' in item:
                        bias_str = str(item['Overall Bias']).strip()
                        if bias_str == "No Bias":
                            label = 0
                        elif bias_str in ["Cognitive Bias", "CognitiveBias"]:
                            label = 1
                        elif bias_str in ["Publication Bias", "PublicationBias", "Selection Bias", "SelectionBias"]:
                            label = 2
                    elif 'label' in item:
                        try:
                            label = int(item['label'])
                            if label not in [0, 1, 2]:
                                continue
                        except (ValueError, TypeError):
                            continue
                    elif 'bias_type' in item:
                        bias_str = str(item['bias_type']).strip().lower()
                        if bias_str in ["no bias", "none", "0"]:
                            label = 0
                        elif bias_str in ["cognitive bias", "cognitive", "1"]:
                            label = 1
                        elif bias_str in ["publication bias", "selection bias", "publication", "selection", "2"]:
                            label = 2
                    # Use numeric bias fields if available
                    elif 'NoBias' in item and 'CognitiveBias' in item and 'PublicationBias' in item:
                        try:
                            no_bias = float(item['NoBias'])
                            cognitive_bias = float(item['CognitiveBias'])
                            publication_bias = float(item['PublicationBias'])
                            
                            # Find the maximum value
                            max_val = max(no_bias, cognitive_bias, publication_bias)
                            if max_val == no_bias:
                                label = 0
                            elif max_val == cognitive_bias:
                                label = 1
                            else:
                                label = 2
                        except (ValueError, TypeError):
                            continue
                    
                    if label is not None:
                        papers.append({
                            'text': text_content,
                            'label': label
                        })
        
        else:
            print("Error: Unrecognized JSON format")
            return pd.DataFrame(columns=['text', 'label'])
        
        # Create DataFrame
        df = pd.DataFrame(papers)
        
        # Validate data
        if len(df) == 0:
            print("Warning: No valid papers found in the JSON file")
            return pd.DataFrame(columns=['text', 'label'])
            
        if 'text' not in df.columns:
            print("Error: No text column found after processing")
            return pd.DataFrame(columns=['text', 'label'])
        
        if 'label' not in df.columns:
            print("Error: No label column found after processing")
            return pd.DataFrame(columns=['text', 'label'])
        
        # Ensure text is string and label is integer
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        
        # Remove any rows with empty text
        original_count = len(df)
        df = df[df['text'].str.strip().str.len() > 0].reset_index(drop=True)
        if len(df) < original_count:
            print(f"Removed {original_count - len(df)} papers with empty text")
        
        # Validate label values
        if not all(df['label'].isin([0, 1, 2])):
            invalid_labels = df[~df['label'].isin([0, 1, 2])]['label'].unique()
            print(f"Warning: Found invalid labels: {invalid_labels}")
            df = df[df['label'].isin([0, 1, 2])].reset_index(drop=True)
        
        print(f"Successfully loaded {len(df)} economics papers from {json_file_path}")
        print("Class distribution:")
        label_names = ['No Bias', 'Cognitive Bias', 'Selection/Publication Bias']
        for label, count in df['label'].value_counts().sort_index().items():
            print(f"  {label_names[label]}: {count} papers")
        
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: {json_file_path} is not a valid JSON file: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"Error loading economics papers from {json_file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])


def sample_papers(papers_df, n_samples=1000, random_seed=42):
    """
    Sample n_samples papers from the dataset
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if len(papers_df) <= n_samples:
        print(f"Dataset has only {len(papers_df)} papers, using all of them")
        return papers_df.copy()
    
    # Sample papers while maintaining class distribution
    sampled_papers = papers_df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    
    print(f"Sampled {len(sampled_papers)} papers from {len(papers_df)} total papers")
    print("Class distribution in sample:")
    print(sampled_papers['label'].value_counts().sort_index())
    
    return sampled_papers


def preprocess_papers(papers_df, feature_names):
    """
    Apply the same preprocessing steps as in the training script
    """
    print("Extracting economics-specific handcrafted features...")
    extractor = EconomicsBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    # Convert to numpy arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # NOTE: The training script doesn't use StandardScaler, so we shouldn't either
    # Using the raw features as they were used during training
    print("Using raw features (same as training - no normalization applied)")
    
    return papers_df['text'].tolist(), features_array, labels_array


def evaluate_model(model_path, papers_df, feature_names, device='cpu'):
    """
    Evaluate the trained model on the given papers
    """
    # Define label names
    label_names = ['No Bias', 'Cognitive Bias', 'Selection/Publication Bias']
    
    # Preprocess the papers
    texts, features_array, labels_array = preprocess_papers(papers_df, feature_names)
    
    # Setup tokenizer (same as training)
    print("Setting up tokenizer...")
    try:
        model_name = 'allenai/scibert_scivocab_uncased'
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Same model configuration as training
    max_seq_length = 64   # Same as training
    max_sections = 2      # Same as training
    max_sents = 4         # Same as training
    
    # Create dataset
    print("Creating evaluation dataset...")
    eval_dataset = HierarchicalResearchPaperDataset(
        texts, labels_array, tokenizer, features_array,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Create dataloader
    batch_size = 1  # Same as training
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        # Load the saved model state dict
        model_state = torch.load(model_path, map_location=device)
        print("Model loaded successfully!")
        
        # We need to import the model class - for now we'll assume it's available
        # In a real scenario, you'd import it from the training script
        from withSmoteTransformerEcon import HierarchicalBiasPredictionModel
        
        # Initialize model with same parameters as training
        model = HierarchicalBiasPredictionModel(
            bert_model_name='allenai/scibert_scivocab_uncased',
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.3
        )
        
        # Load the state dict
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and is compatible")
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
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"="*50)
    print(f"Total papers evaluated: {len(all_labels)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClass Distribution in Test Set:")
    test_distribution = Counter(all_labels)
    for label, count in sorted(test_distribution.items()):
        print(f"  {label_names[label]}: {count} papers")
    
    print(f"\nPredicted Class Distribution:")
    pred_distribution = Counter(all_preds)
    for label, count in sorted(pred_distribution.items()):
        print(f"  {label_names[label]}: {count} papers")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print("="*50)
    try:
        report = classification_report(all_labels, all_preds, 
                                     target_names=label_names,
                                     digits=4)
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print("="*50)
    try:
        cm = confusion_matrix(all_labels, all_preds)
        print("Actual \\ Predicted:", end="")
        for i, name in enumerate(label_names):
            print(f"{name:>15}", end="")
        print()
        
        for i, (actual_label, row) in enumerate(zip(label_names, cm)):
            print(f"{actual_label:>15}", end="")
            for val in row:
                print(f"{val:>15}", end="")
            print()
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    return accuracy, all_preds, all_labels


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature names (same as training - economics specific)
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'coefficient_count', 'hedge_ratio', 'certainty_ratio', 
        'theory_term_ratio', 'jargon_term_ratio', 'econometric_method_count',
        'abstract_claim_ratio', 'results_p_value_density',
        'limitations_mentioned', 'robustness_check_ratio', 'claim_consistency',
        'figure_mentions', 'table_mentions', 
        'citation_but_count', 'self_reference_count', 'method_limitation_ratio'
    ]
    
    # Load papers from JSON
    papers_df = load_papers_from_json('random_papers.json')
    
    if len(papers_df) == 0:
        print("No papers found in the JSON file. Exiting.")
        return
    
    # Use all papers from random_papers.json (already sampled by split_dataset.py)
    sampled_papers = papers_df
    
    # Evaluate the model
    model_path = 'economics.pt'
    accuracy, predictions, true_labels = evaluate_model(
        model_path, sampled_papers, feature_names, device
    )
    
    if accuracy is not None:
        print(f"\nFinal Accuracy on {len(sampled_papers)} papers: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print("Evaluation failed. Please check the model file and dependencies.")


if __name__ == "__main__":
    main()