import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import json
from collections import Counter
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import random
import pickle

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
                start_pos = text_lower.find(marker)
            
            if start_pos != -1:
                # Find end position (start of next section or end of text)
                end_pos = len(text)
                for j in range(i + 1, len(section_markers)):
                    next_marker_pos = text_lower.find('\n' + section_markers[j] + '\n')
                    if next_marker_pos == -1:
                        next_marker_pos = text_lower.find(section_markers[j])
                    if next_marker_pos != -1 and next_marker_pos > start_pos:
                        end_pos = next_marker_pos
                        break
                
                section_text = text[start_pos:end_pos].strip()
                if section_text:
                    sections.append(section_text)
        
        # If no sections were found, treat the entire text as one section
        if not sections:
            sections = [text]
        
        return sections


def load_papers_from_json(json_file_path):
    """
    Load business research papers from a JSON file.
    
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
        
        print(f"Successfully loaded {len(df)} business papers from {json_file_path}")
        return df
        
    except json.JSONDecodeError as e:
        print(f"Error: {json_file_path} is not a valid JSON file: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"Error loading business papers from {json_file_path}: {str(e)}")
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
    print("Extracting business-specific handcrafted features...")
    extractor = BusinessBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    # Convert to numpy arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Note: For evaluation, we need to use the same scaler that was used during training
    # For now, we'll normalize with StandardScaler (ideally we'd save the scaler from training)
    print("Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
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
    eval_dataset = HierarchicalBusinessPaperDataset(
        texts, labels_array, tokenizer, features_array,
        max_seq_length=max_seq_length,
        max_sections=max_sections,
        max_sents=max_sents
    )
    
    # Create dataloader
    batch_size = 8  # Same as training
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        # Load the saved model state dict
        model_state = torch.load(model_path, map_location=device)
        print("Model loaded successfully!")
        
        # We need to import the model class - for now we'll assume it's available
        # In a real scenario, you'd import it from the training script
        from withSmoteTransBusiness import HierarchicalBiasPredictionModel
        
        # Initialize model with same parameters as training
        model = HierarchicalBiasPredictionModel(
            bert_model_name='allenai/scibert_scivocab_uncased',
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.5
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
            
            # Get predictions
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
            print(f"\t{name[:10]}", end="")
        print()
        
        for i, (actual_label, row) in enumerate(zip(label_names, cm)):
            print(f"{actual_label[:15]:<15}", end="")
            for count in row:
                print(f"\t{count}", end="")
            print()
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    return accuracy, all_preds, all_labels


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature names (same as training)
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'correlation_count', 'percentage_count', 'hedge_ratio', 'certainty_ratio', 
        'theory_term_ratio', 'jargon_term_ratio', 'method_term_count',
        'validation_pattern_count', 'abstract_claim_ratio', 'results_stat_density',
        'limitations_mentioned', 'performance_claim_ratio', 'claim_consistency',
        'figure_mentions', 'table_mentions', 'chart_mentions', 
        'citation_but_count', 'self_reference_count'
    ]
    
    # Load papers from JSON
    papers_df = load_papers_from_json('random_papers.json')
    
    if len(papers_df) == 0:
        print("No papers found in the JSON file. Exiting.")
        return
    
    # Use all papers from random_papers.json (already sampled by split_dataset.py)
    sampled_papers = papers_df
    
    # Evaluate the model
    model_path = 'business.pt'
    result = evaluate_model(
        model_path, sampled_papers, feature_names, device
    )
    
    if result is not None:
        accuracy, predictions, true_labels = result
        print(f"\nFinal Accuracy on {len(sampled_papers)} papers: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print("Evaluation failed. Please check the model file and dependencies.")


if __name__ == "__main__":
    main()