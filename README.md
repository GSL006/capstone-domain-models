# Bias Detection Models - Complete Workflow Guide

This directory contains machine learning models for bias detection in academic research papers across different domains (Computer Science, Economics, Technology, and Business).

## 🚀 Quick Setup (One-time Installation)

### Automated Setup (Recommended)
```bash
# Navigate to the project directory
cd capstone-domain-models/

# Run the initialization script
./init.sh
```

The `init.sh` script will:
- ✅ Check Python 3.8+ installation
- 📦 Optionally create virtual environment
- 📚 Install all dependencies from `requirements.txt`
- 📥 Download required NLTK data
- 🔍 Verify all domain directories exist
- 💾 Display system information (GPU/RAM)

### 🔧 Virtual Environment Activation

#### **Windows (CMD):**

```
venv\Scripts\activate
```

#### **Windows (PowerShell):**

```
venv\Scripts\Activate.ps1
```

#### **macOS / Linux:**

```
source venv/bin/activate
```

### Manual Setup (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python download.py
```

## 🚀 Quick Start Workflow

### General Workflow (3 Steps)

#### Step 1: Generate Test Papers
```bash
cd capstone-domain-models/{domain}  # domain = comp, econ, tech, or business
python split_dataset.py             # Creates random_papers.json (5 papers)
```

#### Step 2: Run Bias Detection
```bash
python evaluate.py                  # Outputs 5 predictions (one per line)
```
**Example Output:**
```
No Bias
Cognitive Bias
Publication Bias
No Bias
Publication Bias
```

#### Step 3: Train New Model (Optional)
```bash
python withSmoteTransformer.py      # Domain-specific training script
```
**Creates:** Domain-specific trained model (e.g., `best_hierarchical_model.pt`)

## 📁 Directory Structure

```
capstone-domain-models/
├── README.md                      # This workflow guide
├── requirements.txt               # Python dependencies
├── init.sh                       # Automated setup script
├── download.py                   # Shared NLTK downloader
├── comp/                         # Computer Science models
│   ├── withSmoteTransformerComp.py # CS bias detection model
│   ├── computer_science_papers.json # Full CS dataset
│   ├── split_dataset.py          # Random paper extractor
│   ├── evaluate.py               # Model evaluation script
│   ├── random_papers.json        # Random test papers (generated)
│   └── computer_science.pt       # Trained model
├── econ/                         # Economics models
│   ├── withSmoteTransformerEcon.py # Economics bias detection model
│   ├── economics_papers.json     # Economics dataset
│   ├── split_dataset.py          # Random paper extractor
│   ├── evaluate.py               # Model evaluation script
│   ├── random_papers.json        # Random test papers (generated)
│   ├── economics.pt              # Trained model
│   └── best_hierarchical_model.pt # Alternative model
├── tech/                         # Technology models
│   ├── withSmoteTransTech.py     # Technology bias detection model
│   ├── technology_papers.json    # Technology dataset
│   ├── split_dataset.py          # Random paper extractor
│   ├── evaluate.py               # Model evaluation script
│   ├── random_papers.json        # Random test papers (generated)
│   └── technology.pt             # Trained model
└── business/                     # Business models
    ├── withSmoteTransBusiness.py # Business bias detection model
    ├── business_papers.json      # Business dataset (202,918 papers)
    ├── split_dataset.py          # Random paper extractor
    ├── evaluate.py               # Model evaluation script
    ├── random_papers.json        # Random test papers (generated)
    └── best_hierarchical_business_model.pt # Trained model
```

## 📊 Expected Results

### Model Performance (All Domains)
After running the complete workflow, expect:
- **Training Accuracy**: 75-85% 
- **Test Accuracy**: 65-75% (on unseen papers)
- **Publication Bias Detection**: Highest accuracy (most samples)
- **No Bias Detection**: Lowest accuracy (fewest samples)

### Generated Files
- **Test Papers**: `random_papers.json` (5 random papers)
- **Predictions**: Console output (5 predictions, one per line)
- **Training Models**: Domain-specific `.pt` files (~400-450MB)
- **Charts**: `classification_performance.png`, `confusion_matrix.png` (if training)
- **Feature Analysis**: `{domain}_feature_importance.png` (if training)

### Dataset Sizes
- **Computer Science**: 5,258 papers
- **Economics**: Variable size
- **Technology**: Variable size  
- **Business**: 202,918 papers (largest dataset)

## 🎯 What the Models Detect

### Domain-Specific Features

**Computer Science Model (22 features):**
- **Performance Metrics**: Accuracy, precision, recall, F1-score patterns
- **CS Methods**: CNN, RNN, LSTM, BERT, SVM, Random Forest
- **Evaluation Terms**: Cross-validation, benchmarks, baselines
- **Comparison Patterns**: "Outperforms", "state-of-the-art" claims

**Economics Model (20 features):**
- **Statistical Patterns**: P-values, significance stars, coefficients
- **Economic Jargon**: Econometric methods (OLS, 2SLS, IV, GMM)
- **Theory References**: Neoclassical, Keynesian, Austrian economics
- **Robustness**: Sensitivity analysis, alternative specifications

**Technology Model (22 features):**
- **Tech Performance**: Efficiency, throughput, latency, scalability
- **Technology Stack**: Deep learning, AI, algorithms, frameworks
- **Innovation Terms**: "Cutting-edge", "next-generation", "disruptive"
- **Validation Methods**: Benchmarking, testing, simulation

**Business Model (22 features):**
- **Business Metrics**: ROI, performance, growth, profitability
- **Management Terms**: Strategy, competitive advantage, synergies
- **Research Methods**: Surveys, case studies, interviews
- **Business Jargon**: "Leverage", "paradigm shift", "best practices"

### Common Bias Indicators (All Domains)
- **Language Bias**: Hedge words vs. certainty claims
- **Statistical Reporting**: Cherry-picked results, missing baselines
- **Self-Reference**: Overuse of "our method/approach/results"
- **Limitations**: Acknowledgment vs. omission of study limitations
- **Citation Patterns**: Contradictory references ("but" usage)
- **Visual Elements**: Over-reliance on figures/tables without context

## 🏗️ Model Architecture

All models use identical hierarchical architecture:
1. **SciBERT Base**: Scientific paper-trained transformer (or BERT fallback)
2. **Hierarchical Processing**: Document → Sections → Sentences → Words
3. **Multi-level Attention**: Word, sentence, and section-level attention
4. **Feature Fusion**: Combines BERT embeddings + domain-specific handcrafted features
5. **SMOTE Balancing**: Handles class imbalance in training data
6. **Memory Optimization**: Efficient processing for CPU/GPU execution
7. **Domain Adaptation**: Specialized feature extractors per domain

## 📋 Data Format

All models expect JSON files with this structure:
```json
[
    {
        "Body": "Full paper text...",
        "Overall Bias": "Publication Bias",  // or "Cognitive Bias", "No Bias"
        "Reason": "Justification for bias classification...",
        "Title": "Paper title (optional)",
        "CognitiveBias": 18.0,
        "PublicationBias": 70.0,
        "NoBias": 12.0
    }
]
```

**Required Fields for Evaluation:**
- `Body`: Full paper text content
- `Reason`: Explanation for bias classification (used for feature extraction)
- `Overall Bias` or `OverallBias`: Target classification (for training only)

## 🎯 Model Output

The models classify papers into three categories:
- **0**: No Bias
- **1**: Cognitive Bias  
- **2**: Publication Bias

## 📦 Requirements

All dependencies are listed in `requirements.txt`. Install with:

```bash
# Using the automated setup script (recommended)
./init.sh

# Or manual installation
pip install -r requirements.txt
python download.py  # Download NLTK data
```

**Core Dependencies:**
- `torch>=1.12.0` - Deep learning framework
- `transformers>=4.21.0` - BERT/SciBERT models
- `scikit-learn>=1.1.0` - ML utilities
- `imbalanced-learn>=0.9.0` - SMOTE for class balancing
- `nltk>=3.7` - Text processing
- `pandas>=1.4.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing

## 🔧 Troubleshooting

### Memory Issues
- Models automatically use CPU-only mode
- Batch size set to 1 for memory efficiency
- Reduced model complexity (6 layers vs 12)

### Data Issues
- Script creates dummy data if files missing
- Handles various JSON structures automatically
- Validates and cleans input data

### Training Issues
- SMOTE handles class imbalance automatically
- Custom balancing for very small classes
- Gradient accumulation for stable training

## 💡 Tips for Best Results

1. **Clean Data**: Ensure paper texts are complete and well-formatted
2. **Use Setup Script**: Run `./init.sh` for automated environment setup
3. **Test Before Training**: Use `evaluate.py` with existing models first
4. **Domain Selection**: Choose the domain that best matches your papers:
   - **comp/**: Computer Science, AI, Machine Learning papers
   - **econ/**: Economics, Finance, Policy papers  
   - **tech/**: Technology, Engineering, Systems papers
   - **business/**: Management, Marketing, Strategy papers
5. **Multiple Predictions**: Test with `random_papers.json` (5 papers) for variety

## 🚀 Next Steps

After testing with existing models:
1. **Evaluate Performance**: Test on your own papers using `evaluate.py`
2. **Train New Models**: Use `withSmoteTransformer.py` scripts for custom training
3. **Cross-Domain Testing**: Test business model on economics papers and vice versa
4. **Expand Datasets**: Add more papers to domain-specific JSON files
5. **Feature Engineering**: Modify domain-specific feature extractors
6. **New Domains**: Create new directories following the existing pattern
