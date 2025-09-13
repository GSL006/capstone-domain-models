# Bias Detection Models - Complete Workflow Guide

This directory contains machine learning models for bias detection in academic research papers across different domains (Economics and Computer Science).

## 🚀 Quick Start Workflow

### General Workflow (3 Steps)

#### Step 1: Prepare Dataset
```bash
cd capstone-domain-models/{domain}  # domain = econ or comp
```

**For Computer Science (5,258 papers):**
```bash
python split_dataset.py  # Creates train/test split
```

**For Economics:**
```bash
# Dataset already prepared as economics_papers.json
```

#### Step 2: Train Model
```bash
python withSmoteTransformer.py
```
**Creates:** `best_hierarchical_model.pt` (trained model)

#### Step 3: Evaluate Model
```bash
python evaluate_model.py  # (available for comp/, create for econ/)
```
**Shows:** Accuracy metrics and performance analysis

## 📁 Directory Structure

```
capstone-domain-models/
├── README.md                    # This workflow guide
├── download.py                  # Shared NLTK downloader
├── econ/                        # Economics models
│   ├── withSmoteTransformer.py  # Economics bias detection model
│   ├── economics_papers.json    # Economics dataset
│   ├── best_model.pt           # Trained model weights
│   ├── best_hierarchical_model.pt # Hierarchical model weights
│   ├── first_1000.json         # Subset for testing
│   ├── download.py             # Economics NLTK setup
│   └── shortener.py            # Economics data utilities
└── comp/                       # Computer Science models
    ├── withSmoteTransformer.py # CS bias detection model
    ├── computer_science_papers.json # Full CS dataset (5,258 papers)
    ├── split_dataset.py        # Dataset splitter
    ├── evaluate_model.py       # Model evaluation script
    ├── cs_train.json          # Training set (created by split)
    ├── cs_test.json           # Test set (created by split)
    ├── best_hierarchical_model.pt # Trained model (created by training)
    └── download.py             # CS NLTK setup
```

## 📊 Expected Results

### Model Performance (Both Domains)
After running the complete workflow, expect:
- **Training Accuracy**: 75-85% 
- **Test Accuracy**: 65-75% (on unseen papers)
- **Publication Bias Detection**: Highest accuracy (most samples)
- **No Bias Detection**: Lowest accuracy (fewest samples)

### Generated Files
- **Model**: `best_hierarchical_model.pt` (~400-450MB)
- **Charts**: `classification_performance.png`, `confusion_matrix.png`
- **Feature Analysis**: `{domain}_feature_importance.png`

### Dataset Sizes
- **Computer Science**: 5,258 papers → 4,000 train + 1,258 test
- **Economics**: Variable size → 80% train + 20% test (automatic split)

## 🎯 What the Models Detect

### Domain-Specific Features

**Economics Model:**
- **Statistical Patterns**: P-values, significance stars, coefficients
- **Economic Jargon**: Econometric methods (OLS, 2SLS, IV, GMM)
- **Theory References**: Neoclassical, Keynesian, Austrian economics
- **Robustness**: Sensitivity analysis, alternative specifications

**Computer Science Model:**
- **Performance Metrics**: Accuracy, precision, recall, F1-score patterns
- **CS Methods**: CNN, RNN, LSTM, BERT, SVM, Random Forest
- **Evaluation Terms**: Cross-validation, benchmarks, baselines
- **Comparison Patterns**: "Outperforms", "state-of-the-art" claims

### Common Bias Indicators (Both Domains)
- **Language Bias**: Hedge words vs. certainty claims
- **Statistical Reporting**: Cherry-picked results, missing baselines
- **Self-Reference**: Overuse of "our method/approach/results"
- **Limitations**: Acknowledgment vs. omission of study limitations

## 🏗️ Model Architecture

Both models use identical hierarchical architecture:
1. **SciBERT Base**: Scientific paper-trained transformer
2. **Hierarchical Processing**: Document → Sections → Sentences → Words
3. **Multi-level Attention**: Word, sentence, and section-level attention
4. **Feature Fusion**: Combines BERT embeddings + handcrafted features
5. **SMOTE Balancing**: Handles class imbalance in training data
6. **CPU Optimization**: Memory-efficient for local execution

## 📋 Data Format

Both models expect JSON files with this structure:
```json
[
    {
        "Body": "Full paper text...",
        "OverallBias": "Publication Bias",  // or "Cognitive Bias", "No Bias"
        "CognitiveBias": 18.0,
        "PublicationBias": 70.0,
        "NoBias": 12.0
    }
]
```

## 🎯 Model Output

The models classify papers into three categories:
- **0**: No Bias
- **1**: Cognitive Bias  
- **2**: Publication Bias

## 📦 Requirements

```bash
pip install torch transformers scikit-learn imbalanced-learn nltk pandas numpy matplotlib seaborn
```

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
2. **Balanced Classes**: Use SMOTE balancing (automatic in both models)
3. **Proper Split**: Keep train/test separate 
   - **CS**: Use `split_dataset.py` for clean 4K/1.2K split
   - **Economics**: Model handles automatic 80/20 split
4. **Feature Analysis**: Check domain-specific feature importance plots
5. **Multiple Runs**: Train multiple times for stable results

## 🚀 Next Steps

After getting good accuracy:
1. **Deploy Model**: Use `evaluate_model.py` as template for inference
2. **Expand Dataset**: Add more papers for better generalization  
3. **Fine-tune Features**: Modify feature extractors for your specific needs
4. **Cross-Domain Testing**: Test economics model on CS papers and vice versa
5. **New Domains**: Adapt feature extractors for other academic fields
