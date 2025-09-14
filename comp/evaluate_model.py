import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
import pandas as pd
from datetime import datetime

# Import the model classes and utilities from the main script
from withSmoteTransformer import (
    HierarchicalBiasPredictionModel, 
    ComputerScienceBiasFeatureExtractor,
    HierarchicalResearchPaperDataset,
    load_papers_from_json
)

def evaluate_saved_model(model_path='best_hierarchical_model.pt', 
                        data_path='cs_test.json'):
    """
    Load a saved model and evaluate its accuracy on test data
    
    Args:
        model_path: Path to the saved .pt model file
        data_path: Path to the test JSON data file
    """
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    print(f"Loading model from {model_path}")
    
    # Set device
    device = torch.device('cpu')
    
    # Feature names
    feature_names = [
        'length', 'avg_word_length', 'p_value_count', 'signif_stars_count',
        'performance_count', 'hedge_ratio', 'certainty_ratio', 'theory_term_ratio',
        'jargon_term_ratio', 'cs_method_count', 'abstract_claim_ratio',
        'results_performance_density', 'limitations_mentioned', 'evaluation_ratio',
        'claim_consistency', 'figure_mentions', 'table_mentions', 'comparison_count', 
        'self_reference_count', 'method_limitation_ratio'
    ]
    
    # Load data
    print("Loading data...")
    papers_df = load_papers_from_json(data_path)
    print(f"Loaded {len(papers_df)} papers")
    
    # Extract features
    print("Extracting features...")
    extractor = ComputerScienceBiasFeatureExtractor()
    handcrafted_features = []
    
    for _, row in papers_df.iterrows():
        features = extractor.extract_features(row['text'])
        handcrafted_features.append(features)
    
    # Convert to arrays
    features_array = np.array(handcrafted_features, dtype=np.float32)
    labels_array = np.array(papers_df['label'], dtype=np.int64)
    
    # Use all data from test file (already split)
    test_texts = papers_df['text'].tolist()
    test_features = features_array
    test_labels = labels_array
    
    print(f"Test set size: {len(test_texts)}")
    
    # Setup tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    
    # Create test dataset (match training parameters)
    test_dataset = HierarchicalResearchPaperDataset(
        test_texts, test_labels, tokenizer, test_features,
        max_seq_length=128, max_sections=3, max_sents=6
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize model architecture (must match training configuration)
    print("Initializing model architecture...")
    try:
        model = HierarchicalBiasPredictionModel(
            'allenai/scibert_scivocab_uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.3  # Match training dropout rate
        )
        print("Using SciBERT model architecture")
    except Exception as e:
        print(f"SciBERT failed ({e}), falling back to BERT-base")
        model = HierarchicalBiasPredictionModel(
            'bert-base-uncased', 
            num_classes=3,
            feature_dim=len(feature_names),
            dropout_rate=0.3  # Match training dropout rate
        )
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Evaluate
    print("Evaluating model...")
    print(f"Processing {len(test_dataloader)} test batches...")
    
    all_preds = []
    all_labels = []
    all_probabilities = []  # Store prediction probabilities
    all_texts = []  # Store original texts for analysis
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx+1}/{len(test_dataloader)} ({(batch_idx+1)/len(test_dataloader)*100:.1f}%)...")
            
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                handcrafted_features = batch['handcrafted_features'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, handcrafted_features)
                probabilities = torch.softmax(outputs, dim=1)  # Get probabilities
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Store corresponding text (get from original index)
                batch_start_idx = batch_idx * test_dataloader.batch_size
                batch_end_idx = min(batch_start_idx + test_dataloader.batch_size, len(test_texts))
                all_texts.extend(test_texts[batch_start_idx:batch_end_idx])
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    print(f"Evaluation completed! Processed {len(all_preds)} samples.")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    label_names = ['No Bias', 'Cognitive Bias', 'Publication Bias']
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Test samples: {len(all_labels)}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    class_report = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    class_accuracies = {}
    for i, label_name in enumerate(label_names):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_accuracy = accuracy_score(
                np.array(all_labels)[class_mask], 
                np.array(all_preds)[class_mask]
            )
            class_accuracies[label_name] = class_accuracy
            print(f"{label_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print("Predicted ->")
    print(f"{'Actual':<12} {'No Bias':<10} {'Cognitive':<10} {'Publication':<10}")
    for i, label_name in enumerate(['No Bias', 'Cognitive', 'Publication']):
        row = f"{label_name:<12}"
        for j in range(3):
            row += f"{cm[i,j]:<10}"
        print(row)
    
    # Create results files
    print("\nCreating results files...")
    
    # Calculate additional metrics
    correct_predictions = sum(1 for i in range(len(all_labels)) if all_labels[i] == all_preds[i])
    incorrect_predictions = len(all_labels) - correct_predictions
    
    # Analyze misclassifications by class
    misclass_by_actual = {}
    for label_name in label_names:
        label_idx = label_names.index(label_name)
        misclass_count = sum(1 for i in range(len(all_labels)) 
                           if all_labels[i] == label_idx and all_preds[i] != label_idx)
        misclass_by_actual[label_name] = misclass_count
    
    # Count confidence-based metrics
    low_confidence_count = sum(1 for probs in all_probabilities if max(probs) < 0.6)
    high_confidence_errors = sum(1 for i in range(len(all_labels)) 
                               if all_labels[i] != all_preds[i] and max(all_probabilities[i]) > 0.8)
    
    # 1. Save comprehensive results to results.json
    results_data = {
        'model_path': model_path,
        'test_data_path': data_path,
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_samples': len(all_labels),
        'overall_accuracy': float(accuracy),
        'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'label_names': label_names,
        'summary_stats': {
            'total_samples': len(all_labels),
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions,
            'accuracy_percentage': (correct_predictions / len(all_labels)) * 100,
            'misclassifications_by_actual_class': misclass_by_actual,
            'low_confidence_predictions': low_confidence_count,
            'high_confidence_errors': high_confidence_errors
        }
    }
    
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print("✓ Complete results saved to 'results.json'")
    
    # 2. Create error analysis CSV with ALL original data + predictions
    # Load original test data to preserve all columns
    original_df = pd.read_json(data_path)
    
    # Add prediction columns to ALL samples (not just errors)
    prediction_data = []
    for i in range(len(all_labels)):
        probs = all_probabilities[i] if i < len(all_probabilities) else [0, 0, 0]
        prediction_data.append({
            'actual_class': int(all_labels[i]),
            'predicted_class': int(all_preds[i]),
            'predicted_class_name': label_names[all_preds[i]],
            'prediction_correct': bool(all_labels[i] == all_preds[i]),
            'confidence_no_bias': float(probs[0]),
            'confidence_cognitive_bias': float(probs[1]),
            'confidence_publication_bias': float(probs[2]),
            'prediction_confidence': float(probs[all_preds[i]]),
            'max_confidence': float(max(probs))
        })
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame(prediction_data)
    
    # Combine original data with predictions
    combined_df = pd.concat([original_df.reset_index(drop=True), pred_df], axis=1)
    
    # Add actual class name for clarity (using the actual_class from predictions)
    combined_df['actual_class_name'] = combined_df['actual_class'].map({0: 'No Bias', 1: 'Cognitive Bias', 2: 'Publication Bias'})
    
    # Filter to only incorrect predictions for error analysis
    errors_df = combined_df[combined_df['prediction_correct'] == False].copy()
    
    if len(errors_df) > 0:
        # Reorder columns for better readability
        error_columns = [
            'Title', 'actual_class_name', 'predicted_class_name', 'prediction_confidence', 'max_confidence',
            'Body', 'actual_class', 'predicted_class', 'prediction_correct',
            'confidence_no_bias', 'confidence_cognitive_bias', 'confidence_publication_bias',
            'Subject', 'Institution', 'Journal', 'Publisher', 'Author', 'Reason'
        ]
        # Only include columns that exist
        available_columns = [col for col in error_columns if col in errors_df.columns]
        remaining_columns = [col for col in errors_df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        errors_df[final_columns].to_csv('error_analysis.csv', index=False, encoding='utf-8')
        print(f"✓ Error analysis saved to 'error_analysis.csv' ({len(errors_df)} errors with all original data)")
    else:
        print("✓ No errors found - perfect predictions!")
    
    print(f"\n{'='*50}")
    print("FILES CREATED:")
    print("• results.json - Complete evaluation metrics, confusion matrix, and summary stats")
    if len(errors_df) > 0:
        print(f"• error_analysis.csv - {len(errors_df)} incorrect predictions with ALL original cs_test.json data + prediction columns")
    print(f"{'='*50}")
    
    # Print quick summary stats
    print(f"\nQUICK SUMMARY:")
    print(f"• Total samples: {len(all_labels)}")
    print(f"• Correct predictions: {correct_predictions} ({(correct_predictions/len(all_labels)*100):.1f}%)")
    print(f"• Incorrect predictions: {incorrect_predictions} ({(incorrect_predictions/len(all_labels)*100):.1f}%)")
    print(f"• Low confidence predictions (<60%): {low_confidence_count}")
    print(f"• High confidence errors (>80%): {high_confidence_errors}")
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'true_labels': all_labels,
        'confusion_matrix': cm,
        'error_count': len(errors_df) if len(errors_df) > 0 else 0
    }

if __name__ == "__main__":
    print("=" * 60)
    print("HIERARCHICAL BIAS PREDICTION MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluate the model
    results = evaluate_saved_model()
    
    if results:
        print(f"\n{'='*30}")
        print(f"FINAL ACCURACY: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"{'='*30}")
        print("\nEvaluation completed successfully!")
        print("Check the generated files for detailed analysis.")
    else:
        print("\nEvaluation failed!")
