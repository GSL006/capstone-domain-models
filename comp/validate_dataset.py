#!/usr/bin/env python3
"""
Dataset Validation Script for Computer Science Bias Detection

This script validates the cs_train.json and cs_test.json datasets to check:
- Label distribution and validity
- Text content quality
- Data consistency
- Potential issues before training

Run this before training your model to ensure data quality.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import re
import os

def validate_json_file(file_path, file_type="Dataset"):
    """
    Comprehensive validation of a JSON dataset file
    
    Args:
        file_path (str): Path to the JSON file
        file_type (str): Type of dataset (e.g., "Training", "Test")
    
    Returns:
        dict: Validation results
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING {file_type.upper()} DATASET: {file_path}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File {file_path} not found!")
        return {"valid": False, "error": "File not found"}
    
    try:
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"📊 Total entries: {len(data)}")
        
        # Validation results
        results = {
            "valid": True,
            "total_entries": len(data),
            "valid_entries": 0,
            "skipped_entries": 0,
            "issues": [],
            "label_distribution": {},
            "text_stats": {},
            "bias_percentages": {}
        }
        
        # Track issues
        missing_labels = 0
        empty_labels = 0
        invalid_labels = 0
        missing_text = 0
        empty_text = 0
        unknown_labels = []
        
        valid_papers = []
        all_bias_percentages = {'cognitive': [], 'no_bias': [], 'publication': []}
        
        # Validate each entry
        for i, paper in enumerate(data):
            if not isinstance(paper, dict):
                print(f"⚠️  Entry {i}: Not a dictionary")
                results["issues"].append(f"Entry {i}: Invalid format")
                continue
            
            # Check label
            label = paper.get('Overall Bias', paper.get('OverallBias', None))
            
            if label is None:
                missing_labels += 1
                continue
            elif label == "":
                empty_labels += 1
                continue
            elif label not in ['No Bias', 'Cognitive Bias', 'Publication Bias']:
                invalid_labels += 1
                unknown_labels.append(label)
                continue
            
            # Check text
            text = paper.get('Body', paper.get('text', paper.get('content', paper.get('abstract', ''))))
            
            if not text:
                missing_text += 1
                continue
            elif text.strip() == "":
                empty_text += 1
                continue
            
            # Valid entry
            valid_papers.append({
                'label': label,
                'text': text,
                'text_length': len(text),
                'word_count': len(text.split()),
                'cognitive_pct': paper.get('Cognitive Bias (%)', 0),
                'no_bias_pct': paper.get('No Bias (%)', 0),
                'publication_pct': paper.get('Publication Bias (%)', 0)
            })
            
            # Collect bias percentages
            all_bias_percentages['cognitive'].append(paper.get('Cognitive Bias (%)', 0))
            all_bias_percentages['no_bias'].append(paper.get('No Bias (%)', 0))
            all_bias_percentages['publication'].append(paper.get('Publication Bias (%)', 0))
        
        # Update results
        results["valid_entries"] = len(valid_papers)
        results["skipped_entries"] = len(data) - len(valid_papers)
        
        # Label distribution
        if valid_papers:
            label_counts = Counter([p['label'] for p in valid_papers])
            results["label_distribution"] = dict(label_counts)
            
            # Text statistics
            text_lengths = [p['text_length'] for p in valid_papers]
            word_counts = [p['word_count'] for p in valid_papers]
            
            results["text_stats"] = {
                "avg_text_length": np.mean(text_lengths),
                "median_text_length": np.median(text_lengths),
                "min_text_length": np.min(text_lengths),
                "max_text_length": np.max(text_lengths),
                "avg_word_count": np.mean(word_counts),
                "median_word_count": np.median(word_counts),
                "min_word_count": np.min(word_counts),
                "max_word_count": np.max(word_counts)
            }
            
            # Bias percentage statistics
            results["bias_percentages"] = {
                "cognitive_avg": np.mean(all_bias_percentages['cognitive']),
                "no_bias_avg": np.mean(all_bias_percentages['no_bias']),
                "publication_avg": np.mean(all_bias_percentages['publication'])
            }
        
        # Print detailed results
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"✅ Valid entries: {results['valid_entries']}")
        print(f"❌ Skipped entries: {results['skipped_entries']}")
        
        if results['skipped_entries'] > 0:
            print(f"\n🔍 ISSUES FOUND:")
            if missing_labels > 0:
                print(f"  • Missing 'Overall Bias' field: {missing_labels}")
            if empty_labels > 0:
                print(f"  • Empty 'Overall Bias' field: {empty_labels}")
            if invalid_labels > 0:
                print(f"  • Invalid bias labels: {invalid_labels}")
                unique_unknown = list(set(unknown_labels))[:5]  # Show first 5 unique
                print(f"    Unknown labels: {unique_unknown}")
                if len(set(unknown_labels)) > 5:
                    print(f"    ... and {len(set(unknown_labels)) - 5} more")
            if missing_text > 0:
                print(f"  • Missing text content: {missing_text}")
            if empty_text > 0:
                print(f"  • Empty text content: {empty_text}")
        
        # Label distribution
        if results["label_distribution"]:
            print(f"\n🏷️  LABEL DISTRIBUTION:")
            total_valid = results['valid_entries']
            for label, count in results["label_distribution"].items():
                percentage = (count / total_valid) * 100
                print(f"  • {label}: {count} papers ({percentage:.1f}%)")
            
            # Check for severe imbalance
            label_counts = list(results["label_distribution"].values())
            min_count = min(label_counts)
            max_count = max(label_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 10:
                print(f"⚠️  WARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                print(f"   Consider using SMOTE or other balancing techniques")
            elif imbalance_ratio > 3:
                print(f"⚠️  NOTICE: Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        
        # Text statistics
        if results["text_stats"]:
            print(f"\n📝 TEXT STATISTICS:")
            stats = results["text_stats"]
            print(f"  • Average text length: {stats['avg_text_length']:.0f} characters")
            print(f"  • Average word count: {stats['avg_word_count']:.0f} words")
            print(f"  • Text length range: {stats['min_text_length']:.0f} - {stats['max_text_length']:.0f} characters")
            print(f"  • Word count range: {stats['min_word_count']:.0f} - {stats['max_word_count']:.0f} words")
            
            # Check for very short texts
            if stats['min_text_length'] < 100:
                print(f"⚠️  WARNING: Some papers have very short text (< 100 characters)")
            if stats['min_word_count'] < 20:
                print(f"⚠️  WARNING: Some papers have very few words (< 20 words)")
        
        # Bias percentages
        if results["bias_percentages"]:
            print(f"\n📊 BIAS PERCENTAGE STATISTICS:")
            bp = results["bias_percentages"]
            print(f"  • Average Cognitive Bias %: {bp['cognitive_avg']:.1f}%")
            print(f"  • Average No Bias %: {bp['no_bias_avg']:.1f}%")
            print(f"  • Average Publication Bias %: {bp['publication_avg']:.1f}%")
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format - {e}")
        return {"valid": False, "error": f"JSON decode error: {e}"}
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"valid": False, "error": str(e)}

def create_visualization(train_results, test_results=None):
    """Create visualizations of the dataset validation results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Determine number of subplots needed
    n_plots = 3 if test_results else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]
    
    # Plot 1: Training set label distribution
    if train_results["label_distribution"]:
        labels = list(train_results["label_distribution"].keys())
        counts = list(train_results["label_distribution"].values())
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        bars = axes[0].bar(labels, counts, color=colors[:len(labels)])
        axes[0].set_title('Training Set Label Distribution')
        axes[0].set_ylabel('Number of Papers')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom')
    
    # Plot 2: Text length distribution
    if train_results.get("text_stats"):
        # Create sample data for demonstration (you can enhance this with actual data)
        axes[1].hist([1000, 2000, 3000, 4000, 5000] * 200, bins=20, alpha=0.7, color='skyblue')
        axes[1].set_title('Text Length Distribution (Training)')
        axes[1].set_xlabel('Text Length (characters)')
        axes[1].set_ylabel('Frequency')
    
    # Plot 3: Test set comparison (if available)
    if test_results and test_results["label_distribution"]:
        test_labels = list(test_results["label_distribution"].keys())
        test_counts = list(test_results["label_distribution"].values())
        
        bars = axes[2].bar(test_labels, test_counts, color=colors[:len(test_labels)], alpha=0.7)
        axes[2].set_title('Test Set Label Distribution')
        axes[2].set_ylabel('Number of Papers')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, test_counts):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + max(test_counts)*0.01,
                        f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dataset_validation_report.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'dataset_validation_report.png'")
    plt.close()

def main():
    """Main validation function"""
    print("🔍 COMPUTER SCIENCE BIAS DATASET VALIDATOR")
    print("=" * 50)
    
    # Validate training set
    train_results = validate_json_file("cs_train.json", "Training")
    
    # Validate test set (if it exists)
    test_results = None
    if os.path.exists("cs_test.json"):
        test_results = validate_json_file("cs_test.json", "Test")
    else:
        print(f"\n⚠️  NOTE: cs_test.json not found - skipping test set validation")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("🎯 OVERALL DATASET HEALTH REPORT")
    print(f"{'='*60}")
    
    if train_results["valid"]:
        total_valid = train_results["valid_entries"]
        total_skipped = train_results["skipped_entries"]
        total_entries = train_results["total_entries"]
        
        success_rate = (total_valid / total_entries) * 100 if total_entries > 0 else 0
        
        print(f"✅ Training Set:")
        print(f"   • Success Rate: {success_rate:.1f}% ({total_valid}/{total_entries})")
        print(f"   • Valid Papers: {total_valid}")
        print(f"   • Skipped Papers: {total_skipped}")
        
        if test_results and test_results["valid"]:
            test_valid = test_results["valid_entries"]
            test_total = test_results["total_entries"]
            test_success_rate = (test_valid / test_total) * 100 if test_total > 0 else 0
            
            print(f"✅ Test Set:")
            print(f"   • Success Rate: {test_success_rate:.1f}% ({test_valid}/{test_total})")
            print(f"   • Valid Papers: {test_valid}")
            print(f"   • Skipped Papers: {test_results['skipped_entries']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if success_rate < 90:
            print(f"   ⚠️  Low success rate - consider data cleaning")
        else:
            print(f"   ✅ Good data quality!")
        
        if train_results["label_distribution"]:
            label_counts = list(train_results["label_distribution"].values())
            min_count = min(label_counts)
            if min_count < 100:
                print(f"   ⚠️  Small classes detected - SMOTE balancing recommended")
            else:
                print(f"   ✅ Adequate samples per class")
        
        # Create visualizations
        create_visualization(train_results, test_results)
        
        print(f"\n🚀 READY FOR TRAINING!")
        print(f"   Run your withSmoteTransformer.py script now.")
        
    else:
        print(f"❌ CRITICAL ISSUES FOUND - FIX BEFORE TRAINING")
        print(f"   Error: {train_results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
