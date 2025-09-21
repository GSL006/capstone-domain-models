#!/usr/bin/env python3
"""
Dataset Class Distribution Analyzer
Analyzes the class distribution in CS bias detection dataset
"""

import json
import os
from collections import Counter
import argparse

def analyze_dataset(json_file_path):
    """
    Analyze class distribution in the dataset JSON file
    
    Args:
        json_file_path: Path to the JSON dataset file
    """
    
    print(f"📊 Analyzing Dataset: {json_file_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ Error: File '{json_file_path}' not found!")
        return
    
    try:
        # Load JSON data
        print("📂 Loading JSON data...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded {len(data)} entries")
        
        # Extract bias labels (only for papers with all required fields)
        bias_labels = []
        missing_labels = 0
        empty_labels = 0
        missing_body = 0
        missing_reason = 0
        empty_body = 0
        empty_reason = 0
        
        for i, paper in enumerate(data):
            if not isinstance(paper, dict):
                continue
                
            # Check for bias label (Overall Bias or OverallBias)
            label = paper.get('Overall Bias', paper.get('OverallBias', None))
            
            # Check for Body field
            body = paper.get('Body', None)
            
            # Check for Reason field  
            reason = paper.get('Reason', None)
            
            # Track missing fields
            if label is None:
                missing_labels += 1
                continue
            elif label == "" or label.strip() == "":
                empty_labels += 1
                continue
                
            if body is None:
                missing_body += 1
                continue
            elif body == "" or body.strip() == "":
                empty_body += 1
                continue
                
            if reason is None:
                missing_reason += 1
                continue
            elif reason == "" or reason.strip() == "":
                empty_reason += 1
                continue
            
            # Only add to valid labels if ALL THREE fields are present and non-empty
            bias_labels.append(label.strip())
        
        # Count class distribution
        label_counts = Counter(bias_labels)
        total_valid_labels = len(bias_labels)
        
        print(f"\n📈 CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        # Display results
        print(f"📋 Total entries in file: {len(data)}")
        print(f"✅ Trainable papers (all 3 fields present): {total_valid_labels}")
        print(f"\n📊 FIELD VALIDATION BREAKDOWN:")
        print(f"   ⚠️  Missing 'Overall Bias'/'OverallBias': {missing_labels}")
        print(f"   ⚠️  Empty 'Overall Bias'/'OverallBias': {empty_labels}")
        print(f"   ⚠️  Missing 'Body': {missing_body}")
        print(f"   ⚠️  Empty 'Body': {empty_body}")
        print(f"   ⚠️  Missing 'Reason': {missing_reason}")
        print(f"   ⚠️  Empty 'Reason': {empty_reason}")
        
        excluded_total = missing_labels + empty_labels + missing_body + empty_body + missing_reason + empty_reason
        print(f"   🗑️  Total excluded papers: {excluded_total}")
        print(f"   📈 Inclusion rate: {(total_valid_labels/len(data)*100):.1f}%")
        
        print(f"\n🎯 CLASS BREAKDOWN:")
        print("-" * 60)
        print(f"{'Class Name':<20} {'Count':<10} {'Percentage':<12} {'Visual'}")
        print("-" * 60)
        
        # Sort by count (descending)
        for label, count in label_counts.most_common():
            percentage = (count / total_valid_labels) * 100
            # Create visual bar
            bar_length = int(percentage / 2)  # Scale down for display
            visual_bar = "█" * bar_length + "░" * (50 - bar_length)
            
            print(f"{label:<20} {count:<10} {percentage:<11.2f}% {visual_bar}")
        
        print("-" * 60)
        print(f"{'TOTAL':<20} {total_valid_labels:<10} {'100.00%':<12}")
        
        # Class imbalance analysis
        print(f"\n⚖️  CLASS IMBALANCE ANALYSIS:")
        print("-" * 60)
        
        if len(label_counts) > 1:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count
            
            print(f"📊 Largest class: {max_count} samples")
            print(f"📊 Smallest class: {min_count} samples")
            print(f"📊 Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 10:
                print("🚨 SEVERE imbalance detected (>10:1 ratio)")
            elif imbalance_ratio > 3:
                print("⚠️  MODERATE imbalance detected (>3:1 ratio)")
            else:
                print("✅ BALANCED dataset (<3:1 ratio)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 60)
        
        if missing_labels > 0:
            print(f"🔧 Consider handling {missing_labels} entries with missing labels")
        
        if empty_labels > 0:
            print(f"🔧 Consider handling {empty_labels} entries with empty labels")
        
        if len(label_counts) > 1:
            if imbalance_ratio > 5:
                print("🔧 Consider using SMOTE or class weights for severe imbalance")
                print("🔧 Consider stratified sampling for train/test splits")
            
            minority_classes = [label for label, count in label_counts.items() 
                              if count < total_valid_labels * 0.1]  # <10% of data
            if minority_classes:
                print(f"🔧 Pay special attention to minority classes: {minority_classes}")
        
        # Export summary
        summary = {
            "file_path": json_file_path,
            "total_entries": len(data),
            "trainable_papers": total_valid_labels,
            "inclusion_rate_percent": (total_valid_labels/len(data)*100),
            "field_validation": {
                "missing_bias_labels": missing_labels,
                "empty_bias_labels": empty_labels,
                "missing_body": missing_body,
                "empty_body": empty_body,
                "missing_reason": missing_reason,
                "empty_reason": empty_reason,
                "total_excluded": excluded_total
            },
            "class_distribution": dict(label_counts),
            "class_percentages": {label: (count/total_valid_labels)*100 
                                for label, count in label_counts.items()},
            "imbalance_ratio": max(label_counts.values()) / min(label_counts.values()) if len(label_counts) > 1 else 1.0
        }
        
        # Save summary to JSON
        summary_file = json_file_path.replace('.json', '_class_analysis.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed analysis saved to: {summary_file}")
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in '{json_file_path}'")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Analyze class distribution in CS bias dataset')
    parser.add_argument('json_file', nargs='?', default='computer_science_papers.json',
                       help='Path to JSON dataset file (default: computer_science_papers.json)')
    
    args = parser.parse_args()
    
    # Analyze the specified file
    analyze_dataset(args.json_file)
    
    # Also analyze train and test files if they exist
    train_file = 'cs_train.json'
    test_file = 'cs_test.json'
    
    if os.path.exists(train_file) and args.json_file != train_file:
        print(f"\n" + "="*80)
        analyze_dataset(train_file)
    
    if os.path.exists(test_file) and args.json_file != test_file:
        print(f"\n" + "="*80)
        analyze_dataset(test_file)

if __name__ == "__main__":
    main()
