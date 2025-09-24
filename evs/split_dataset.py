#!/usr/bin/env python3
"""
Split Dataset Script for Environmental Science Domain
Extracts random papers from environmental_science_papers.json for testing
"""

import json
import random
import os

def extract_random_entries(input_file='environmental_science_papers.json', 
                          output_file='random_papers.json', num_papers=1000):
    """
    Extract random entries from the environmental science dataset.
    
    Parameters:
    -----------
    input_file : str
        Path to the main environmental science dataset JSON file
    output_file : str  
        Path where random papers will be saved
    num_papers : int
        Number of papers to extract (default: 1000)
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Error: Input file '{input_file}' not found!")
            return False
        
        print(f"📖 Loading papers from: {input_file}")
        
        # Load the dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        papers_list = []
        if isinstance(data, list):
            papers_list = data
        elif isinstance(data, dict):
            if 'papers' in data:
                papers_list = data['papers']
            else:
                # Convert dict to list
                papers_list = list(data.values())
        
        print(f"📊 Total papers in dataset: {len(papers_list)}")
        
        # Filter papers with required fields
        valid_papers = []
        for idx, paper in enumerate(papers_list):
            if not isinstance(paper, dict):
                continue
                
            # Check for bias label (multiple possible field names)
            bias_label = (
                paper.get('Overall Bias') or 
                paper.get('OverallBias') or
                paper.get('bias_label') or
                paper.get('bias_type')
            )
            
            # Check for text content
            text_content = (
                paper.get('Body') or 
                paper.get('text') or 
                paper.get('content') or
                paper.get('Abstract') or
                paper.get('abstract')
            )
            
            # Validate fields
            if bias_label and text_content:
                if isinstance(text_content, str) and len(text_content.strip()) > 50:
                    valid_papers.append((idx, paper, bias_label))
        
        print(f"✅ Valid papers (with required fields): {len(valid_papers)}")
        
        if len(valid_papers) == 0:
            print("❌ No valid papers found with required fields!")
            return False
        
        # Randomly select papers (up to num_papers or all available if fewer)
        num_to_select = min(num_papers, len(valid_papers))
        selected_papers_info = random.sample(valid_papers, num_to_select)
        
        print(f"\n🎯 Selected {num_to_select} papers:")
        output_data = []
        for i, (idx, paper, bias) in enumerate(selected_papers_info, 1):
            print(f"  Paper {i} (#{idx}):")
            print(f"    Title: {paper.get('Title', paper.get('title', 'N/A'))}")
            print(f"    Bias Label: {bias}")
            
            # Get text content
            text_content = (
                paper.get('Body') or 
                paper.get('text') or 
                paper.get('content') or
                paper.get('Abstract') or
                paper.get('abstract', '')
            )
            print(f"    Text length: {len(text_content)}")
            output_data.append(paper)
        
        # Save to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 {num_to_select} random papers saved to: {output_file}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error extracting papers: {e}")
        return False

def main():
    """Main function to extract random papers"""
    print("🌍 Environmental Science Papers - Random Extractor")
    print("=" * 50)
    
    success = extract_random_entries(
        input_file='environmental_science_papers.json',
        output_file='random_papers.json',
        num_papers=1000
    )
    
    if success:
        print("\n🎉 Random paper extraction completed successfully!")
        print("\n📋 Next steps:")
        print("  1. Run: python evaluate.py")
        print("  2. View predictions for the 1000 random papers")
    else:
        print("\n❌ Failed to extract random papers")
        print("Please check if 'environmental_science_papers.json' exists and is valid")

if __name__ == "__main__":
    main()
