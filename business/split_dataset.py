import json
import random
import os
from collections import Counter

def extract_random_entries(input_file='business_papers.json', 
                          output_file='random_papers.json', num_papers=1000):
    """
    Extract random entries from the business dataset that have all required fields
    and save them to a new JSON file.
    """
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total papers in {input_file}: {len(data)}")
        
        # Find papers with all required fields
        valid_papers = []
        missing_bias = 0
        missing_body = 0
        missing_reason = 0
        
        for i, paper in enumerate(data):
            if not isinstance(paper, dict):
                continue
            
            # Check for Overall Bias field (try both variants)
            bias_label = paper.get('Overall Bias') or paper.get('OverallBias')
            if not bias_label or not bias_label.strip():
                missing_bias += 1
                continue
            
            # Check for Body field
            body = paper.get('Body', '')
            if not body or not body.strip():
                missing_body += 1
                continue
            
            # Check for Reason field
            reason = paper.get('Reason', '')
            if not reason or not reason.strip():
                missing_reason += 1
                continue
            
            # If we reach here, all fields are present and non-empty
            valid_papers.append((i, paper, bias_label))
        
        print(f"Papers missing Overall Bias: {missing_bias}")
        print(f"Papers missing Body: {missing_body}")
        print(f"Papers missing Reason: {missing_reason}")
        print(f"Valid papers with all fields: {len(valid_papers)}")
        
        if len(valid_papers) == 0:
            print("No valid papers found with all required fields!")
            return False
        
        # Count bias distribution in valid papers
        bias_counts = Counter([bias for _, _, bias in valid_papers])
        print(f"\nBias distribution in valid papers:")
        for bias, count in bias_counts.items():
            print(f"  {bias}: {count}")
        
        # Randomly select papers (up to num_papers or all available if fewer)
        num_to_select = min(num_papers, len(valid_papers))
        selected_papers_info = random.sample(valid_papers, num_to_select)
        
        print(f"\nSelected {num_to_select} papers:")
        output_data = []
        for i, (idx, paper, bias) in enumerate(selected_papers_info, 1):
            print(f"  Paper {i} (#{idx}):")
            print(f"    Title: {paper.get('Title', 'N/A')}")
            print(f"    Overall Bias: {bias}")
            print(f"    Body length: {len(paper.get('Body', ''))}")
            print(f"    Reason length: {len(paper.get('Reason', ''))}")
            output_data.append(paper)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{num_to_select} random papers saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    """Main function to extract random entries"""
    success = extract_random_entries()
    
    if success:
        print("✅ Random papers extraction completed successfully!")
    else:
        print("❌ Failed to extract random papers")

if __name__ == "__main__":
    main()
