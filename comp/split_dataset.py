import json
import numpy as np
from collections import Counter


def extract_single_random_entry(input_file='computer_science_papers.json', 
                               output_file='single_random_paper.json'):
    """
    Extract a single random entry from the computer science dataset
    
    Args:
        input_file: Input JSON file with all papers
        output_file: Output file for the single random paper
    """
    
    print(f"Loading {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total papers loaded: {len(data)}")
        
        # Filter papers to only include those with all required fields
        valid_papers = []
        for i, paper in enumerate(data):
            if not isinstance(paper, dict):
                continue
                
            # Check for bias label (Overall Bias or OverallBias)
            bias_label = paper.get('Overall Bias') or paper.get('OverallBias')
            
            # Check for Body field
            body = paper.get('Body', '')
            
            # Check for Reason field  
            reason = paper.get('Reason', '')
            
            # Only include papers with all three fields present and non-empty
            if bias_label and bias_label.strip() and body and body.strip() and reason and reason.strip():
                valid_papers.append((i, paper, bias_label))
        
        print(f"Valid papers (with all required fields): {len(valid_papers)}")
        
        if len(valid_papers) == 0:
            print("Error: No valid papers found!")
            return False
        
        # Randomly select 1 paper
        np.random.seed(42)  # For reproducibility
        random_index = np.random.randint(0, len(valid_papers))
        selected_paper_info = valid_papers[random_index]
        
        original_index, selected_paper, bias_label = selected_paper_info
        
        print(f"\nRandomly selected paper:")
        print(f"  Original index: {original_index}")
        print(f"  Title: {selected_paper.get('Title', 'No title')}")
        print(f"  Bias label: {bias_label}")
        print(f"  Body length: {len(selected_paper.get('Body', ''))} characters")
        print(f"  Reason length: {len(selected_paper.get('Reason', ''))} characters")
        
        # Save the single paper to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([selected_paper], f, indent=2, ensure_ascii=False)
        
        print(f"\nSingle random paper saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return False


if __name__ == "__main__":
    print("Computer Science Dataset - Single Random Entry Extractor")
    print("=" * 60)
    
    # Extract a single random entry from the main dataset
    success = extract_single_random_entry()
    
    if success:
        print("\n" + "=" * 60)
        print("Single random entry extraction completed!")
        print("\nFile created:")
        print("- single_random_paper.json (1 random paper for testing)")
    else:
        print("Single entry extraction failed!")
