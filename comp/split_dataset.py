import json
import numpy as np
from collections import Counter


def extract_random_entries(input_file='computer_science_papers.json', 
                          output_file='random_papers.json', num_papers=1000):
    """
    Extract random entries from the computer science dataset
    
    Args:
        input_file: Input JSON file with all papers
        output_file: Output file for the random papers
        num_papers: Number of papers to extract
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
        
        # Randomly select papers (up to num_papers or all available if fewer)
        np.random.seed(42)  # For reproducibility
        num_to_select = min(num_papers, len(valid_papers))
        selected_indices = np.random.choice(len(valid_papers), num_to_select, replace=False)
        selected_papers_info = [valid_papers[i] for i in selected_indices]
        
        print(f"\nRandomly selected {num_to_select} papers:")
        output_data = []
        for i, (original_index, paper, bias_label) in enumerate(selected_papers_info, 1):
            print(f"  Paper {i}:")
            print(f"    Original index: {original_index}")
            print(f"    Title: {paper.get('Title', 'No title')}")
            print(f"    Bias label: {bias_label}")
            print(f"    Body length: {len(paper.get('Body', ''))} characters")
            print(f"    Reason length: {len(paper.get('Reason', ''))} characters")
            output_data.append(paper)
        
        # Save the papers to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{num_to_select} random papers saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return False


if __name__ == "__main__":
    print("Computer Science Dataset - Random Entries Extractor")
    print("=" * 60)
    
    # Extract random entries from the main dataset
    success = extract_random_entries()
    
    if success:
        print("\n" + "=" * 60)
        print("Random entries extraction completed!")
        print("\nFile created:")
        print("- random_papers.json (1000 random papers for testing)")
    else:
        print("Random entries extraction failed!")
