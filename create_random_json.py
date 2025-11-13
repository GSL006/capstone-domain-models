"""
Script to create random.json files for each model folder with 1000 random papers.

Usage:
    python create_random_json.py

This script:
- Creates random.json files for each model folder with 1000 random papers
- If random.json already exists, it will be skipped
- Looks for main dataset files (*_papers.json) in each folder
- If main dataset doesn't exist, uses random_papers.json if available
- If neither exists, the folder will be skipped

The random.json files will be used by evaluate_all_models.py for evaluation.
"""

import json
import random
import os
from pathlib import Path

# Model folder configurations
MODEL_CONFIGS = {
    'business': {
        'dataset_file': 'business_papers.json',
        'random_file': 'random.json'
    },
    'comp': {
        'dataset_file': 'computer_science_papers.json',
        'random_file': 'random.json'
    },
    'econ': {
        'dataset_file': 'economics_papers.json',
        'random_file': 'random.json'
    },
    'evs': {
        'dataset_file': 'environmental_science_papers.json',
        'random_file': 'random.json'
    },
    'tech': {
        'dataset_file': 'technology_papers.json',
        'random_file': 'random.json'
    },
    'health_science': {
        'dataset_file': 'health_sciences_papers.json',
        'random_file': 'random.json'
    },
    'humanities': {
        'dataset_file': 'humanities_papers.json',
        'random_file': 'random.json'
    },
    'physical_science': {
        'dataset_file': 'physical_sciences_papers.json',
        'random_file': 'random.json'
    },
    'social_science': {
        'dataset_file': 'social_science_papers.json',
        'random_file': 'random.json'
    }
}

def find_dataset_file(folder_path, possible_names):
    """Find the dataset file in the folder"""
    for name in possible_names:
        file_path = os.path.join(folder_path, name)
        if os.path.exists(file_path):
            return file_path
    return None

def extract_papers_from_json(json_file_path, num_papers=1000, use_first=False):
    """
    Extract papers from JSON file.
    If use_first=True, returns first num_papers papers.
    Otherwise, returns num_papers random papers.
    """
    if not os.path.exists(json_file_path):
        return None
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: {json_file_path} is not a list, skipping...")
            return None
        
        # Filter valid papers (must have Body and Overall Bias)
        valid_papers = []
        for paper in data:
            if not isinstance(paper, dict):
                continue
            
            # Check for Body field
            body = paper.get('Body', paper.get('text', paper.get('content', '')))
            if not body or not body.strip():
                continue
            
            # Check for Overall Bias field
            bias = paper.get('Overall Bias') or paper.get('OverallBias')
            if not bias or (isinstance(bias, str) and not bias.strip()):
                continue
            
            valid_papers.append(paper)
        
        if len(valid_papers) == 0:
            print(f"Warning: No valid papers found in {json_file_path}")
            return None
        
        # Select papers
        if use_first:
            selected = valid_papers[:min(num_papers, len(valid_papers))]
        else:
            random.seed(42)  # For reproducibility
            selected = random.sample(valid_papers, min(num_papers, len(valid_papers)))
        
        print(f"  Selected {len(selected)} papers from {len(valid_papers)} valid papers")
        return selected
        
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None

def create_random_json_for_folder(folder_name, config, base_path='.'):
    """Create random.json for a specific model folder"""
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_name} does not exist, skipping...")
        return False
    
    random_file_path = os.path.join(folder_path, config['random_file'])
    
    # Check if random.json already exists
    if os.path.exists(random_file_path):
        print(f"[OK] {folder_name}: {config['random_file']} already exists, skipping...")
        return True
    
    print(f"\nProcessing {folder_name}...")
    
    # Try to find dataset file
    dataset_file = None
    if config['dataset_file']:
        dataset_file = os.path.join(folder_path, config['dataset_file'])
        if not os.path.exists(dataset_file):
            dataset_file = None
    
    # If dataset file not found, try common names
    if not dataset_file:
        possible_names = [
            config['dataset_file'] if config['dataset_file'] else None,
            f"{folder_name}_papers.json",
            "papers.json",
            "dataset.json"
        ]
        possible_names = [n for n in possible_names if n]
        dataset_file = find_dataset_file(folder_path, possible_names)
    
    # Try to extract papers
    papers = None
    if dataset_file:
        print(f"  Using dataset file: {os.path.basename(dataset_file)}")
        papers = extract_papers_from_json(dataset_file, num_papers=1000, use_first=False)
    
    # If no papers from dataset, try random_papers.json
    if not papers:
        random_papers_file = os.path.join(folder_path, 'random_papers.json')
        if os.path.exists(random_papers_file):
            print(f"  Using existing random_papers.json")
            papers = extract_papers_from_json(random_papers_file, num_papers=1000, use_first=True)
    
    # If still no papers, give up
    if not papers:
        print(f"  [X] Could not find dataset file for {folder_name}")
        return False
    
    # Save to random.json
    try:
        with open(random_file_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Created {config['random_file']} with {len(papers)} papers")
        return True
    except Exception as e:
        print(f"  [X] Error saving {config['random_file']}: {e}")
        return False

def main():
    """Main function to create random.json files for all model folders"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("Creating random.json files for all model folders...")
    print("=" * 60)
    
    results = {}
    for folder_name, config in MODEL_CONFIGS.items():
        results[folder_name] = create_random_json_for_folder(folder_name, config, base_path)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for folder_name, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"{status} {folder_name}")
    
    successful = sum(1 for s in results.values() if s)
    print(f"\nSuccessfully processed {successful}/{len(results)} folders")

if __name__ == "__main__":
    main()

