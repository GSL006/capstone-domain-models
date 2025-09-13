import json
import numpy as np
from collections import Counter


def split_cs_dataset(input_file='computer_science_papers.json', 
                    train_size=4000, 
                    test_size=1258,
                    train_output='cs_train.json',
                    test_output='cs_test.json'):
    """
    Split computer science dataset into training and testing sets
    
    Args:
        input_file: Input JSON file with all papers
        train_size: Number of papers for training (4000)
        test_size: Number of papers for testing (1258) 
        train_output: Output file for training set
        test_output: Output file for testing set
    """
    
    print(f"Loading {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total papers loaded: {len(data)}")
        
        # Analyze class distribution
        labels = []
        for paper in data:
            if 'OverallBias' in paper:
                if paper['OverallBias'] == 'No Bias':
                    labels.append(0)
                elif paper['OverallBias'] == 'Cognitive Bias':
                    labels.append(1)
                elif paper['OverallBias'] == 'Publication Bias':
                    labels.append(2)
                else:
                    labels.append(2)  # Default to publication bias
            else:
                # Fallback logic
                labels.append(2)
        
        label_counts = Counter(labels)
        print(f"Class distribution: {label_counts}")
        
        # Create stratified split to maintain class proportions
        class_indices = {0: [], 1: [], 2: []}
        for i, label in enumerate(labels):
            class_indices[label].append(i)
        
        print(f"Papers per class:")
        for class_id, indices in class_indices.items():
            class_names = {0: 'No Bias', 1: 'Cognitive Bias', 2: 'Publication Bias'}
            print(f"  {class_names[class_id]}: {len(indices)} papers")
        
        # Calculate split proportions for each class
        train_indices = []
        test_indices = []
        
        total_papers = len(data)
        train_ratio = train_size / total_papers
        
        for class_id, indices in class_indices.items():
            np.random.shuffle(indices)  # Randomize order
            
            n_train_class = int(len(indices) * train_ratio)
            n_train_class = min(n_train_class, len(indices) - 1)  # Ensure at least 1 for test
            
            train_indices.extend(indices[:n_train_class])
            test_indices.extend(indices[n_train_class:])
        
        # Shuffle the final indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Limit to desired sizes
        train_indices = train_indices[:train_size]
        test_indices = test_indices[:test_size]
        
        print(f"\nActual split:")
        print(f"Training set: {len(train_indices)} papers")
        print(f"Test set: {len(test_indices)} papers")
        
        # Create training set
        train_data = [data[i] for i in train_indices]
        with open(train_output, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        
        # Create test set  
        test_data = [data[i] for i in test_indices]
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Verify class distribution in splits
        train_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in test_indices]
        
        print(f"\nTraining set class distribution: {Counter(train_labels)}")
        print(f"Test set class distribution: {Counter(test_labels)}")
        
        print(f"\nDataset split successfully!")
        print(f"Training data saved to: {train_output}")
        print(f"Test data saved to: {test_output}")
        
        return True
        
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return False


if __name__ == "__main__":
    print("Computer Science Dataset Splitter")
    print("=" * 40)
    
    # Split the main dataset
    success = split_cs_dataset()
    
    if success:
        print("\n" + "=" * 40)
        print("Dataset splitting completed!")
        print("\nFiles created:")
        print("- cs_train.json (4000 papers for training)")
        print("- cs_test.json (1258 papers for testing)")
    else:
        print("Dataset splitting failed!")
