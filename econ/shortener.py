import json
import sys
import os

# Add parent directory to path to import shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shortener import get_first_n_objects_and_save

# Example usage for economics papers
if __name__ == "__main__":
    input_file = 'economics_papers.json'  # Economics papers file
    output_file = 'first_1000.json'      # Output file for first 1000 papers

    if get_first_n_objects_and_save(input_file, output_file, 1000):
        print(f"Economics data saved to {output_file}")
    else:
        print("Failed to extract and save economics data.")
