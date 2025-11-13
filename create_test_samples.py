#!/usr/bin/env python3
"""
Create Test Samples for Streamlit UI
Creates test_sample_{domain}.json files with 5 papers each
Creates test_samples_multi.json with one paper from each domain
"""

import json
import random
import os
from pathlib import Path

# All domains configuration
ALL_DOMAINS = {
    'comp': 'comp',
    'econ': 'econ', 
    'tech': 'tech',
    'business': 'business',
    'evs': 'evs',
    'health_science': 'health_science',
    'humanities': 'humanities',
    'physical_science': 'physical_science',
    'social_science': 'social_science'
}

def find_papers_for_domain(domain_path, domain_name, num_papers=5):
    """Find papers for a specific domain"""
    
    # Check for random.json first, then random_papers.json
    json_file = os.path.join(domain_path, 'random.json')
    if not os.path.exists(json_file):
        json_file = os.path.join(domain_path, 'random_papers.json')
    
    if not os.path.exists(json_file):
        print(f"⚠️ Warning: No data file found for {domain_name}")
        return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        if not papers:
            print(f"⚠️ Warning: No papers found in {json_file}")
            return []
        
        # If we need fewer papers than available, sample randomly
        if len(papers) > num_papers:
            selected_papers = random.sample(papers, num_papers)
        else:
            selected_papers = papers
        
        # Clean and format papers
        formatted_papers = []
        for paper in selected_papers:
            if not isinstance(paper, dict):
                continue
            
            formatted_paper = {
                'Subject': paper.get('Subject', f'{domain_name.title()};Research'),
                'Body': paper.get('Body', paper.get('text', '')),
                'Reason': paper.get('Reason', '')
            }
            
            # Add Overall Bias if available (for reference)
            true_label = paper.get('Overall Bias') or paper.get('OverallBias')
            if true_label:
                formatted_paper['Actual_Bias'] = true_label
            
            formatted_papers.append(formatted_paper)
        
        return formatted_papers
        
    except Exception as e:
        print(f"❌ Error loading from {json_file}: {e}")
        return []

def create_domain_test_file(domain_key, domain_path, num_papers=1):
    """Create test_sample_{domain}.json with 1 paper"""
    
    print(f"📄 Creating test_sample_{domain_key}.json...")
    
    papers = find_papers_for_domain(domain_path, domain_key, num_papers)
    
    if not papers:
        print(f"   ❌ Failed to create test_sample_{domain_key}.json")
        return False
    
    filename = f'test_sample_{domain_key}.json'
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Created {filename} with {len(papers)} papers")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating {filename}: {e}")
        return False

def create_multi_domain_test_file():
    """Create test_samples_multi.json with one paper from each domain"""
    
    print(f"\n📄 Creating test_samples_multi.json...")
    
    multi_papers = []
    
    for domain_key, domain_path in ALL_DOMAINS.items():
        papers = find_papers_for_domain(domain_path, domain_key, num_papers=1)
        
        if papers:
            paper = papers[0]
            paper['_source_domain'] = domain_key
            multi_papers.append(paper)
            print(f"   ✅ Added paper from {domain_key}")
        else:
            print(f"   ⚠️ Skipped {domain_key} (no papers found)")
    
    if not multi_papers:
        print("   ❌ No papers found for multi-domain file")
        return False
    
    try:
        # Remove internal _source_domain field before saving
        clean_papers = []
        for paper in multi_papers:
            clean_paper = {k: v for k, v in paper.items() if not k.startswith('_')}
            clean_papers.append(clean_paper)
        
        with open('test_samples_multi.json', 'w', encoding='utf-8') as f:
            json.dump(clean_papers, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Created test_samples_multi.json with {len(clean_papers)} papers")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating test_samples_multi.json: {e}")
        return False

def display_summary():
    """Display summary of created test samples"""
    
    print(f"\n📊 Test Samples Summary:")
    print(f"=" * 60)
    
    total_files = 0
    total_papers = 0
    
    for domain_key in ALL_DOMAINS.keys():
        filename = f'test_sample_{domain_key}.json'
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                    num_papers = len(papers) if isinstance(papers, list) else 1
                    total_papers += num_papers
                    total_files += 1
                    print(f"✅ {filename}: {num_papers} papers")
            except:
                print(f"⚠️ {filename}: Error reading file")
        else:
            print(f"❌ {filename}: Not created")
    
    if os.path.exists('test_samples_multi.json'):
        try:
            with open('test_samples_multi.json', 'r', encoding='utf-8') as f:
                papers = json.load(f)
                num_papers = len(papers) if isinstance(papers, list) else 1
                print(f"\n✅ test_samples_multi.json: {num_papers} papers")
        except:
            print(f"\n⚠️ test_samples_multi.json: Error reading file")
    
    print(f"\n📈 Total: {total_files} domain files, {total_papers} total papers")

def main():
    """Main function to create test samples"""
    
    print("🎯 Creating Test Samples for Streamlit UI")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check if we're in the right directory
    if not os.path.exists('comp') or not os.path.exists('main.py'):
        print("❌ Error: Please run this script from the capstone-domain-models directory")
        return
    
    # Create test_sample_{domain}.json files with 1 paper each
    print("\n📝 Creating individual domain test files (1 paper each)...")
    print("-" * 60)
    
    for domain_key, domain_path in ALL_DOMAINS.items():
        create_domain_test_file(domain_key, domain_path, num_papers=1)
    
    # Create test_samples_multi.json with one paper from each domain
    create_multi_domain_test_file()
    
    # Display summary
    display_summary()
    
    print(f"\n🎉 Test samples created successfully!")
    print(f"\n📁 Files created:")
    print(f"   • test_samples_multi.json (one paper from each domain)")
    for domain_key in ALL_DOMAINS.keys():
        print(f"   • test_sample_{domain_key}.json (1 paper)")
    
    print(f"\n🚀 You can now test the Streamlit UI with these files!")
    print(f"   Run: streamlit run main.py")

if __name__ == "__main__":
    main()
