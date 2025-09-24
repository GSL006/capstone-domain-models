#!/usr/bin/env python3
"""
Create Test Samples for Streamlit UI
Randomly picks 5 papers (one from each domain) and saves them in a single JSON file
"""

import json
import random
import os
from pathlib import Path

def find_domain_specific_paper(domain_path, domain_name):
    """Find a paper that truly belongs to the specified domain"""
    
    json_file = os.path.join(domain_path, 'random_papers.json')
    
    if not os.path.exists(json_file):
        print(f"⚠️ Warning: {json_file} not found for {domain_name}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        if not papers:
            print(f"⚠️ Warning: No papers found in {json_file}")
            return None
        
        # Define domain-specific keywords that should appear first in subject
        domain_keywords = {
            'comp': ['computer science', 'computer', 'computational'],
            'econ': ['economics', 'economic', 'economy'],
            'tech': ['technology', 'engineering', 'technical'],
            'business': ['business', 'management', 'entrepreneurship'],
            'evs': ['environmental', 'environment', 'ecology', 'sustainability']
        }
        
        # Filter papers that truly belong to this domain using occurrence count priority
        domain_papers = []
        keywords = domain_keywords.get(domain_name, [])
        
        for paper in papers:
            subject = paper.get('Subject', '').lower()
            
            # Count occurrences for each domain in this paper's subject
            domain_scores = {}
            
            for check_domain, check_keywords in domain_keywords.items():
                total_count = 0
                first_position = float('inf')
                
                for keyword in check_keywords:
                    count = subject.count(keyword)
                    if count > 0:
                        total_count += count
                        pos = subject.find(keyword)
                        if pos < first_position:
                            first_position = pos
                
                if total_count > 0:
                    domain_scores[check_domain] = {
                        'count': total_count,
                        'first_position': first_position
                    }
            
            # Determine the primary domain for this paper
            if domain_scores:
                # Sort by count (descending), then by first position (ascending)
                sorted_domains = sorted(
                    domain_scores.items(),
                    key=lambda x: (-x[1]['count'], x[1]['first_position'])
                )
                primary_domain = sorted_domains[0][0]
                
                # If this paper's primary domain matches our target domain, include it
                if primary_domain == domain_name:
                    domain_papers.append(paper)
        
        # If no domain-specific papers found, use any paper but fix the subject
        if not domain_papers:
            print(f"   ⚠️ No domain-specific papers found for {domain_name}, using random paper")
            domain_papers = papers[:10]  # Take first 10 as candidates
        
        # Pick a random paper from filtered set
        selected_paper = random.choice(domain_papers)
        
        # Ensure subject starts with domain-specific content
        domain_subjects = {
            'comp': '(B/T) Computer Science;(B/T) Data Science;(B/T) Technology',
            'econ': '(B/T) Business - Economics;(B/T) Economics;(SOC) Economics', 
            'tech': '(B/T) Technology;(PHY) Engineering - Electrical;(B/T) Computer Science',
            'business': '(B/T) Business - General;(B/T) Business - Management;(B/T) Data Science',
            'evs': '(ENV) Environmental Sciences;(PHY) Geology;(ENV) Sustainability'
        }
        
        # Override subject to ensure proper domain detection
        selected_paper['Subject'] = domain_subjects.get(domain_name, f'{domain_name.title()};Research')
        
        # Ensure we have Body field
        if 'Body' not in selected_paper and 'text' in selected_paper:
            selected_paper['Body'] = selected_paper['text']
        
        # Add source indicator
        selected_paper['_source_domain'] = domain_name
        
        return selected_paper
        
    except Exception as e:
        print(f"❌ Error loading from {json_file}: {e}")
        return None

def create_test_samples():
    """Create test samples with one paper from each domain"""
    
    # Domain configuration
    domains = {
        'comp': 'comp',
        'econ': 'econ', 
        'tech': 'tech',
        'business': 'business',
        'evs': 'evs'
    }
    
    test_papers = []
    
    print("🔍 Collecting random papers from each domain...")
    
    for domain_key, domain_path in domains.items():
        print(f"📄 Finding domain-specific paper from {domain_key}...")
        
        paper = find_domain_specific_paper(domain_path, domain_key)
        
        if paper:
            # Clean up the paper for testing
            test_paper = {
                'Subject': paper.get('Subject', f'{domain_key.title()};Research'),
                'Body': paper.get('Body', paper.get('text', '')),
                'Reason': paper.get('Reason', ''),
                '_source_domain': domain_key,
                '_paper_id': paper.get('id', f'{domain_key}_sample')
            }
            
            # Add Overall Bias if available (for reference)
            true_label = None
            if 'Overall Bias' in paper:
                true_label = paper['Overall Bias']
            elif 'OverallBias' in paper:
                true_label = paper['OverallBias']
            
            if true_label:
                test_paper['_true_label'] = true_label
                # Also include it as a visible field for easy checking
                test_paper['Actual_Bias'] = true_label
            
            test_papers.append(test_paper)
            subject_preview = test_paper['Subject'][:60] + "..." if len(test_paper['Subject']) > 60 else test_paper['Subject']
            print(f"   ✅ Added {domain_key} paper")
            print(f"      Subject: {subject_preview}")
            print(f"      True Label: {true_label or 'Unknown'}")
        else:
            print(f"   ❌ Failed to load paper from {domain_key}")
    
    return test_papers

def save_test_samples(test_papers, output_file='test_samples.json'):
    """Save test papers to JSON file"""
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_papers, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(test_papers)} test papers to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving to {output_file}: {e}")
        return False

def create_single_paper_samples(test_papers):
    """Create individual JSON files for each paper (for testing single paper upload)"""
    
    for i, paper in enumerate(test_papers):
        domain = paper['_source_domain']
        filename = f'test_sample_{domain}.json'
        
        # Keep actual bias for reference, remove only internal _fields
        clean_paper = {k: v for k, v in paper.items() if not k.startswith('_')}
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(clean_paper, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Created single paper test: {filename}")
            
        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")

def display_summary(test_papers):
    """Display summary of created test samples"""
    
    print(f"\n📊 Test Samples Summary:")
    print(f"=" * 50)
    print(f"Total papers: {len(test_papers)}")
    
    for i, paper in enumerate(test_papers, 1):
        domain = paper['_source_domain']
        subject = paper['Subject']
        body_length = len(paper['Body'])
        true_label = paper.get('_true_label', 'Unknown')
        
        print(f"\n📝 Paper {i} ({domain.upper()}):")
        print(f"   Subject: {subject}")
        print(f"   Body Length: {body_length:,} characters")
        print(f"   True Label: {true_label}")
        print(f"   Preview: {paper['Body'][:100]}...")

def main():
    """Main function to create test samples"""
    
    print("🎯 Creating Test Samples for Streamlit UI")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check if we're in the right directory
    if not os.path.exists('comp') or not os.path.exists('main.py'):
        print("❌ Error: Please run this script from the capstone-domain-models directory")
        return
    
    # Create test samples
    test_papers = create_test_samples()
    
    if not test_papers:
        print("❌ No test papers could be created!")
        return
    
    # Save multi-paper test file
    if save_test_samples(test_papers, 'test_samples_multi.json'):
        print("✅ Multi-paper test file created: test_samples_multi.json")
    
    # Create single paper test files
    print("\n🔄 Creating individual paper test files...")
    create_single_paper_samples(test_papers)
    
    # Display summary
    display_summary(test_papers)
    
    print(f"\n🎉 Test samples created successfully!")
    print(f"📁 Files created:")
    print(f"   • test_samples_multi.json (all 5 papers)")
    print(f"   • test_sample_comp.json (computer science)")
    print(f"   • test_sample_econ.json (economics)")
    print(f"   • test_sample_tech.json (technology)")
    print(f"   • test_sample_business.json (business)")
    print(f"   • test_sample_evs.json (environmental science)")
    
    print(f"\n🚀 You can now test the Streamlit UI with these files!")
    print(f"   Run: streamlit run main.py")

if __name__ == "__main__":
    main()
