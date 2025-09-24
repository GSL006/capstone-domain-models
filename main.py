import streamlit as st
import json
import os
import subprocess
import tempfile
import re
from pathlib import Path

def extract_domain_from_subject(subject):
    """Extract domain from subject field using occurrence count priority with first occurrence as tiebreaker"""
    if not subject:
        return None
    
    subject_lower = subject.lower()
    
    # Domain keywords mapping
    domain_keywords = {
        'business': ['business'],
        'comp': ['computer science', 'computer'],
        'econ': ['economics'],
        'evs': ['environmental'],
        'tech': ['technology', 'engineering']
    }
    
    # Count occurrences and track first occurrence position for each domain
    domain_scores = {}
    
    for domain, keywords in domain_keywords.items():
        total_count = 0
        first_position = float('inf')
        
        for keyword in keywords:
            count = subject_lower.count(keyword)
            if count > 0:
                total_count += count
                # Find first occurrence position of this keyword
                pos = subject_lower.find(keyword)
                if pos < first_position:
                    first_position = pos
        
        if total_count > 0:
            domain_scores[domain] = {
                'count': total_count,
                'first_position': first_position
            }
    
    if not domain_scores:
        return None
    
    # Sort by count (descending), then by first position (ascending) for tiebreaker
    sorted_domains = sorted(
        domain_scores.items(),
        key=lambda x: (-x[1]['count'], x[1]['first_position'])
    )
    
    return sorted_domains[0][0]

def run_evaluation(domain, json_file_path):
    """Run the appropriate evaluation script for the domain"""
    domain_scripts = {
        'comp': 'comp/evaluate_upload.py',
        'econ': 'econ/evaluate_upload.py', 
        'tech': 'tech/evaluate_upload.py',
        'business': 'business/evaluate_upload.py',
        'evs': 'evs/evaluate_upload.py'
    }
    
    if domain not in domain_scripts:
        return f"❌ Unsupported domain: {domain}"
    
    script_path = domain_scripts[domain]
    
    # Check if script exists
    if not os.path.exists(script_path):
        return f"❌ Evaluation script not found: {script_path}"
    
    try:
        # Run the evaluation script
        result = subprocess.run(
            ['python', script_path, json_file_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"❌ Error running evaluation:\n{result.stderr}"
            
    except Exception as e:
        return f"❌ Failed to run evaluation: {str(e)}"

def validate_json_structure(data):
    """Validate that the JSON has the required structure"""
    if isinstance(data, list):
        # Multiple papers
        for i, paper in enumerate(data):
            if not isinstance(paper, dict):
                return False, f"Paper {i+1} is not a valid object"
            if 'Subject' not in paper:
                return False, f"Paper {i+1} missing 'Subject' field"
            if 'Body' not in paper and 'text' not in paper:
                return False, f"Paper {i+1} missing 'Body' or 'text' field"
        return True, "Valid"
    elif isinstance(data, dict):
        # Single paper
        if 'Subject' not in data:
            return False, "Missing 'Subject' field"
        if 'Body' not in data and 'text' not in data:
            return False, "Missing 'Body' or 'text' field"
        return True, "Valid"
    else:
        return False, "JSON must be an object or array of objects"

def main():
    st.set_page_config(
        page_title="Research Paper Bias Detection",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🔍 Research Paper Bias Detection System")
    st.markdown("Upload a JSON file containing research papers to detect potential biases")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system analyzes research papers for potential biases:
        - **Cognitive Bias**: Reasoning and decision-making biases
        - **Publication Bias**: Selection bias in published results
        - **No Bias**: No significant bias detected
        
        **Supported Domains:**
        - 💻 Computer Science & Technology
        - 💰 Economics & Business
        - 🌱 Environmental Science
        """)
        
        st.header("📋 JSON Format")
        st.markdown("""
        **Single Paper:**
        ```json
        {
            "Subject": "Technology;Computer Science",
            "Body": "Your paper content...",
            "Reason": "Optional reasoning..."
        }
        ```
        
        **Multiple Papers:**
        ```json
        [
            {
                "Subject": "Economics",
                "Body": "Paper 1 content..."
            },
            {
                "Subject": "Technology", 
                "Body": "Paper 2 content..."
            }
        ]
        ```
        """)
    
    # Main interface
    uploaded_file = st.file_uploader(
        "📁 Choose a JSON file",
        type=['json'],
        help="Upload a JSON file containing one or more research papers"
    )
    
    if uploaded_file is not None:
        try:
            # Read and parse JSON
            file_content = uploaded_file.read().decode('utf-8')
            data = json.loads(file_content)
            
            # Validate JSON structure
            is_valid, validation_message = validate_json_structure(data)
            
            if not is_valid:
                st.error(f"❌ Invalid JSON structure: {validation_message}")
                return
            
            st.success("✅ JSON file loaded successfully!")
            
            # Convert single paper to list for uniform processing
            papers = [data] if isinstance(data, dict) else data
            
            st.info(f"📊 Found {len(papers)} paper(s) to analyze")
            
            # Process each paper
            for i, paper in enumerate(papers):
                st.markdown("---")
                st.subheader(f"📄 Paper {i+1}")
                
                # Extract subject and domain
                subject = paper.get('Subject', 'Unknown')
                domain = extract_domain_from_subject(subject)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Subject:** {subject}")
                    st.markdown(f"**Domain:** {domain if domain else '❓ Unknown'}")
                
                with col2:
                    text_preview = paper.get('Body', paper.get('text', ''))[:200]
                    st.markdown(f"**Content Preview:** {text_preview}...")
                
                if domain:
                    # Create temporary file for this paper
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                        json.dump([paper], temp_file, indent=2)
                        temp_file_path = temp_file.name
                    
                    try:
                        st.markdown("🔄 **Running bias analysis...**")
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        progress_bar.progress(50)
                        
                        # Run evaluation
                        result = run_evaluation(domain, temp_file_path)
                        progress_bar.progress(100)
                        
                        # Display results
                        st.markdown("### 📋 Analysis Results:")
                        
                        # Parse results to make them more readable
                        if "📊" in result or "Accuracy:" in result:
                            st.success("✅ Analysis completed successfully!")
                            st.code(result, language="text")
                        elif "❌" in result:
                            st.error(result)
                        else:
                            st.info(result)
                            
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                else:
                    st.warning(f"⚠️ Could not determine domain for subject: '{subject}'. Supported domains: Computer Science, Technology, Economics, Business, Environmental Science")
                    
                    # Show manual domain selection
                    manual_domain = st.selectbox(
                        f"Select domain manually for Paper {i+1}:",
                        ['', 'comp', 'tech', 'econ', 'business', 'evs'],
                        format_func=lambda x: {
                            '': 'Select domain...',
                            'comp': '💻 Computer Science',
                            'tech': '💻 Technology', 
                            'econ': '💰 Economics',
                            'business': '💰 Business',
                            'evs': '🌱 Environmental Science'
                        }.get(x, x),
                        key=f"domain_select_{i}"
                    )
                    
                    if manual_domain and st.button(f"🔍 Analyze Paper {i+1}", key=f"analyze_{i}"):
                        # Create temporary file for this paper
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                            json.dump([paper], temp_file, indent=2)
                            temp_file_path = temp_file.name
                        
                        try:
                            st.markdown("🔄 **Running bias analysis...**")
                            progress_bar = st.progress(0)
                            progress_bar.progress(50)
                            
                            result = run_evaluation(manual_domain, temp_file_path)
                            progress_bar.progress(100)
                            
                            st.markdown("### 📋 Analysis Results:")
                            if "📊" in result or "Accuracy:" in result:
                                st.success("✅ Analysis completed successfully!")
                                st.code(result, language="text")
                            elif "❌" in result:
                                st.error(result)
                            else:
                                st.info(result)
                                
                        finally:
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit for Research Paper Bias Detection*")

if __name__ == "__main__":
    main()
