#!/usr/bin/env python3
"""
Script to run evaluate.py for all domains and save results to a text file.
Captures all output including accuracy metrics and error logs.
"""

import subprocess
import os
import sys
import datetime
from pathlib import Path

# All domains configuration
ALL_DOMAINS = {
    'business': 'business',
    'comp': 'comp',
    'econ': 'econ',
    'evs': 'evs',
    'tech': 'tech',
    'health_science': 'health_science',
    'humanities': 'humanities',
    'physical_science': 'physical_science',
    'social_science': 'social_science'
}

def run_evaluation(domain_name, domain_path, base_path='.'):
    """Run evaluate.py for a specific domain and capture output"""
    
    domain_dir = os.path.join(base_path, domain_path)
    evaluate_script = os.path.join(domain_dir, 'evaluate.py')
    
    result = {
        'domain': domain_name,
        'path': domain_path,
        'success': False,
        'stdout': '',
        'stderr': '',
        'return_code': None,
        'error': None
    }
    
    # Check if evaluate.py exists
    if not os.path.exists(evaluate_script):
        result['error'] = f"evaluate.py not found in {domain_dir}"
        return result
    
    print(f"\n{'='*80}")
    print(f"Evaluating {domain_name.upper()}")
    print(f"{'='*80}")
    print(f"Directory: {domain_dir}")
    print(f"Script: {evaluate_script}")
    
    try:
        # Run the evaluation script
        process = subprocess.Popen(
            [sys.executable, evaluate_script],
            cwd=domain_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        stdout, stderr = process.communicate(timeout=600)  # 10 minute timeout
        
        result['return_code'] = process.returncode
        result['stdout'] = stdout
        result['stderr'] = stderr
        
        if process.returncode == 0:
            result['success'] = True
            print(f"✅ {domain_name}: SUCCESS")
        else:
            result['error'] = f"Script exited with code {process.returncode}"
            print(f"❌ {domain_name}: FAILED (exit code {process.returncode})")
        
    except subprocess.TimeoutExpired:
        process.kill()
        result['error'] = "Evaluation timed out after 10 minutes"
        result['stdout'] = "TIMEOUT - Process was killed"
        print(f"⏱️ {domain_name}: TIMEOUT")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ {domain_name}: EXCEPTION - {str(e)}")
    
    return result

def extract_metrics(stdout):
    """Extract key metrics from stdout"""
    import re
    metrics = {}
    
    # Look for accuracy - try multiple patterns
    accuracy_patterns = [
        r'Accuracy[:\s]+(\d+\.\d{4})',  # "Accuracy: 0.6040"
        r'accuracy[:\s]+(\d+\.\d{4})',  # "accuracy: 0.6040"
        r'Overall Accuracy[:\s]+(\d+\.\d{4})',  # "Overall Accuracy: 0.6040"
        r'Accuracy[:\s]+(\d+\.\d{2})%',  # "Accuracy: 60.40%"
        r'(\d+\.\d{4})\s*\([^)]*%\)',  # "0.6040 (60.40%)"
    ]
    
    for pattern in accuracy_patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            try:
                acc_value = float(match.group(1))
                # If it's a percentage (0-100), convert to decimal
                if acc_value > 1.0:
                    acc_value = acc_value / 100.0
                metrics['accuracy_value'] = acc_value
                metrics['accuracy_line'] = match.group(0)
                break
            except:
                continue
    
    # Also capture the full accuracy line for reference
    if 'accuracy_value' not in metrics:
        lines = stdout.split('\n')
        for line in lines:
            if 'Accuracy' in line or 'accuracy' in line:
                metrics['accuracy_line'] = line.strip()
                break
    
    # Look for evaluation results section
    if 'Evaluation Results' in stdout:
        metrics['has_evaluation_results'] = True
    
    # Count papers evaluated - be more specific to avoid matching accuracy
    papers_match = re.search(r'(\d+)\s+papers?\s+evaluated', stdout, re.IGNORECASE)
    if papers_match:
        metrics['papers_evaluated'] = int(papers_match.group(1))
    
    # Look for F1, Precision, Recall if available
    f1_match = re.search(r'F1[^:]*:\s*(\d+\.\d{4})', stdout, re.IGNORECASE)
    if f1_match:
        metrics['f1_score'] = float(f1_match.group(1))
    
    precision_match = re.search(r'Precision[^:]*:\s*(\d+\.\d{4})', stdout, re.IGNORECASE)
    if precision_match:
        metrics['precision'] = float(precision_match.group(1))
    
    recall_match = re.search(r'Recall[^:]*:\s*(\d+\.\d{4})', stdout, re.IGNORECASE)
    if recall_match:
        metrics['recall'] = float(recall_match.group(1))
    
    return metrics

def save_results_to_file(results, output_file='evaluation_results_all.txt'):
    """Save all evaluation results to a text file"""
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total Domains: {len(results)}\n")
        
        # Count successes and failures
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary section
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        for result in results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            f.write(f"{result['domain'].upper():<20} {status}\n")
            if result['error']:
                f.write(f"  Error: {result['error']}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed results for each domain
        for i, result in enumerate(results, 1):
            domain = result['domain'].upper()
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            
            f.write(f"\n{'='*80}\n")
            f.write(f"DOMAIN {i}/{len(results)}: {domain} - {status}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Path: {result['path']}\n")
            f.write(f"Return Code: {result['return_code']}\n")
            
            if result['error']:
                f.write(f"Error: {result['error']}\n")
            
            # Extract and display metrics
            metrics = extract_metrics(result['stdout'])
            if metrics:
                f.write("\nKey Metrics:\n")
                f.write("-" * 80 + "\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
            
            # Standard output
            f.write("\n" + "-" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write("-" * 80 + "\n")
            if result['stdout']:
                f.write(result['stdout'])
            else:
                f.write("(No output)\n")
            
            # Standard error (if any)
            if result['stderr']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("STDERR:\n")
                f.write("-" * 80 + "\n")
                f.write(result['stderr'])
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Failed domains summary at the end
        if failed:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("FAILED DOMAINS - DEBUGGING INFORMATION\n")
            f.write("=" * 80 + "\n")
            
            for result in failed:
                f.write(f"\n{result['domain'].upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Path: {result['path']}\n")
                f.write(f"Error: {result['error'] or 'Unknown error'}\n")
                f.write(f"Return Code: {result['return_code']}\n")
                
                if result['stderr']:
                    f.write("\nError Output:\n")
                    f.write(result['stderr'])
                
                if result['stdout']:
                    f.write("\nStandard Output:\n")
                    f.write(result['stdout'])
                
                f.write("\n")
        
        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

def main():
    """Main function to run all evaluations"""
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("RUNNING ALL DOMAIN EVALUATIONS")
    print("=" * 80)
    print(f"Base directory: {base_path}")
    print(f"Total domains: {len(ALL_DOMAINS)}")
    print(f"Domains: {', '.join(ALL_DOMAINS.keys())}")
    
    results = []
    
    # Run evaluation for each domain
    for domain_name, domain_path in ALL_DOMAINS.items():
        result = run_evaluation(domain_name, domain_path, base_path)
        results.append(result)
        
        # Print quick summary
        if result['success']:
            metrics = extract_metrics(result['stdout'])
            if 'accuracy_value' in metrics:
                print(f"   Accuracy: {metrics['accuracy_value']:.4f}")
            if 'papers_evaluated' in metrics:
                print(f"   Papers evaluated: {metrics['papers_evaluated']}")
        else:
            print(f"   Error: {result['error']}")
    
    # Save results to file
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    save_results_to_file(results)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   - {r['domain']}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   - {r['domain']}: {r['error']}")
    
    print(f"\n📄 Detailed results saved to: evaluation_results_all.txt")

if __name__ == "__main__":
    main()

