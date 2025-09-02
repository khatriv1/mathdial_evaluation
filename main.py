# mathdial_evaluation/main.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the 5 evaluation functions that exist for MathDial
from evaluation.evaluate_zero_shot import evaluate_zero_shot
from evaluation.evaluate_few_shot import evaluate_few_shot
from evaluation.evaluate_auto_cot import evaluate_auto_cot
from evaluation.evaluate_self_consistency import evaluate_self_consistency
from evaluation.evaluate_active_prompt import evaluate_active_prompt

import config

def create_comparison_visualization(comparison_df, output_dir):
    """Create comparison visualizations for all MathDial techniques using the 4 metrics."""
    plt.style.use('default')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Prepare data
    techniques = comparison_df['Technique'].tolist()
    x = np.arange(len(techniques))
    
    # First subplot: All 4 metrics comparison
    metrics = ['Accuracy (%)', 'Cohen\'s κ (*100)', 'Krippendorff\'s α (*100)', 'ICC (*100)']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    width = 0.18
    
    for i, metric in enumerate(metrics):
        offset = width * i
        if metric == 'Accuracy (%)':
            values = comparison_df['Accuracy']
        elif metric == 'Cohen\'s κ (*100)':
            values = comparison_df['Cohens_Kappa']
        elif metric == 'Krippendorff\'s α (*100)':
            values = comparison_df['Krippendorffs_Alpha']
        else:  # ICC (*100)
            values = comparison_df['ICC']
        
        bars = ax1.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Prompting Technique', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('MathDial Teacher Move Classification: Metrics Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(techniques, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Second subplot: Ranking by Accuracy
    sorted_df = comparison_df.sort_values('Accuracy', ascending=False)
    y_pos = np.arange(len(sorted_df))
    
    bars = ax2.barh(y_pos, sorted_df['Accuracy'], alpha=0.8, 
                    color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(sorted_df))])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_df['Technique'])
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Techniques Ranked by Classification Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # Add value labels
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        accuracy = row['Accuracy']
        ax2.text(accuracy + 1, i, f'{accuracy:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mathdial_all_techniques_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_mathdial_evaluations(data_path, api_key, output_dir="results", limit=None, techniques=None):
    """Run MathDial teacher move classification evaluations for all techniques."""
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"mathdial_evaluation_all_techniques_{timestamp}"
    output_path = os.path.join(output_dir, base_dir)
    
    # Handle existing directories
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_dir}_{counter}")
        counter += 1
    
    os.makedirs(output_path, exist_ok=True)
    print(f"\nMathDial evaluation results will be saved in: {output_path}")
    
    # Store results from each technique
    all_results = {}
    detailed_results = []
    
    # All available techniques for MathDial (only 5)
    if techniques is None:
        techniques = {
            'Zero-shot': evaluate_zero_shot,
            'Few-shot': evaluate_few_shot,
            'Auto-CoT': evaluate_auto_cot,
            'Self-Consistency': evaluate_self_consistency,
            'Active Prompting': evaluate_active_prompt
        }
    
    # Process each technique
    for technique_name, evaluate_func in techniques.items():
        print(f"\n{'='*60}")
        print(f"Running {technique_name} evaluation...")
        print(f"{'='*60}")
        
        # Create technique-specific directory
        technique_dir = os.path.join(output_path, technique_name.lower().replace(' ', '_'))
        os.makedirs(technique_dir, exist_ok=True)
        
        try:
            # Special handling for self-consistency (needs n_samples parameter)
            if technique_name == 'Self-Consistency':
                results, metrics = evaluate_func(data_path, api_key,
                                               output_dir=technique_dir, 
                                               limit=limit, n_samples=5)
            else:
                results, metrics = evaluate_func(data_path, api_key,
                                               output_dir=technique_dir, limit=limit)
            
            all_results[technique_name] = metrics
            
            # Collect detailed results
            results_df = pd.DataFrame(results)
            results_df['Technique'] = technique_name
            detailed_results.append(results_df)
            
            print(f"✓ {technique_name} completed successfully")
            
        except Exception as e:
            print(f"✗ {technique_name} failed: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    if not all_results:
        print("No successful evaluations completed!")
        return None, None
    
    # Create comparison DataFrame
    comparison_data = []
    for technique_name, metrics in all_results.items():
        comparison_data.append({
            'Technique': technique_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Cohens_Kappa': metrics.get('cohens_kappa', 0),
            'Krippendorffs_Alpha': metrics.get('krippendorffs_alpha', 0),
            'ICC': metrics.get('icc', 0),
            'Num_Samples': metrics.get('num_samples', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison results
    comparison_df.to_csv(f"{output_path}/mathdial_all_techniques_comparison.csv", index=False)
    
    # Create comparison visualization
    create_comparison_visualization(comparison_df, output_path)
    
    # Combine detailed results
    if detailed_results:
        all_detailed_results = pd.concat(detailed_results, ignore_index=True)
        all_detailed_results.to_csv(f"{output_path}/mathdial_all_detailed_results.csv", index=False)
    
    # Generate comprehensive summary report
    with open(f"{output_path}/mathdial_comprehensive_report.txt", 'w') as f:
        f.write("=== MathDial Teacher Move Classification: Comprehensive Evaluation Report ===\n\n")
        
        f.write("Dataset: MathDial Tutoring Conversations\n")
        f.write("Task: Classify teacher utterances into 4 categories\n")
        f.write("Categories: generic, focus, probing, telling\n")
        f.write(f"Number of conversations evaluated: {limit if limit else 'All available (287)'}\n")
        f.write(f"Total techniques evaluated: {len(all_results)}\n\n")
        
        # Overall metrics comparison
        f.write("=== Overall Metrics Comparison ===\n")
        f.write("Metrics Used:\n")
        f.write("- Accuracy: Exact match percentage for move sequences\n")
        f.write("- Cohen's Kappa (κ): Agreement beyond chance (multiplied by 100)\n")
        f.write("- Krippendorff's Alpha (α): Reliability measure (multiplied by 100)\n")
        f.write("- ICC: Intraclass Correlation Coefficient (multiplied by 100)\n\n")
        
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best performing technique for each metric
        f.write("=== Best Performing Techniques by Metric ===\n")
        for metric in ['Accuracy', 'Cohens_Kappa', 'Krippendorffs_Alpha', 'ICC']:
            best_technique = comparison_df.loc[comparison_df[metric].idxmax()]
            metric_name = {
                'Accuracy': 'Accuracy',
                'Cohens_Kappa': 'Cohen\'s Kappa (κ*100)',
                'Krippendorffs_Alpha': 'Krippendorff\'s Alpha (α*100)',
                'ICC': 'Intraclass Correlation (ICC*100)'
            }[metric]
            f.write(f"{metric_name}: {best_technique['Technique']} ({best_technique[metric]:.1f})\n")
        
        # Teacher Move Categories explanation
        f.write("\n\n=== Teacher Move Category Definitions ===\n\n")
        category_descriptions = {
            "GENERIC": "Greeting/Farewell and General inquiry",
            "FOCUS": "Seeking strategy, guiding student focus, recalling relevant information", 
            "PROBING": "Asking for explanation, seeking self-correction, perturbing question, seeking world knowledge",
            "TELLING": "Revealing strategy and revealing answer"
        }
        
        for category, description in category_descriptions.items():
            f.write(f"{category}: {description}\n")
    
    print(f"\n{'='*70}")
    print("MATHDIAL EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved in: {output_path}")
    
    return comparison_df, detailed_results

if __name__ == "__main__":
    print("=" * 70)
    print("MathDial Teacher Move Classification: Comprehensive Evaluation Suite")
    print("=" * 70)
    print(f"Using MathDial dataset: {config.DATA_PATH}")
    
    # Ask user for evaluation parameters
    print("\nEvaluation Options:")
    print("This will evaluate different prompting strategies for teacher move classification.")
    
    limit_input = input("\nEnter number of conversations to evaluate (press Enter for all 287): ")
    
    if limit_input.strip():
        try:
            limit = int(limit_input)
            if limit < 10:
                print("WARNING: Very small sample size may not provide reliable results.")
        except ValueError:
            print("Invalid input. Using all available conversations.")
            limit = None
    else:
        limit = None
    
    # Ask which techniques to evaluate
    print("\nAvailable prompting techniques:")
    all_techniques = [
        "1. Zero-shot",
        "2. Few-shot",
        "3. Auto-CoT",
        "4. Self-Consistency",
        "5. Active Prompting",
        "6. All techniques"
    ]
    
    for technique in all_techniques:
        print(technique)
    
    technique_input = input("\nEnter technique numbers (comma-separated) or 6 for all: ")
    
    try:
        if not technique_input.strip() or '6' in technique_input:
            selected_techniques = None
        else:
            selected_indices = [int(idx.strip()) for idx in technique_input.split(",")]
            technique_map = {
                1: ("Zero-shot", evaluate_zero_shot),
                2: ("Few-shot", evaluate_few_shot),
                3: ("Auto-CoT", evaluate_auto_cot),
                4: ("Self-Consistency", evaluate_self_consistency),
                5: ("Active Prompting", evaluate_active_prompt)
            }
            
            selected_techniques = {}
            for idx in selected_indices:
                if idx in technique_map:
                    name, func = technique_map[idx]
                    selected_techniques[name] = func
    
    except ValueError:
        print("Invalid input. Running all techniques.")
        selected_techniques = None
    
    # Run evaluations
    try:
        print("\nStarting comprehensive MathDial evaluation...")
        
        comparison_df, detailed_results = run_mathdial_evaluations(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=limit,
            techniques=selected_techniques
        )
        
        if comparison_df is not None:
            print("\nAll evaluations completed successfully!")
            print("\nTop techniques by Accuracy:")
            top_techniques = comparison_df.nlargest(3, 'Accuracy')
            for rank, (idx, row) in enumerate(top_techniques.iterrows(), 1):
                print(f"{rank}. {row['Technique']}: {row['Accuracy']:.1f}%")
        else:
            print("\nNo evaluations completed successfully.")
            
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())