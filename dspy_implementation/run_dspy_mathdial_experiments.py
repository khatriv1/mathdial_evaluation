# dspy_implementation/run_dspy_mathdial_experiments.py

import sys
import os
import shutil
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CLEAR DSPY CACHE FUNCTION
def clear_dspy_cache():
    """Clear ALL DSPy cache locations"""
    import glob
    
    cache_dirs = [
        os.path.expanduser("~/.cache/dspy"),
        os.path.expanduser("~/.dspy_cache"),
        os.path.expanduser("~/.dspy"),
        os.path.expanduser("~/dspy_cache"),
        ".dspy_cache",
        "dspy_cache",
        "/tmp/dspy*",
        "/var/tmp/dspy*",
        os.path.expanduser("~/.cache/litellm"),
        os.path.expanduser("~/.litellm"),
    ]
    
    for pattern in ["*.dspy", ".dspy*", "dspy_cache*"]:
        cache_dirs.extend(glob.glob(pattern))
    
    cleared_count = 0
    for cache_dir in cache_dirs:
        if '*' in cache_dir:
            for path in glob.glob(cache_dir):
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                        print(f"✓ Cleared: {path}")
                        cleared_count += 1
                    except Exception as e:
                        print(f"Could not clear {path}: {e}")
        else:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✓ Cleared: {cache_dir}")
                    cleared_count += 1
                except Exception as e:
                    print(f"Could not clear {cache_dir}: {e}")
    
    if cleared_count == 0:
        print("No cache found - starting fresh")
    else:
        print(f"Cleared {cleared_count} cache location(s)")
    
    os.environ['DSPY_CACHEDIR'] = '/tmp/dspy_no_cache_' + str(time.time())
    os.environ['LITELLM_CACHE'] = 'FALSE'
    
    print("="*60)

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dspy_mathdial_classifier import (
    load_training_data,
    train_dspy_module,
    test_dspy_module,
    test_api_connection
)

def create_comparison_charts(summary_results, output_dir='results'):
    """Create comparison charts for 100/200/300 samples - MathDial version"""
    
    # Prepare data
    techniques = list(summary_results.keys())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    
    # Top chart - Metrics Comparison for MathDial
    metrics_names = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)', 'Cohen\'s κ']
    x = np.arange(len(techniques))
    width = 0.15
    
    colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics_names):
        if metric == 'Accuracy (%)':
            values = [summary_results[t]['accuracy'] * 100 for t in techniques]
        elif metric == 'Precision (%)':
            values = [summary_results[t]['precision'] * 100 for t in techniques]
        elif metric == 'Recall (%)':
            values = [summary_results[t]['recall'] * 100 for t in techniques]
        elif metric == 'F1 Score (%)':
            values = [summary_results[t]['f1'] * 100 for t in techniques]
        else:  # Cohen's κ
            values = [summary_results[t]['kappa'] * 100 for t in techniques]
        
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_offset = 1
            else:
                va = 'top'
                y_offset = -1
            ax1.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{val:.1f}', ha='center', va=va, fontsize=8)
    
    ax1.set_xlabel('DSPy Training Size (GPT-3.5)', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('MathDial Teacher Moves: DSPy GPT-3.5 Performance on 100/200/300 Training Samples', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace('DSPy_Module_', '') + ' samples' for t in techniques])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylim(-10, 110)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Bottom chart - Accuracy Ranking
    sorted_techniques = sorted(summary_results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True)
    
    names = [t[0].replace('DSPy_Module_', 'DSPy-') + ' samples' for t in sorted_techniques]
    scores = [t[1]['accuracy'] * 100 for t in sorted_techniques]
    
    # Color based on training size
    def get_color(name):
        if '300' in name:
            return '#2ecc71'
        elif '200' in name:
            return '#3498db'
        else:  # 100
            return '#e74c3c'
    
    colors_ranked = [get_color(n) for n in names]
    bars = ax2.barh(range(len(names)), scores, color=colors_ranked, alpha=0.8)
    
    # Add percentage labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax2.text(score + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=11)
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title('DSPy GPT-3.5 Modules Ranked by Teacher Move Classification Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, 'dspy_mathdial_gpt35_comparison_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to {chart_path}")
    plt.close()
    
    # Create accuracy trend chart
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract sizes from summary_results keys
    sizes = []
    for key in summary_results.keys():
        size = int(key.replace('DSPy_Module_', ''))
        if size not in sizes:
            sizes.append(size)
    sizes.sort()
    
    accuracy_scores = [summary_results[f'DSPy_Module_{s}']['accuracy'] * 100 for s in sizes]
    f1_scores = [summary_results[f'DSPy_Module_{s}']['f1'] * 100 for s in sizes]
    kappa_scores = [summary_results[f'DSPy_Module_{s}']['kappa'] * 100 for s in sizes]
    
    ax3.plot(sizes, accuracy_scores, 'o-', label='Accuracy (%)', linewidth=2, markersize=8, color='#3498db')
    ax3.plot(sizes, f1_scores, 's-', label='F1 Score (%)', linewidth=2, markersize=8, color='#2ecc71')
    ax3.plot(sizes, kappa_scores, '^-', label="Cohen's κ (×100)", linewidth=2, markersize=8, color='#e74c3c')
    
    # Add value labels
    for i, size in enumerate(sizes):
        ax3.text(size, accuracy_scores[i] + 1, f'{accuracy_scores[i]:.1f}', ha='center', fontsize=9)
        ax3.text(size, f1_scores[i] - 2, f'{f1_scores[i]:.1f}', ha='center', fontsize=9)
    
    ax3.set_xlabel('Training Sample Size', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('MathDial DSPy GPT-3.5 Performance vs Training Size', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(sizes)
    ax3.set_xticklabels(sizes)
    ax3.set_ylim(0, 100)
    
    trend_path = os.path.join(output_dir, 'dspy_mathdial_gpt35_accuracy_trend.png')
    plt.savefig(trend_path, dpi=300, bbox_inches='tight')
    print(f"Trend chart saved to {trend_path}")
    plt.close()

def run_all_experiments(clear_cache=False):
    """Run DSPy experiments with 100, 200, 300 samples using GPT-3.5"""
    
    # CLEAR CACHE IF REQUESTED
    if clear_cache:
        print("="*60)
        print("CLEARING DSPY CACHE")
        print("="*60)
        clear_dspy_cache()
    
    print("="*60)
    print("DSPY EXPERIMENTS FOR MATHDIAL TEACHER MOVES")
    print("Model: GPT-3.5-turbo")
    print("Training sizes: 100, 200, 300 samples")
    print("Test set: 150 holdout samples")
    print("="*60)
    
    # Test API connection
    test_api_connection()
    
    print("\nDSPy will learn patterns from the 300-sample training set")
    print("Using GPT-3.5-turbo for all experiments")
    if not clear_cache:
        print("Using cached responses for speed (add --clear-cache for fresh calls)")
    print("="*60)
    
    # UPDATED PATHS - USING DSPY_300.CSV
    training_path = '../data/dspy_300.csv'  # Your 557-row file with utterances
    test_path = '../data/hold_out150.csv'   # Your holdout test set
    
    # Check if files exist
    if not os.path.exists(training_path):
        print(f"ERROR: {training_path} not found!")
        print("Please ensure dspy_300.csv is in the data folder")
        return
    
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found!")
        print("Please ensure hold_out150.csv is in the data folder")
        return
    
    # Load all 557 training examples from dspy_300.csv
    print(f"\nLoading training data from {training_path}...")
    all_training_examples = load_training_data(training_path, sample_size=None)
    print(f"✓ Loaded {len(all_training_examples)} training utterance examples")
    
    # STANDARD SIZES: 100, 200, 300
    sample_sizes = [100, 200, 300]
    
    all_results = []
    summary_results = {}
    
    for size in sample_sizes:
        print("\n" + "="*60)
        print(f"EXPERIMENT: DSPy Module with {size} training samples")
        print(f"Model: GPT-3.5-turbo")
        print("="*60)
        
        # Train module with specified sample size
        module = train_dspy_module(all_training_examples, size)
        
        # Test module on 150 holdout samples
        module_name = f'DSPy_Module_{size}'
        results_df = test_dspy_module(module, test_path, module_name)
        
        # Save individual results
        os.makedirs('results', exist_ok=True)
        results_file = f'results/dspy_mathdial_results_{size}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        # Store summary
        summary_results[module_name] = {
            'accuracy': float(results_df['accuracy'].iloc[0]),
            'precision': float(results_df['precision'].iloc[0]),
            'recall': float(results_df['recall'].iloc[0]),
            'f1': float(results_df['f1'].iloc[0]),
            'kappa': float(results_df['kappa'].iloc[0]),
            'model': 'GPT-3.5-turbo'
        }
        
        all_results.append(results_df)
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv('results/dspy_mathdial_all_results.csv', index=False)
    
    # Save summary JSON
    summary_with_metadata = {
        'dataset': 'MathDial',
        'model': 'GPT-3.5-turbo',
        'training_sizes': sample_sizes,
        'test_size': 150,
        'timestamp': datetime.now().isoformat(),
        'results': summary_results
    }
    
    with open('results/dspy_mathdial_summary.json', 'w') as f:
        json.dump(summary_with_metadata, f, indent=2)
    print("\nSummary saved to results/dspy_mathdial_summary.json")
    
    # Generate comparison charts
    print("\n" + "="*60)
    print("Generating comparison charts...")
    create_comparison_charts(summary_results)
    
    # Print final summary table
    print("\n" + "="*60)
    print("DSPY MATHDIAL GPT-3.5 RESULTS SUMMARY")
    print("="*60)
    print(f"{'Module':<20} {'Accuracy':<12} {'F1':<10} {'Kappa':<10}")
    print("-"*52)
    for module, metrics in summary_results.items():
        acc_pct = metrics['accuracy'] * 100
        f1_pct = metrics['f1'] * 100
        print(f"{module:<20} {acc_pct:>7.1f}%     {f1_pct:>6.1f}%     {metrics['kappa']:>6.3f}")
    
    # Show improvement from 100 to 300
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    improvement = (summary_results['DSPy_Module_300']['accuracy'] - 
                  summary_results['DSPy_Module_100']['accuracy']) * 100
    print(f"Accuracy Improvement (100→300): {improvement:+.1f}%")
    
    print("\n" + "="*60)
    print("DSPY MathDial GPT-3.5 experiments complete!")
    print("\nFiles saved in results/:")
    print("  - dspy_mathdial_results_100.csv")
    print("  - dspy_mathdial_results_200.csv")
    print("  - dspy_mathdial_results_300.csv")
    print("  - dspy_mathdial_all_results.csv")
    print("  - dspy_mathdial_summary.json")
    print("  - dspy_mathdial_gpt35_comparison_chart.png")
    print("  - dspy_mathdial_gpt35_accuracy_trend.png")
    print("  - mathdial_module_100_learned.json")
    print("  - mathdial_module_200_learned.json")
    print("  - mathdial_module_300_learned.json")
    print("="*60)

if __name__ == "__main__":
    # Check for command line argument
    import sys
    clear_cache = '--clear-cache' in sys.argv
    run_all_experiments(clear_cache=clear_cache)