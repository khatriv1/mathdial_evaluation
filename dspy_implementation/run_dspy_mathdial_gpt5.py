# dspy_implementation/run_dspy_mathdial_gpt5.py
# python3 run_dspy_mathdial_gpt5.py --clear-cache

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

# Import from GPT-5 version
from dspy_mathdial_classifier_gpt5 import (
    load_training_data,
    train_dspy_module,
    test_dspy_module,
    test_api_connection,
    calculate_mathdial_metrics,
    TEACHER_MOVES
)

def create_comparison_charts(summary_results, output_dir='results'):
    """Create comparison charts for MathDial DSPy GPT-5 results"""
    
    # Define sizes at the beginning of this function
    sizes = [100, 200, 300]
    
    # Prepare data
    techniques = list(summary_results.keys())
    
    # Create figure with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left - Overall Accuracy by Training Size
    ax1 = axes[0, 0]
    accuracies = [summary_results[f'DSPy_MathDial_GPT5_{s}']['overall_accuracy'] for s in sizes]
    
    bars = ax1.bar(sizes, accuracies, color='#3498db', alpha=0.8, width=40)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Training Sample Size', fontsize=12)
    ax1.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax1.set_title('MathDial Classification: Overall Accuracy vs Training Size', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Top right - Per-Move Accuracy for Best Model
    ax2 = axes[0, 1]
    best_model = 'DSPy_MathDial_GPT5_300'
    move_accs = summary_results[best_model]['per_move_accuracy']
    
    moves = []
    accs = []
    counts = []
    for move in TEACHER_MOVES:
        if move in move_accs:
            moves.append(move.capitalize())
            accs.append(move_accs[move][0])
            counts.append(move_accs[move][1])
    
    x_pos = np.arange(len(moves))
    bars = ax2.bar(x_pos, accs, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
    
    for i, (bar, acc, count) in enumerate(zip(bars, accs, counts)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(moves)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Per-Move Accuracy - GPT-5 (300 samples)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.3)
    
    # Bottom left - Scaffolding Ratio vs Telling Rate
    ax3 = axes[1, 0]
    
    scaffolding_ratios = [summary_results[f'DSPy_MathDial_GPT5_{s}']['scaffolding_ratio'] for s in sizes]
    telling_rates = [summary_results[f'DSPy_MathDial_GPT5_{s}']['telling_rate'] for s in sizes]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, scaffolding_ratios, width, 
                   label='Scaffolding (Focus + Probing)', color='#2ecc71', alpha=0.8)
    bars2 = ax3.bar(x + width/2, telling_rates, width,
                   label='Telling Rate', color='#e74c3c', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Training Sample Size', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Teaching Strategy Distribution by Training Size', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sizes)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Bottom right - Accuracy Improvement Trend
    ax4 = axes[1, 1]
    
    ax4.plot(sizes, accuracies, 'o-', linewidth=3, markersize=10, 
             color='#9b59b6', label='Overall Accuracy')
    
    # Add trend line
    z = np.polyfit(sizes, accuracies, 1)
    p = np.poly1d(z)
    ax4.plot(sizes, p(sizes), "--", alpha=0.5, color='gray', label='Trend')
    
    for i, (size, acc) in enumerate(zip(sizes, accuracies)):
        ax4.text(size, acc + 1.5, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Training Sample Size', fontsize=12)
    ax4.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax4.set_title('DSPy MathDial GPT-5: Learning Curve', fontsize=14, fontweight='bold')
    ax4.set_xlim(80, 320)
    ax4.set_ylim(min(accuracies) - 5, max(accuracies) + 10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('DSPy MathDial Teacher Move Classification - GPT-5 Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, 'dspy_mathdial_gpt5_comparison_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\nGPT-5 comparison chart saved to {chart_path}")
    plt.close()

def create_confusion_matrix_chart(confusion_df, output_dir='results'):
    """Create confusion matrix heatmap for best model"""
    try:
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Remove 'All' row and column for cleaner matrix
        confusion_clean = confusion_df.drop('All', axis=0).drop('All', axis=1)
        
        # Normalize to percentages
        confusion_norm = confusion_clean.div(confusion_clean.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        sns.heatmap(confusion_norm, annot=True, fmt='.1f', cmap='YlOrRd',
                    square=True, cbar_kws={'label': 'Percentage (%)'},
                    ax=ax, vmin=0, vmax=100)
        
        ax.set_xlabel('Predicted Move', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Move', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - GPT-5 MathDial (300 samples)', fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        ax.set_xticklabels([m.capitalize() for m in confusion_clean.columns], rotation=45)
        ax.set_yticklabels([m.capitalize() for m in confusion_clean.index], rotation=0)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, 'dspy_mathdial_gpt5_confusion_matrix.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {chart_path}")
        plt.close()
    except ImportError:
        print("Warning: seaborn not installed. Skipping confusion matrix.")
    except Exception as e:
        print(f"Warning: Could not create confusion matrix: {e}")

def run_all_experiments(clear_cache=False):
    """Run DSPy experiments for MathDial with 100, 200, 300 samples using GPT-5"""
    
    # CLEAR CACHE IF REQUESTED
    if clear_cache:
        print("="*60)
        print("CLEARING DSPY CACHE FOR GPT-5 RUN")
        print("="*60)
        clear_dspy_cache()
    
    print("="*60)
    print("DSPY EXPERIMENTS FOR MATHDIAL - GPT-5")
    print("="*60)
    print("Task: Teacher move classification in math tutoring")
    print("Categories: Focus, Probing, Telling, Generic")
    print("Model: GPT-5 (Reasoning Model)")
    print("Training sizes: 100, 200, 300 samples")
    print("Test set: 150 holdout samples")
    print("="*60)
    
    # Test API connection
    test_api_connection()
    
    print("\nDSPy will learn teacher move patterns from math tutoring dialogues using GPT-5")
    print("NO mathdial_rubric.py needed - DSPy learns from data!")
    print("Note: GPT-5 responses may be slower due to internal reasoning")
    if not clear_cache:
        print("Using cached responses for speed (add --clear-cache for fresh calls)")
    print("="*60)
    
    # Paths
    training_path = '../data/dspy_300.csv'  # Your 557-row file with utterances
    test_path = '../data/hold_out150.csv'
    # Check if files exist
    if not os.path.exists(training_path):
        print(f"ERROR: {training_path} not found!")
        print("Please ensure you have the MathDial training data with conversations")
        print("\nExpected CSV format:")
        print("- question: The math problem")
        print("- student_incorrect_solution: Student's wrong answer")
        print("- conversation: Full dialogue with labels like:")
        print("  Teacher: (generic)Hi|EOM|Student: ...|EOM|Teacher: (focus)...")
        return
    
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found!")
        print("Please ensure you have the MathDial test data")
        return
    
    # Load all 300 training samples
    print(f"\nLoading training data from {training_path}...")
    all_training_examples = load_training_data(training_path, sample_size=300)
    print(f"✓ Loaded {len(all_training_examples)} training examples for GPT-5")
    print(f"  (Multiple examples per conversation - one per teacher utterance)")
    
    # Sample sizes: 100, 200, 300 - DEFINE SIZES HERE
    sample_sizes = [100, 200, 300]
    
    all_results = []
    summary_results = {}
    
    for size in sample_sizes:
        print("\n" + "="*60)
        print(f"EXPERIMENT: DSPy MathDial Module with {size} training samples")
        print(f"Model: GPT-5 (Reasoning Model)")
        print("="*60)
        
        # Train module
        module = train_dspy_module(all_training_examples, size)
        
        # Test module
        module_name = f'DSPy_MathDial_GPT5_{size}'
        results_df = test_dspy_module(module, test_path, module_name)
        
        if results_df.empty:
            print(f"Warning: No results for {module_name}")
            continue
        
        # Save individual results
        os.makedirs('results', exist_ok=True)
        results_file = f'results/dspy_mathdial_gpt5_results_{size}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        # Append to all results
        all_results.append(results_df)
        
        # Calculate metrics
        metrics = calculate_mathdial_metrics(results_df)
        
        # Store summary
        summary_results[module_name] = metrics
        
        # Print results
        print(f"\n{module_name} Results:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
        print(f"  Scaffolding Ratio: {metrics['scaffolding_ratio']:.1f}%")
        print(f"  Telling Rate: {metrics['telling_rate']:.1f}%")
        print(f"  Total Utterances: {metrics['total_utterances']}")
        print(f"  Unique Conversations: {metrics['unique_conversations']}")
        
        print(f"\n  Per-Move Accuracy:")
        for move in TEACHER_MOVES:
            if move in metrics['per_move_accuracy']:
                acc, count = metrics['per_move_accuracy'][move]
                print(f"    {move.capitalize():10s}: {acc:5.1f}% ({count} samples)")
    
    # Check if we have results to process
    if not all_results:
        print("\nNo results to process. Check your data files and API connection.")
        return
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv('results/dspy_mathdial_gpt5_all_results.csv', index=False)
    
    # Save summary JSON
    summary_with_metadata = {
        'task': 'MathDial Teacher Move Classification',
        'model': 'GPT-5',
        'model_settings': {
            'note': 'Using default settings for GPT-5'
        },
        'num_categories': 4,
        'categories': TEACHER_MOVES,
        'training_sizes': sample_sizes,
        'test_size': 150,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    # Convert confusion matrices to serializable format
    for key in summary_results:
        summary_with_metadata['results'][key] = {}
        for metric, value in summary_results[key].items():
            if metric == 'confusion_matrix':
                summary_with_metadata['results'][key][metric] = str(value)
            else:
                summary_with_metadata['results'][key][metric] = value
    
    with open('results/dspy_mathdial_gpt5_summary.json', 'w') as f:
        json.dump(summary_with_metadata, f, indent=2, default=str)
    print("\nGPT-5 summary saved to results/dspy_mathdial_gpt5_summary.json")
    
    # Generate comparison charts
    print("\n" + "="*60)
    print("Generating GPT-5 comparison charts...")
    if summary_results:
        create_comparison_charts(summary_results)
    else:
        print("No results to chart")
    
    # Create confusion matrix for best model
    if not all_results_df.empty:
        best_model_df = all_results_df[all_results_df['Technique'] == 'DSPy_MathDial_GPT5_300']
        if len(best_model_df) > 0:
            confusion = pd.crosstab(
                best_model_df['true_move'],
                best_model_df['predicted_move'],
                margins=True
            )
            create_confusion_matrix_chart(confusion)
    
    # Print final summary table
    print("\n" + "="*60)
    print("DSPY MATHDIAL GPT-5 RESULTS SUMMARY")
    print("="*60)
    
    if summary_results:
        print(f"{'Module':<30} {'Overall Acc':<15} {'Scaffolding':<15} {'Telling':<12}")
        print("-"*72)
        for module, metrics in summary_results.items():
            name = module.replace('DSPy_MathDial_GPT5_', '') + ' samples'
            print(f"{name:<30} {metrics['overall_accuracy']:>7.1f}%      "
                  f"{metrics['scaffolding_ratio']:>7.1f}%      {metrics['telling_rate']:>7.1f}%")
        
        # Show improvement - only if we have multiple results
        if len(summary_results) > 1:
            print("\n" + "="*60)
            print("GPT-5 IMPROVEMENT ANALYSIS")
            print("="*60)
            
            # Check if we have both 100 and 300 results
            if 'DSPy_MathDial_GPT5_100' in summary_results and 'DSPy_MathDial_GPT5_300' in summary_results:
                improvement = summary_results['DSPy_MathDial_GPT5_300']['overall_accuracy'] - summary_results['DSPy_MathDial_GPT5_100']['overall_accuracy']
                print(f"Overall Accuracy Improvement (100→300): {improvement:+.1f}%")
            
            best_model = max(summary_results.items(), key=lambda x: x[1]['overall_accuracy'])[0]
            print(f"Best Performance: {best_model}")
            
            # Best performing move types across all models
            print("\nMost Accurately Classified Moves with GPT-5:")
            all_best = {}
            for module in summary_results.values():
                for move, (acc, count) in module['per_move_accuracy'].items():
                    if move not in all_best:
                        all_best[move] = []
                    all_best[move].append(acc)
            
            for move, accs in sorted(all_best.items(), key=lambda x: np.mean(x[1]), reverse=True):
                print(f"  {move.capitalize()}: {np.mean(accs):.1f}%")
    
        print("\n" + "="*60)
        print("KEY INSIGHTS:")
        print("="*60)
        
        # Re-define sizes here for the calculation (FIXED)
        sizes_for_calc = []
        for size in [100, 200, 300]:
            if f'DSPy_MathDial_GPT5_{size}' in summary_results:
                sizes_for_calc.append(size)
        
        if sizes_for_calc:
            # Calculate average scaffolding across all sizes that have results
            avg_scaffolding = np.mean([summary_results[f'DSPy_MathDial_GPT5_{s}']['scaffolding_ratio'] for s in sizes_for_calc])
            avg_telling = np.mean([summary_results[f'DSPy_MathDial_GPT5_{s}']['telling_rate'] for s in sizes_for_calc])
            
            print(f"Average Scaffolding Ratio: {avg_scaffolding:.1f}%")
            print(f"Average Telling Rate: {avg_telling:.1f}%")
            
            if avg_scaffolding > 50:
                print("✓ Good balance: Majority of moves are scaffolding (Focus + Probing)")
            else:
                print("⚠ Consider improvement: Scaffolding ratio below 50%")
        else:
            print("No results available for insight calculation")
    else:
        print("No results to summarize")
    
    print("\n" + "="*60)
    print("DSPY MathDial GPT-5 experiments complete!")
    print("\nFiles saved in results/:")
    print("  - dspy_mathdial_gpt5_results_100.csv, _200.csv, _300.csv")
    print("  - dspy_mathdial_gpt5_all_results.csv")
    print("  - dspy_mathdial_gpt5_summary.json")
    print("  - mathdial_gpt5_module_100_learned.json, _200, _300")
    print("  - dspy_mathdial_gpt5_comparison_chart.png")
    print("  - dspy_mathdial_gpt5_confusion_matrix.png (if seaborn installed)")
    print("="*60)
    print("\nNote: GPT-5 is a reasoning model optimized for complex classification")
    print("It should show significant improvement over GPT-3.5 for understanding")
    print("the pedagogical nuances in teacher-student interactions")
    print("="*60)

if __name__ == "__main__":
    # Check for command line argument
    import sys
    clear_cache = '--clear-cache' in sys.argv
    
    # Check for seaborn (needed for confusion matrix)
    try:
        import seaborn
    except ImportError:
        print("Warning: seaborn not installed. Confusion matrix will be skipped.")
        print("Install with: pip install seaborn")
    
    run_all_experiments(clear_cache=clear_cache)