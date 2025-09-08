"""
MathDial RUBRIC APO Runner
Automatically runs Rubric APO with 3 different evaluation sample sizes: 2, 4, 8 conversations
(Approximately 10, 20, 50 utterances to match other datasets)
Optimizes RUBRIC definitions for teacher moves while keeping prompting strategies at baseline
"""

import sys
import os
import time
from datetime import datetime
import json

# Add parent directory to path
sys.path.append('..')
sys.path.append('.')

# Import the MathDial RUBRIC APO system (correct class name)
from mathdial_apo_system import MathDialRubricAPO

# Import config
from config import (
    OPENAI_API_KEY,
    MODEL_ID,
    MODEL_NAME,
    RESULTS_DIR,
    VALIDATION_SAMPLE_SIZE,
    ACTIVE_POOL_PATH,
    TEACHER_MOVES,
    ACCURACY_THRESHOLD,
    EVALUATION_SAMPLE_SIZES
)

# Define techniques
TECHNIQUES = ['zero_shot', 'few_shot', 'auto_cot', 'self_consistency', 'active_prompting']
NUM_RUBRIC_VARIATIONS = 5  # Generate 5 rubric variations

def check_files_exist():
    """Check if all required files exist"""
    print("Checking required files...")
    
    files_to_check = {
        'APO Training Data': '../data/apo_training_100.csv',
        'Active pool': ACTIVE_POOL_PATH,
        'Zero-shot': '../prompting/zero_shot.py',
        'Few-shot': '../prompting/few_shot.py',
        'Auto-CoT': '../prompting/auto_cot.py',
        'Self-consistency': '../prompting/self_consistency.py',
        'Active prompting': '../prompting/active_prompt.py',
        'MathDial Rubric': '../utils/mathdial_rubric.py'
    }
    
    all_good = True
    for name, path in files_to_check.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: Found")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            all_good = False
    
    return all_good

def run_automated_mathdial_rubric_apo():
    """Run RUBRIC APO optimization for MathDial teacher moves"""
    
    print("="*70)
    print("STARTING AUTOMATED RUBRIC APO FOR MATHDIAL")
    print("OPTIMIZING TEACHER MOVE RUBRIC DEFINITIONS (NOT PROMPTS)")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {OPENAI_API_KEY[:20]}...{OPENAI_API_KEY[-10:]}")
    print(f"APO Training Data: ../data/apo_training_100.csv")
    print(f"Active Pool: {ACTIVE_POOL_PATH}")
    print("="*70)
    
    # Check files exist
    if not check_files_exist():
        print("\nERROR: Some required files are missing!")
        print("Please ensure all files are in the correct locations.")
        return
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\nMATHDIAL RUBRIC APO CONFIGURATION:")
    print(f"Data: Using apo_training_100.csv for optimization")
    print(f"      (holdout_150.csv saved for final testing after APO)")
    print(f"Evaluation Sample Sizes: {EVALUATION_SAMPLE_SIZES} conversations")
    print(f"  (Targeting ~10/20/50 utterances to match other datasets)")
    print(f"Techniques (all at baseline): {len(TECHNIQUES)} ({', '.join(TECHNIQUES)})")
    print(f"Rubric variations to generate: {NUM_RUBRIC_VARIATIONS}")
    print(f"Total rubrics to test: {NUM_RUBRIC_VARIATIONS + 1} (baseline + {NUM_RUBRIC_VARIATIONS} variations)")
    print(f"Teacher Moves: {', '.join(TEACHER_MOVES)}")
    print()
    
    print("KEY DIFFERENCE FROM PROMPT APO:")
    print("-" * 40)
    print("• PROMPTS: Remain at baseline (unchanged)")
    print("• RUBRICS: Will be optimized with 5 variations")
    print("• Each rubric tested with ALL 5 techniques")
    print()
    
    # Calculate total estimates
    total_api_calls = 0
    for eval_size in EVALUATION_SAMPLE_SIZES:
        # 6 rubrics (baseline + 5 variations) × 5 techniques × eval_size samples
        # But now we need to estimate based on utterances not conversations
        estimated_utterances = eval_size * 6  # Average 6 utterances per conversation
        calls_per_run = 6 * 5 * estimated_utterances
        total_api_calls += calls_per_run
        print(f"Eval size {eval_size} conversations (~{estimated_utterances} utterances): ~{calls_per_run} API calls")
    
    # Add calls for generating rubric variations (one call per run)
    total_api_calls += len(EVALUATION_SAMPLE_SIZES)
    
    # Add calls for active prompting pool preparation (if needed)
    total_api_calls += len(EVALUATION_SAMPLE_SIZES) * 20  
    
    total_time_minutes = total_api_calls * 2 / 60  # ~2 seconds per call
    
    # Cost estimate
    total_cost = total_api_calls * 0.002  # Rough estimate for GPT-3.5
    
    print(f"\nTOTAL ESTIMATES:")
    print(f"Total API calls: ~{total_api_calls}")
    print(f"Estimated time: {total_time_minutes:.0f} minutes (~{total_time_minutes/60:.1f} hours)")
    print(f"Estimated cost with {MODEL_NAME}: ${total_cost:.2f}")
    
    # Ask for confirmation
    response = input(f"\nReady to start MathDial RUBRIC APO optimization? (y/n): ").lower().strip()
    if response != 'y':
        print("Rubric APO cancelled.")
        return
    
    print(f"\nStarting rubric optimization...")
    print("This will take approximately {:.1f} hours. Please wait...".format(total_time_minutes/60))
    print()
    
    overall_start_time = time.time()
    all_results = {}
    
    # Run RUBRIC APO for each evaluation sample size
    for i, eval_size in enumerate(EVALUATION_SAMPLE_SIZES, 1):
        print("="*70)
        print(f"RUN {i}/3: EVALUATION_SAMPLE_SIZE = {eval_size} conversations")
        print(f"         (Target: ~{eval_size*5} utterances)")
        print("="*70)
        
        run_start_time = time.time()
        
        try:
            # Initialize MathDial RUBRIC APO for this evaluation size
            print("Initializing MathDial Rubric APO system...")
            print(f"Will optimize RUBRIC definitions while keeping prompts at baseline...")
            
            rubric_apo = MathDialRubricAPO(
                api_key=OPENAI_API_KEY,
                validation_sample_size=VALIDATION_SAMPLE_SIZE,
                evaluation_sample_size=eval_size
            )
            
            print(f"Rubric APO initialized successfully!")
            print(f"Using apo_training_100.csv for APO optimization")
            print(f"Using {eval_size} conversations from APO set for this run")
            
            # Calculate and display actual utterance count
            actual_utterances = sum(len(row.get('ground_truth_moves', [])) 
                                   for _, row in rubric_apo.validation_data.iterrows())
            print(f"  → {actual_utterances} teacher utterances (target: ~{eval_size*5})")
            
            # Store utterance count for later
            if eval_size not in all_results:
                all_results[eval_size] = {}
            all_results[eval_size]['utterance_count'] = actual_utterances
            
            print(f"Will generate {NUM_RUBRIC_VARIATIONS} rubric variations")
            print(f"Testing with all {len(TECHNIQUES)} prompting techniques")
            
            # Run optimization
            print(f"\nStarting rubric optimization...")
            best_candidate = rubric_apo.optimize_rubrics()
            
            # Prepare output data
            output_data = {
                'evaluation_sample_size': eval_size,
                'actual_utterances': actual_utterances,
                'baseline_rubric': rubric_apo.baseline_rubric,
                'best_rubric': best_candidate.rubric_definitions,
                'all_variations': rubric_apo.rubric_variations,  # ALL 5 VARIATIONS SAVED
                'technique_scores': best_candidate.performance_scores,
                'average_score': best_candidate.average_score,
                'detailed_metrics': best_candidate.detailed_metrics
            }
            
            # Save results with eval_size in filename
            results_file = os.path.join(RESULTS_DIR, f"optimized_rubrics_eval{eval_size}.json")
            with open(results_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {results_file}")
            
            # Generate comparison report
            report = generate_mathdial_rubric_report(output_data, eval_size, actual_utterances)
            report_file = os.path.join(RESULTS_DIR, f"rubric_comparison_eval{eval_size}.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            print(f"Report saved to: {report_file}")
            
            # Track results
            all_results[eval_size]['best_candidate'] = best_candidate
            all_results[eval_size]['results_file'] = results_file
            all_results[eval_size]['report_file'] = report_file
            all_results[eval_size]['average_score'] = best_candidate.average_score
            
            run_end_time = time.time()
            run_elapsed = (run_end_time - run_start_time) / 60
            
            print(f"\nRUN {i} COMPLETE!")
            print(f"Time: {run_elapsed:.1f} minutes")
            print(f"Conversations: {eval_size}, Utterances: {actual_utterances}")
            print(f"Average score across all techniques: {best_candidate.average_score:.3f}")
            print(f"Best performing technique: {max(best_candidate.performance_scores, key=best_candidate.performance_scores.get)}")
            print()
            
        except Exception as e:
            print(f"Error during run {i} (eval_size={eval_size}): {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next run...")
            continue
    
    overall_end_time = time.time()
    total_elapsed = (overall_end_time - overall_start_time) / 60
    
    # Final summary
    print("="*70)
    print("AUTOMATED MATHDIAL RUBRIC APO COMPLETE!")
    print("="*70)
    print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
    print(f"Total runs completed: {len(all_results)}")
    print()
    
    print("FILES USED/CREATED:")
    print("-" * 40)
    print("INPUT FILES (pre-existing):")
    print("  • ../data/apo_training_100.csv - APO training data")
    print("  • ../data/active_prompting_pool_20.csv - Active prompting examples")
    print("")
    print("OUTPUT FILES (created by APO):")
    for i, eval_size in enumerate(EVALUATION_SAMPLE_SIZES, 1):
        if eval_size in all_results:
            print(f"  {i}. results/optimized_rubrics_eval{eval_size}.json - Best rubrics from {eval_size} conversations")
            print(f"     results/rubric_comparison_eval{eval_size}.txt - Comparison report")

    print("\nRESULTS SUMMARY:")
    print("-" * 40)
    print("Conversations | Utterances | Average Score | Best Technique")
    print("-" * 60)
    for eval_size in EVALUATION_SAMPLE_SIZES:
        if eval_size in all_results:
            data = all_results[eval_size]
            utterance_count = data.get('utterance_count', eval_size * 6)
            best_tech = max(data['best_candidate'].performance_scores, 
                           key=data['best_candidate'].performance_scores.get)
            print(f"     {eval_size:2d}      |    {utterance_count:3d}     |    {data['average_score']:.3f}      | {best_tech}")
    
    print("\nKEY INSIGHTS:")
    print("• All prompting strategies remained at baseline")
    print("• Only rubric definitions were optimized")
    print(f"• Generated and tested {NUM_RUBRIC_VARIATIONS} rubric variations")
    print("• Each variation tested with ALL 5 techniques")
    print("• Conversation counts chosen to match utterance volumes of other datasets")
    
    print("\nNEXT STEPS:")
    print("1. Review the JSON files to see all 5 rubric variations")
    print("2. Check which rubric performed best for each technique")
    print("3. Test the best rubric on holdout_150.csv samples")
    print("4. Compare improvement from baseline to optimized rubric")

def generate_mathdial_rubric_report(output_data: dict, eval_size: int, actual_utterances: int) -> str:
    """Generate a comparison report for MathDial rubric optimization"""
    
    report = f"MATHDIAL RUBRIC OPTIMIZATION REPORT - {eval_size} CONVERSATIONS\n"
    report += "="*60 + "\n\n"
    
    report += "CONFIGURATION:\n"
    report += f"• Model: {MODEL_NAME}\n"
    report += f"• Evaluation conversations: {eval_size}\n"
    report += f"• Actual teacher utterances: {actual_utterances}\n"
    report += f"• Rubric variations tested: 5 + baseline = 6 total\n"
    report += f"• Techniques tested: 5 (all at baseline)\n\n"
    
    report += "BASELINE RUBRIC:\n"
    report += "-"*40 + "\n"
    for move, definition in output_data['baseline_rubric'].items():
        report += f"• {move.capitalize()}: {definition[:60]}...\n"
    
    report += "\n" + "BEST RUBRIC:\n"
    report += "-"*40 + "\n"
    for move, definition in output_data['best_rubric'].items():
        report += f"• {move.capitalize()}: {definition[:60]}...\n"
    
    report += "\n" + "PERFORMANCE BY TECHNIQUE:\n"
    report += "-"*40 + "\n"
    for technique, score in output_data['technique_scores'].items():
        report += f"• {technique:20s}: {score:.3f}\n"
    
    report += f"\nAVERAGE SCORE: {output_data['average_score']:.3f}\n"
    
    report += "\n" + "ALL 5 VARIATIONS TESTED:\n"
    report += "-"*40 + "\n"
    for i, variation in enumerate(output_data['all_variations'], 1):
        report += f"Variation {i}:\n"
        report += f"  Generic: {variation.get('generic', 'N/A')[:50]}...\n"
    
    return report

def show_mathdial_rubric_apo_demo():
    """Show what MathDial RUBRIC APO will do without actually running it"""
    
    print("MATHDIAL RUBRIC APO DEMO MODE")
    print("="*50)
    print("This shows what MathDial RUBRIC APO will do:\n")
    
    print("1. DATA USED:")
    print("   - APO optimization: apo_training_100.csv")
    print("   - Active pool: active_prompting_pool_20.csv")
    print("   - Holdout: holdout_150.csv (NOT used during APO)\n")
    
    print("2. SAMPLING STRATEGY:")
    print("   - Run 1: 2 conversations (~10 utterances)")
    print("   - Run 2: 4 conversations (~20 utterances)")
    print("   - Run 3: 8 conversations (~50 utterances)")
    print("   Matches utterance counts of other datasets\n")
    
    print("3. RUBRIC GENERATION:")
    print("   Will generate 5 variations of teacher move definitions")
    print("   Using the exact prompt from the paper\n")
    
    print("4. WHAT STAYS THE SAME (baseline):")
    print("   ✓ zero_shot.py")
    print("   ✓ few_shot.py")
    print("   ✓ auto_cot.py")
    print("   ✓ self_consistency.py")
    print("   ✓ active_prompt.py\n")
    
    print("5. WHAT CHANGES:")
    print("   ✗ mathdial_rubric.py definitions (6 variations tested)\n")
    
    print("6. TESTING PROCESS:")
    print("   For each rubric (baseline + 5 variations):")
    print("   - Update mathdial_rubric.py")
    print("   - Test with ALL 5 techniques")
    print("   - Measure accuracy")
    print("   - Save results\n")
    
    print("7. OUTPUT:")
    print("   - 3 JSON files (eval2, eval4, eval8)")
    print("   - Each contains ALL 5 variations")
    print("   - Best rubric identified")
    print("   - Scores for each technique")

def check_requirements():
    """Check if everything is set up correctly for MathDial"""
    
    print("CHECKING SYSTEM REQUIREMENTS FOR MATHDIAL RUBRIC APO")
    print("="*50)
    
    # Check Python packages
    required_packages = ['pandas', 'openai', 'sklearn', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}: Found")
        except ImportError:
            print(f"✗ {package}: Missing")
            missing_packages.append(package)
    
    # Check model configuration
    print(f"\n✓ Model configured: {MODEL_NAME}")
    
    # Check data files
    print("\nData files:")
    apo_path = '../data/apo_training_100.csv'
    if os.path.exists(apo_path):
        print(f"✓ APO data: Found at {apo_path}")
        try:
            import pandas as pd
            df = pd.read_csv(apo_path)
            print(f"  {len(df)} rows loaded")
        except Exception as e:
            print(f"✗ Data loading error: {e}")
    else:
        print(f"✗ APO data: Not found at {apo_path}")
    
    if os.path.exists(ACTIVE_POOL_PATH):
        print(f"✓ Active pool: Found at {ACTIVE_POOL_PATH}")
    else:
        print(f"✗ Active pool: Not found at {ACTIVE_POOL_PATH}")
    
    # Check rubric file
    rubric_path = '../utils/mathdial_rubric.py'
    if os.path.exists(rubric_path):
        print(f"\n✓ MathDial rubric file: Found")
        with open(rubric_path, 'r') as f:
            content = f.read()
            if 'move_definitions' in content:
                print("  → Contains move definitions")
    else:
        print(f"\n✗ MathDial rubric file: Not found at {rubric_path}")
    
    # Check API key
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
        print(f"\n✓ API key: Configured")
    else:
        print(f"\n✗ API key: Not properly configured")
    
    # Summary
    if missing_packages:
        print(f"\nTO FIX MISSING PACKAGES:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    all_good = not missing_packages and os.path.exists(apo_path) and OPENAI_API_KEY
    print(f"\nREADY TO RUN MATHDIAL RUBRIC APO: {'Yes ✓' if all_good else 'No ✗'}")

def main():
    """Main function with menu options"""
    
    print("="*70)
    print("MATHDIAL TEACHER MOVES RUBRIC APO SYSTEM")
    print("Optimizing RUBRIC definitions (not prompts)")
    print("Using Model: " + MODEL_NAME)
    print("="*70)
    print("\nChoose an option:")
    print("1. Run automated RUBRIC APO optimization (2-3 hours)")
    print("2. Show demo/simulation (1 minute)")
    print("3. Check system requirements")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_automated_mathdial_rubric_apo()
    elif choice == "2":
        show_mathdial_rubric_apo_demo()
    elif choice == "3":
        check_requirements()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()