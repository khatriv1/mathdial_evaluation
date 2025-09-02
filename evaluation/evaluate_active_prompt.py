# mathdial_evaluation/evaluation/evaluate_active_prompt.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List, Tuple

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.active_prompt import (
    get_active_prompt_prediction,
    prepare_active_prompting_data
)
from utils.data_loader import MathDialDataLoader
from utils.metrics import MathDialMetrics

def evaluate_active_prompt(data_path: str, api_key: str, output_dir: str = "results/active_prompt", limit: int = None):
    """Evaluate Active Prompting technique on MathDial dataset with separate pool."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess test data
    try:
        loader = MathDialDataLoader(data_path)
        df, teacher_moves = loader.load_data(sample_size=limit)
        
        if df.empty:
            raise Exception("No valid conversations found in the data file")
        
        print(f"\nTotal samples for evaluation: {len(df)}")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize metrics calculator
    metrics_calc = MathDialMetrics(teacher_moves)
    
    # ========== PREPARE ACTIVE PROMPTING DATA USING SEPARATE POOL ==========
    print("\n" + "="*60)
    print("PHASE 1: PREPARING ACTIVE PROMPTING EXAMPLES")
    print("="*60)
    
    # Load the separate pool file from data folder
    pool_path = 'data/active_prompting_pool_20.csv'
    
    if os.path.exists(pool_path):
        print(f"Loading separate pool from: {pool_path}")
        pool_loader = MathDialDataLoader(pool_path)
        pool_df, _ = pool_loader.load_data()
        print(f"Pool size: {len(pool_df)} samples")
        print("✓ Using separate pool - NO OVERLAP with test set!")
        print("✓ Can evaluate on FULL test set without bias!")
    else:
        # Fallback if pool file doesn't exist
        print(f"WARNING: Pool file not found at {pool_path}")
        print("Falling back to using first 20 samples from test set")
        pool_loader = MathDialDataLoader(data_path)
        pool_df, _ = pool_loader.load_data(sample_size=20)
        # If using fallback, need to skip those samples in evaluation
        if limit is None or limit > 20:
            # Reload data skipping first 20
            df_full, _ = loader.load_data()
            df = df_full.iloc[20:limit] if limit else df_full.iloc[20:]
            print(f"Note: Evaluation will skip first 20 samples to avoid bias")
    
    # Get 8 examples (1 uncertain + 1 wrong per category) from the pool
    uncertainty_data = prepare_active_prompting_data(pool_df, client)
    
    if not uncertainty_data:
        print("Warning: No uncertainty data prepared. Using empty examples.")
        uncertainty_data = []
    else:
        print(f"\n✓ Prepared {len(uncertainty_data)} examples for Active Prompting")
        
        # Display the examples
        print("\nActive Prompting Examples (from separate pool):")
        for i, (utterance, classification) in enumerate(uncertainty_data, 1):
            print(f"  Example {i}: {utterance[:60]}... → {classification}")
    
    # ========== EVALUATION PHASE ==========
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION WITH ACTIVE PROMPTING")
    print("="*60)
    
    # Store results
    all_ground_truth = []
    all_predictions = []
    detailed_results = []
    
    # Process each conversation
    total = len(df)
    print(f"\nEvaluating on {total} test conversations")
    if os.path.exists(pool_path):
        print("(Testing on FULL dataset - pool is separate, no bias!)")
    else:
        print("(Testing on remaining samples after excluding pool)")
    
    for seq, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\n{'='*60}")
        print(f"Processing conversation {seq}/{total}")
        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Question: {row['question'][:100]}...")
        
        conversation_id = row['conversation_id']
        
        try:
            # Get ground truth moves
            ground_truth_moves = row['ground_truth_moves']
            all_ground_truth.append(ground_truth_moves)
            
            # Get full context
            conversation = row['cleaned_conversation']
            question = row['question']
            student_solution = row.get('student_incorrect_solution', '')
            student_profile = row.get('student_profile', '')
            
            # Parse conversation and classify each teacher utterance
            lines = conversation.split('\n')
            context = ""
            predicted_moves = []
            teacher_utterances = []
            
            for line in lines:
                if line.startswith('Teacher:'):
                    utterance = line.replace('Teacher:', '').strip()
                    
                    if utterance:
                        # Get prediction with context up to this point
                        prediction = get_active_prompt_prediction(
                            utterance, 
                            context,
                            client,
                            question=question,
                            student_solution=student_solution,
                            student_profile=student_profile,
                            uncertainty_data=uncertainty_data
                        )
                        predicted_moves.append(prediction)
                        teacher_utterances.append(utterance)
                    
                    # Add to context for next utterance
                    context += f"Teacher: {utterance}\n"
                    
                elif line.startswith('Student:'):
                    student_utterance = line.replace('Student:', '').strip()
                    context += f"Student: {student_utterance}\n"
            
            all_predictions.append(predicted_moves)
            
            # Calculate position-by-position accuracy for this conversation
            conversation_correct = sum(1 for i in range(min(len(ground_truth_moves), len(predicted_moves)))
                                     if ground_truth_moves[i] == predicted_moves[i])
            move_accuracy = (conversation_correct / len(ground_truth_moves)) * 100 if ground_truth_moves else 0
            
            # Store detailed result
            detailed_results.append({
                'conversation_id': conversation_id,
                'question': row['question'][:100],
                'num_teacher_moves': len(ground_truth_moves),
                'ground_truth_moves': ground_truth_moves,
                'predicted_moves': predicted_moves,
                'position_accuracy': move_accuracy,
                'correct_positions': conversation_correct,
                'num_uncertainty_examples': len(uncertainty_data)
            })
            
            print(f"  Ground truth: {ground_truth_moves}")
            print(f"  Predictions:  {predicted_moves}")
            print(f"  Position accuracy: {move_accuracy:.1f}% ({conversation_correct}/{len(ground_truth_moves)})")
            
        except Exception as e:
            print(f"Error processing conversation {conversation_id}: {str(e)}")
            all_predictions.append([])
            continue
        
        time.sleep(1)  # Rate limiting
    
    # Calculate metrics using position-by-position accuracy
    metrics = metrics_calc.comprehensive_evaluation(all_ground_truth, all_predictions)
    
    # Print results
    print("\n" + "="*60)
    metrics_calc.print_results(metrics, "Active Prompting (8 Examples, Separate Pool)")
    
    # Save results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    # Save the uncertainty examples used
    if uncertainty_data:
        uncertainty_df = pd.DataFrame(uncertainty_data, columns=['teacher_utterance', 'classification'])
        uncertainty_df.to_csv(f"{output_dir}/uncertainty_examples.csv", index=False)
        print(f"Uncertainty examples saved to {output_dir}/uncertainty_examples.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Active Prompting (8 Examples)',
        'position_accuracy': metrics['accuracy'],
        'total_correct': metrics['total_correct'],
        'total_positions': metrics['total_positions'],
        'mean_conversation_accuracy': metrics['mean_conversation_accuracy'],
        'exact_match_accuracy': metrics['exact_match_accuracy'],
        'cohens_kappa': metrics['cohens_kappa'],
        'krippendorffs_alpha': metrics['krippendorffs_alpha'],
        'icc': metrics['icc'],
        'num_samples': metrics['num_samples'],
        'num_uncertainty_examples': len(uncertainty_data)
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total examples prepared: {len(uncertainty_data)}")
    print(f"Position Accuracy: {metrics['accuracy']:.1f}%")
    
    return detailed_results, metrics

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    try:
        print("\nStarting Active Prompting evaluation (Separate Pool, No Bias)...")
        print("Using POSITION-BY-POSITION accuracy metrics")
        print(f"Using test data: {config.DATA_PATH}")
        print(f"Using pool data: data/active_prompting_pool_20.csv")
        
        results, metrics = evaluate_active_prompt(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=None  # Set to None for full evaluation or a number for testing
        )
        print(f"\nFinal Position Accuracy: {metrics['accuracy']:.1f}%")
        
    except Exception as e:
        print(f"\n✗ Active Prompting failed: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())