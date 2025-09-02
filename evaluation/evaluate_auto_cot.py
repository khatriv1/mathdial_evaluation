# mathdial_evaluation/evaluation/evaluate_auto_cot.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.auto_cot import get_auto_cot_prediction
from utils.data_loader import MathDialDataLoader
from utils.metrics import MathDialMetrics

def evaluate_auto_cot(data_path: str, api_key: str, output_dir: str = "results/auto_cot", limit: int = None):
    """Evaluate Auto-CoT prompting technique on MathDial dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        loader = MathDialDataLoader(data_path)
        df, teacher_moves = loader.load_data()
        
        if df.empty:
            raise Exception("No valid conversations found in the data file")
        
        if limit:
            df = df.head(limit)
        print(f"\nEvaluating on {len(df)} conversations")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize metrics calculator
    metrics_calc = MathDialMetrics(teacher_moves)
    
    # Store results
    all_ground_truth = []
    all_predictions = []
    detailed_results = []
    
    # Process each conversation
    total = len(df)
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
            student_solution = row['student_incorrect_solution']
            student_profile = row.get('student_profile', '')
            
            # Display the conversation for verification
            print("\n--- Conversation ---")
            lines = conversation.split('\n')
            for i, line in enumerate(lines[:10]):  # Show first 10 lines
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... ({len(lines)-10} more lines)")
            print("--- End Conversation ---\n")
            
            # Parse conversation and classify each teacher utterance
            context = ""
            predicted_moves = []
            teacher_utterances = []
            
            for line in lines:
                if line.startswith('Teacher:'):
                    utterance = line.replace('Teacher:', '').strip()
                    
                    if utterance:
                        # Get Auto-CoT prediction WITH FULL CONTEXT
                        prediction = get_auto_cot_prediction(
                            utterance, context, client,
                            question=question,
                            student_solution=student_solution,
                            student_profile=student_profile
                        )
                        predicted_moves.append(prediction)
                        teacher_utterances.append(utterance)
                    
                    context += f"Teacher: {utterance}\n"
                    
                elif line.startswith('Student:'):
                    student_utterance = line.replace('Student:', '').strip()
                    context += f"Student: {student_utterance}\n"
            
            all_predictions.append(predicted_moves)
            
            # Display utterance-by-utterance comparison
            print("\nUtterance-by-utterance analysis:")
            for i in range(min(5, len(teacher_utterances))):  # Show first 5
                ground_truth = ground_truth_moves[i] if i < len(ground_truth_moves) else "N/A"
                prediction = predicted_moves[i] if i < len(predicted_moves) else "N/A"
                match = "✓" if ground_truth == prediction else "✗"
                print(f"  {i+1}. Teacher: \"{teacher_utterances[i][:50]}...\"")
                print(f"     Ground truth: {ground_truth}, Predicted: {prediction} {match}")
            
            if len(teacher_utterances) > 5:
                print(f"  ... ({len(teacher_utterances)-5} more utterances)")
            
            # Calculate metrics for this conversation
            exact_match = (len(ground_truth_moves) == len(predicted_moves) and 
                          all(g == p for g, p in zip(ground_truth_moves, predicted_moves)))
            
            min_len = min(len(ground_truth_moves), len(predicted_moves))
            if min_len > 0:
                move_matches = sum(1 for i in range(min_len) 
                                 if ground_truth_moves[i] == predicted_moves[i])
                move_accuracy = move_matches / len(ground_truth_moves)
            else:
                move_accuracy = 0.0
            
            # Store detailed result
            detailed_results.append({
                'conversation_id': conversation_id,
                'question': row['question'][:100],
                'num_teacher_moves': len(ground_truth_moves),
                'ground_truth_moves': ground_truth_moves,
                'predicted_moves': predicted_moves,
                'exact_match': exact_match,
                'move_accuracy': move_accuracy,
                'conversation_snippet': conversation[:200] + "..."
            })
            
            print(f"\nSummary:")
            print(f"  Ground truth moves: {ground_truth_moves}")
            print(f"  Predicted moves: {predicted_moves}")
            print(f"  Exact match: {exact_match}")
            print(f"  Move accuracy: {move_accuracy:.3f}")
            
        except Exception as e:
            print(f"Error processing conversation {conversation_id}: {str(e)}")
            continue
        
        time.sleep(1)  # Rate limiting
    
    if not all_ground_truth:
        raise Exception("No valid predictions were generated")
    
    # Calculate metrics
    metrics = metrics_calc.comprehensive_evaluation(all_ground_truth, all_predictions)
    
    # Print results
    print("\n" + "="*60)
    metrics_calc.print_results(metrics, "Auto-CoT")
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Auto-CoT',
        'accuracy': metrics['accuracy'],
        'cohens_kappa': metrics['cohens_kappa'],
        'krippendorffs_alpha': metrics['krippendorffs_alpha'],
        'icc': metrics['icc'],
        'num_samples': metrics['num_samples']
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print(f"\nStarting Auto-CoT evaluation on MathDial dataset...")
        print(f"Using data file: {config.DATA_PATH}")
        
        results, metrics = evaluate_auto_cot(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=None  # Set to small number for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())