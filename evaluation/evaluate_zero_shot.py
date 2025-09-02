# mathdial_evaluation/evaluation/evaluate_zero_shot.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List, Tuple

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.zero_shot import get_zero_shot_prediction
from utils.data_loader import MathDialDataLoader
from utils.metrics import MathDialMetrics

def evaluate_zero_shot(data_path: str, api_key: str, output_dir: str = "results/zero_shot", limit: int = None):
    """Evaluate Zero-shot prompting technique on MathDial dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        loader = MathDialDataLoader(data_path)
        df, teacher_moves = loader.load_data(sample_size=limit)
        
        if df.empty:
            raise Exception("No valid conversations found in the data file")
        
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
                        prediction = get_zero_shot_prediction(
                            utterance, 
                            context,
                            client,
                            question=question,
                            student_solution=student_solution,
                            student_profile=student_profile
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
                'correct_positions': conversation_correct
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
    metrics_calc.print_results(metrics, "Zero-shot")
    
    # Save results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    metrics_summary = {
        'technique': 'Zero-shot',
        'position_accuracy': metrics['accuracy'],
        'total_correct': metrics['total_correct'],
        'total_positions': metrics['total_positions'],
        'mean_conversation_accuracy': metrics['mean_conversation_accuracy'],
        'exact_match_accuracy': metrics['exact_match_accuracy'],
        'cohens_kappa': metrics['cohens_kappa'],
        'krippendorffs_alpha': metrics['krippendorffs_alpha'],
        'icc': metrics['icc'],
        'num_samples': metrics['num_samples']
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    return detailed_results, metrics

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    try:
        print("\nStarting Zero-shot evaluation on MathDial dataset...")
        print("Using POSITION-BY-POSITION accuracy metrics")
        
        results, metrics = evaluate_zero_shot(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=None
        )
        print(f"\nFinal Position Accuracy: {metrics['accuracy']:.1f}%")
        
    except Exception as e:
        print(f"\nError: {str(e)}")