# dspy_implementation/dspy_mathdial_classifier.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
import pandas as pd
import numpy as np
import json
from typing import Dict, List
import time

# Import from parent directory
from config import OPENAI_API_KEY
# Remove the calculate_agreement_metrics import - not needed for MathDial
from utils.data_loader import MathDialDataLoader

# Configure DSPy with OpenAI
print(f"Setting up OpenAI API...")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# DSPy v3 configuration
lm = dspy.LM('openai/gpt-3.5-turbo-0125', api_key=OPENAI_API_KEY, max_tokens=500, temperature=0)
dspy.configure(lm=lm)

# Test the API connection
print("Testing API connection...")
try:
    test_prompt = "Say 'API working' if you can read this"
    test_response = lm(test_prompt)
    print(f"API Test successful: Connection established")
except Exception as e:
    print(f"WARNING: API test failed: {e}")
    print("Please check your API key and internet connection")

class MathDialSignature(dspy.Signature):
    """Classify teacher utterances in math tutoring dialogues."""
    
    teacher_utterance = dspy.InputField(desc="The teacher's utterance to classify")
    conversation_context = dspy.InputField(desc="Previous conversation context")
    question = dspy.InputField(desc="The math problem being discussed")
    student_solution = dspy.InputField(desc="Student's incorrect solution")
    
    teacher_move = dspy.OutputField(desc="One of: generic, focus, probing, telling")
    reasoning = dspy.OutputField(desc="Brief explanation of why this move was chosen")

class MathDialClassifier(dspy.Module):
    """DSPy module for MathDial teacher move classification"""
    
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(MathDialSignature)
    
    def forward(self, teacher_utterance, conversation_context="", question="", student_solution=""):
        prediction = self.prog(
            teacher_utterance=teacher_utterance,
            conversation_context=conversation_context,
            question=question,
            student_solution=student_solution
        )
        return prediction

def load_training_data(csv_path: str, sample_size: int = None):
    """Load and prepare MathDial training data for DSPy"""
    
    print(f"Loading MathDial data from {csv_path}...")
    loader = MathDialDataLoader(csv_path)
    df, teacher_moves = loader.load_data(sample_size=sample_size)
    
    print(f"Loaded {len(df)} conversations")
    
    examples = []
    
    # Process each conversation
    for _, row in df.iterrows():
        full_conversation = row.get('cleaned_conversation', '')
        ground_truth_moves = row.get('ground_truth_moves', [])
        question = row.get('question', '')
        student_solution = row.get('student_incorrect_solution', '')
        
        # Parse conversation into utterances
        lines = full_conversation.split('\n')
        conversation_context = ""
        teacher_utterance_idx = 0
        
        for line in lines:
            if line.startswith('Teacher:'):
                teacher_utterance = line.replace('Teacher:', '').strip()
                
                if teacher_utterance and teacher_utterance_idx < len(ground_truth_moves):
                    # Create DSPy example
                    example = dspy.Example(
                        teacher_utterance=teacher_utterance,
                        conversation_context=conversation_context if conversation_context else "Start of conversation",
                        question=question[:200] if question else "Math problem",  # Truncate if too long
                        student_solution=student_solution[:200] if student_solution else "Student work",
                        teacher_move=ground_truth_moves[teacher_utterance_idx],
                        reasoning=f"Teacher uses {ground_truth_moves[teacher_utterance_idx]} strategy"
                    ).with_inputs('teacher_utterance', 'conversation_context', 'question', 'student_solution')
                    
                    examples.append(example)
                    teacher_utterance_idx += 1
                
                conversation_context += f"Teacher: {teacher_utterance}\n"
                
            elif line.startswith('Student:'):
                student_utterance = line.replace('Student:', '').strip()
                conversation_context += f"Student: {student_utterance}\n"
    
    print(f"Created {len(examples)} training examples from conversations")
    return examples

def mathdial_metric(example, pred, trace=None):
    """Metric for MathDial - exact match on teacher move"""
    try:
        true_move = example.teacher_move.lower().strip()
        pred_move = pred.teacher_move.lower().strip()
        
        # Check if prediction is one of valid moves
        valid_moves = ['generic', 'focus', 'probing', 'telling']
        if pred_move not in valid_moves:
            return False
        
        return true_move == pred_move
    except:
        return False

def train_dspy_module(training_examples, sample_size):
    """Train a DSPy module for MathDial with specific sample size"""
    print(f"\nTraining DSPy module with {sample_size} utterance examples...")
    print("DSPy is learning teacher move patterns from data...")
    
    # Use subset of training examples
    train_set = training_examples[:sample_size]
    
    # Initialize module
    mathdial_module = MathDialClassifier()
    
    # Configure optimizer
    from dspy.teleprompt import BootstrapFewShot
    
    teleprompter = BootstrapFewShot(
        metric=mathdial_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        max_rounds=1
    )
    
    # Compile the module
    print(f"Optimizing module with {len(train_set)} examples...")
    start_time = time.time()
    
    try:
        optimized_module = teleprompter.compile(mathdial_module, trainset=train_set)
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f} seconds")
        
        if elapsed < 10:
            print("Note: Training used cached responses for speed")
        else:
            print("Training with fresh API calls completed")
    except Exception as e:
        print(f"Training error: {e}")
        print("Returning base module without optimization")
        return mathdial_module
    
    # Save module configuration
    os.makedirs('results', exist_ok=True)
    
    try:
        learned_file = f'results/mathdial_module_{sample_size}_learned.json'
        
        learned_data = {
            'sample_size': sample_size,
            'training_time': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'module_structure': {
                'type': 'ChainOfThought',
                'signature': 'MathDialSignature',
                'inputs': ['teacher_utterance', 'conversation_context', 'question', 'student_solution'],
                'outputs': ['teacher_move', 'reasoning'],
                'teacher_moves': ['generic', 'focus', 'probing', 'telling']
            },
            'optimization': {
                'method': 'BootstrapFewShot',
                'max_demos': 3,
                'max_labeled': 4,
                'optimization_rounds': 1
            }
        }
        
        with open(learned_file, 'w') as f:
            json.dump(learned_data, f, indent=2)
        
        print(f"Module configuration saved to {learned_file}")
        
    except Exception as e:
        print(f"Note: Could not save module details: {e}")
    
    print(f"MathDial module trained with {sample_size} examples")
    return optimized_module

def test_dspy_module(module, test_csv_path, module_name):
    """Test DSPy module on MathDial holdout set"""
    
    print(f"\nTesting {module_name} on holdout set...")
    
    # Load test data
    loader = MathDialDataLoader(test_csv_path)
    test_df, _ = loader.load_data()
    
    all_predictions = []
    all_true_labels = []
    
    start_time = time.time()
    api_calls = 0
    
    # Process each conversation
    for idx, row in test_df.iterrows():
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {idx + 1}/{len(test_df)} conversations - Time: {elapsed:.1f}s")
        
        full_conversation = row.get('cleaned_conversation', '')
        ground_truth_moves = row.get('ground_truth_moves', [])
        question = row.get('question', '')
        student_solution = row.get('student_incorrect_solution', '')
        
        # Parse conversation
        lines = full_conversation.split('\n')
        conversation_context = ""
        teacher_utterance_idx = 0
        
        for line in lines:
            if line.startswith('Teacher:'):
                teacher_utterance = line.replace('Teacher:', '').strip()
                
                if teacher_utterance and teacher_utterance_idx < len(ground_truth_moves):
                    true_label = ground_truth_moves[teacher_utterance_idx]
                    
                    try:
                        # Get DSPy prediction
                        prediction = module(
                            teacher_utterance=teacher_utterance,
                            conversation_context=conversation_context if conversation_context else "Start of conversation",
                            question=question[:200] if question else "Math problem",
                            student_solution=student_solution[:200] if student_solution else "Student work"
                        )
                        api_calls += 1
                        
                        pred_move = prediction.teacher_move.lower().strip()
                        
                        # Validate prediction
                        if pred_move not in ['generic', 'focus', 'probing', 'telling']:
                            pred_move = 'generic'  # Default fallback
                        
                    except Exception as e:
                        print(f"Error predicting utterance in conversation {idx}: {e}")
                        pred_move = 'generic'
                    
                    all_predictions.append(pred_move)
                    all_true_labels.append(true_label)
                    teacher_utterance_idx += 1
                
                conversation_context += f"Teacher: {teacher_utterance}\n"
                
            elif line.startswith('Student:'):
                student_utterance = line.replace('Student:', '').strip()
                conversation_context += f"Student: {student_utterance}\n"
        
        # Rate limiting
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"Testing completed in {total_time:.1f} seconds ({api_calls} API calls)")
    
    # Calculate metrics using sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='weighted', zero_division=0
    )
    kappa = cohen_kappa_score(all_true_labels, all_predictions)
    
    print(f"\nResults for {module_name}:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  Kappa: {kappa:.3f}")
    print(f"  Total utterances evaluated: {len(all_predictions)}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'technique': module_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'total_utterances': len(all_predictions)
    }, index=[0])
    
    return results_df

def test_api_connection():
    """Test if API is working properly for MathDial"""
    print("\n" + "="*60)
    print("TESTING API CONNECTION FOR MATHDIAL")
    print("="*60)
    
    test_utterance = "Good job! Now can you tell me what the next step would be?"
    test_context = "Student: I multiplied 5 by 3 to get 15."
    
    print(f"Test utterance: {test_utterance}")
    print(f"Context: {test_context}")
    
    try:
        classifier = MathDialClassifier()
        start = time.time()
        result = classifier(
            teacher_utterance=test_utterance,
            conversation_context=test_context,
            question="Solve 5 x 3 + 2",
            student_solution="5 x 3 = 15"
        )
        elapsed = time.time() - start
        
        print(f"\nAPI call took: {elapsed:.2f} seconds")
        print(f"Results:")
        print(f"  Teacher Move: {result.teacher_move}")
        print(f"  Reasoning: {result.reasoning}")
        
        if elapsed < 0.5:
            print("\nNote: Using cached responses for speed")
        else:
            print("\nSUCCESS: Fresh API call completed!")
            
    except Exception as e:
        print(f"ERROR: API test failed - {e}")
        print("Check your API key in config.py")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    test_api_connection()