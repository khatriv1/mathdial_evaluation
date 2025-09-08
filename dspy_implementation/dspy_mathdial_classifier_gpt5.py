# dspy_implementation/dspy_mathdial_classifier_gpt5.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import time
import re

# FIX for GPT-5 temperature variations during optimization
import litellm
litellm.drop_params = True  # Drop unsupported parameter variations

# Import from parent directory
from config import OPENAI_API_KEY

# Configure DSPy with GPT-5
print(f"Setting up GPT-5 API for MathDial...")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# DSPy v3 configuration with GPT-5 - REQUIRED parameters for reasoning models
lm = dspy.LM('openai/gpt-5', api_key=OPENAI_API_KEY, temperature=1.0, max_tokens=20000)
dspy.configure(lm=lm)

# Test the API connection
print("Testing GPT-5 API connection...")
try:
    test_prompt = "Say 'GPT-5 API working' if you can read this"
    test_response = lm(test_prompt)
    print(f"GPT-5 API Test successful: Connection established")
except Exception as e:
    print(f"WARNING: GPT-5 API test failed: {e}")
    print("Please check your API key and GPT-5 access")

# MathDial teacher move categories
TEACHER_MOVES = ['focus', 'probing', 'telling', 'generic']

class TeacherUtterance:
    """Represents a single teacher utterance to classify"""
    def __init__(self, text: str, label: str = None):
        self.text = text
        self.label = label  # Ground truth label
        
class MathDialSignature(dspy.Signature):
    """Classify teacher utterances in math tutoring conversations."""
    
    # Context about the math problem
    problem = dspy.InputField(desc="The math problem being discussed")
    student_solution = dspy.InputField(desc="The student's incorrect solution attempt")
    
    # The conversation context
    conversation_history = dspy.InputField(desc="Previous exchanges in the conversation")
    teacher_utterance = dspy.InputField(desc="The current teacher utterance to classify")
    
    # Output classification
    move_type = dspy.OutputField(desc="Teacher move type: one of 'focus', 'probing', 'telling', or 'generic'")
    reasoning = dspy.OutputField(desc="Brief explanation for the classification")

class MathDialClassifier(dspy.Module):
    """DSPy module for MathDial teacher move classification"""
    
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(MathDialSignature)
    
    def forward(self, problem, student_solution, conversation_history, teacher_utterance):
        prediction = self.prog(
            problem=problem,
            student_solution=student_solution,
            conversation_history=conversation_history,
            teacher_utterance=teacher_utterance
        )
        return prediction

def parse_conversation(conversation_text: str) -> Tuple[List[TeacherUtterance], Dict]:
    """
    Parse MathDial conversation to extract teacher utterances and labels
    
    Format: Teacher: (focus)Text here|EOM|Student: Response|EOM|...
    
    Returns:
        Tuple of (teacher_utterances, metadata)
    """
    utterances = []
    metadata = {
        'problem': '',
        'student_solution': '',
        'full_conversation': []
    }
    
    # Split by |EOM| markers
    exchanges = conversation_text.split('|EOM|')
    
    for exchange in exchanges:
        exchange = exchange.strip()
        if not exchange:
            continue
            
        # Extract label if present (e.g., "Teacher: (focus)...")
        label_match = re.match(r'Teacher:\s*\((\w+)\)\s*(.*)', exchange, re.DOTALL)
        
        if label_match:
            label = label_match.group(1).lower()
            text = label_match.group(2).strip()
            utterances.append(TeacherUtterance(text, label))
            metadata['full_conversation'].append(('Teacher', text, label))
        elif exchange.startswith('Teacher:'):
            # Teacher without label (shouldn't happen in training data)
            text = exchange.replace('Teacher:', '').strip()
            utterances.append(TeacherUtterance(text, None))
            metadata['full_conversation'].append(('Teacher', text, None))
        elif exchange.startswith('Student:'):
            # Student utterance
            text = exchange.replace('Student:', '').strip()
            metadata['full_conversation'].append(('Student', text, None))
    
    return utterances, metadata

def load_training_data(csv_path: str, sample_size: int = None):
    """Load and prepare MathDial training data for DSPy"""
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.head(sample_size)
    
    print(f"Loading {len(df)} training samples...")
    
    examples = []
    
    for _, row in df.iterrows():
        # Parse conversation - handle both 'conversation' and 'cleaned_conversation' columns
        conversation_col = 'conversation' if 'conversation' in row else 'cleaned_conversation'
        
        if conversation_col not in row or pd.isna(row[conversation_col]):
            continue
            
        teacher_utterances, metadata = parse_conversation(row[conversation_col])
        
        # Get problem context
        problem = row.get('question', '')
        student_solution = row.get('student_incorrect_solution', '')
        
        # Create examples for each teacher utterance
        conversation_so_far = ""
        
        for i, utterance in enumerate(teacher_utterances):
            if utterance.label:  # Only use utterances with labels for training
                example = dspy.Example(
                    problem=problem,
                    student_solution=student_solution,
                    conversation_history=conversation_so_far,
                    teacher_utterance=utterance.text,
                    move_type=utterance.label,
                    reasoning=f"This is a {utterance.label} move in the tutoring dialogue"
                ).with_inputs('problem', 'student_solution', 'conversation_history', 'teacher_utterance')
                
                examples.append(example)
            
            # Update conversation history
            conversation_so_far += f"Teacher: {utterance.text}\n"
            
            # Add student response if available
            for j, (speaker, text, _) in enumerate(metadata['full_conversation'][i*2+1:i*2+2]):
                if speaker == 'Student':
                    conversation_so_far += f"Student: {text}\n"
                    break
    
    return examples

def mathdial_metric(example, pred, trace=None):
    """Metric for MathDial teacher move classification"""
    try:
        true_move = example.move_type.lower().strip()
        pred_move = pred.move_type.lower().strip()
        
        # Normalize the move types
        if pred_move in TEACHER_MOVES:
            return true_move == pred_move
        
        # Try to match partial strings
        for move in TEACHER_MOVES:
            if move in pred_move:
                return true_move == move
                
        return False
    except:
        return False

def train_dspy_module(training_examples, sample_size):
    """Train a DSPy module for MathDial with GPT-5"""
    print(f"\nTraining DSPy MathDial module with {sample_size} samples using GPT-5...")
    print("GPT-5 reasoning model will learn teacher move patterns from tutoring dialogues...")
    
    # Use subset of training examples
    train_set = training_examples[:sample_size]
    
    # Initialize module
    mathdial_module = MathDialClassifier()
    
    # Configure optimizer - adjusted for GPT-5
    from dspy.teleprompt import BootstrapFewShot
    
    teleprompter = BootstrapFewShot(
        metric=mathdial_metric,
        max_bootstrapped_demos=3,  # Reduced for GPT-5
        max_labeled_demos=4,        # Reduced for GPT-5
        max_rounds=1                # Single round for GPT-5
    )
    
    # Compile the module
    print(f"Optimizing module with {len(train_set)} examples using GPT-5...")
    start_time = time.time()
    
    try:
        optimized_module = teleprompter.compile(mathdial_module, trainset=train_set)
        elapsed = time.time() - start_time
        print(f"GPT-5 training completed in {elapsed:.1f} seconds")
        
        if elapsed < 10:
            print("Note: Training used cached responses for speed")
        else:
            print("Training with fresh GPT-5 API calls completed")
    except Exception as e:
        print(f"Training error: {e}")
        print("Returning base module without optimization")
        return mathdial_module
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save what DSPy learned
    try:
        learned_file = f'results/mathdial_gpt5_module_{sample_size}_learned.json'
        
        module_str = str(optimized_module)
        
        learned_data = {
            'task': 'MathDial Teacher Move Classification',
            'model': 'GPT-5',
            'sample_size': sample_size,
            'training_time': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'module_structure': {
                'type': 'ChainOfThought',
                'signature': 'MathDialSignature',
                'inputs': ['problem', 'student_solution', 'conversation_history', 'teacher_utterance'],
                'outputs': ['move_type', 'reasoning'],
                'num_categories': 4,
                'categories': TEACHER_MOVES,
                'reasoning': 'Chain-of-thought reasoning with GPT-5'
            },
            'optimization': {
                'method': 'BootstrapFewShot',
                'max_demos': 3,
                'max_labeled': 4,
                'optimization_rounds': 1,
                'examples_evaluated': min(sample_size, 10)
            },
            'model_settings': {
                'model': 'gpt-5',
                'temperature': 1.0,
                'max_tokens': 20000,
                'note': 'Required settings for DSPy reasoning models, litellm.drop_params=True to handle variations'
            },
            'note': 'DSPy v3 with GPT-5 reasoning model for teacher move classification in math tutoring',
            'module_string_preview': module_str[:500] if len(module_str) > 500 else module_str
        }
        
        with open(learned_file, 'w') as f:
            json.dump(learned_data, f, indent=2)
        
        print(f"GPT-5 module configuration saved to {learned_file}")
        
    except Exception as e:
        print(f"Note: Could not save module details: {e}")
    
    print(f"GPT-5 module trained with {sample_size} samples")
    return optimized_module

def test_dspy_module(module, test_csv_path, module_name):
    """Test DSPy module on MathDial holdout set with GPT-5"""
    
    print(f"\nTesting {module_name} on holdout set with GPT-5...")
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    results = []
    
    start_time = time.time()
    api_calls = 0
    
    for idx, row in test_df.iterrows():
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {idx + 1}/{len(test_df)} - Time: {elapsed:.1f}s - GPT-5")
        
        # Parse conversation - handle both 'conversation' and 'cleaned_conversation' columns
        conversation_col = 'conversation' if 'conversation' in row else 'cleaned_conversation'
        
        if conversation_col not in row or pd.isna(row[conversation_col]):
            continue
            
        teacher_utterances, metadata = parse_conversation(row[conversation_col])
        
        # Get problem context
        problem = row.get('question', '')
        student_solution = row.get('student_incorrect_solution', '')
        
        # Test each teacher utterance
        conversation_so_far = ""
        
        for i, utterance in enumerate(teacher_utterances):
            if not utterance.label:  # Skip if no ground truth
                continue
                
            try:
                # Get DSPy prediction with GPT-5
                prediction = module(
                    problem=problem,
                    student_solution=student_solution,
                    conversation_history=conversation_so_far,
                    teacher_utterance=utterance.text
                )
                api_calls += 1
                
                # Extract predicted move
                pred_move = prediction.move_type.lower().strip()
                
                # Normalize prediction
                for move in TEACHER_MOVES:
                    if move in pred_move:
                        pred_move = move
                        break
                
                if pred_move not in TEACHER_MOVES:
                    pred_move = 'generic'  # Default fallback
                
                # Check if correct
                is_correct = pred_move == utterance.label
                
                results.append({
                    'conversation_id': idx,
                    'utterance_num': i + 1,
                    'teacher_utterance': utterance.text[:100],  # First 100 chars
                    'true_move': utterance.label,
                    'predicted_move': pred_move,
                    'correct': is_correct,
                    'reasoning': str(prediction.reasoning)[:200] if hasattr(prediction, 'reasoning') else '',
                    'Technique': module_name,
                    'Model': 'GPT-5'
                })
                
            except Exception as e:
                print(f"Error predicting utterance {i} in conversation {idx}: {e}")
                results.append({
                    'conversation_id': idx,
                    'utterance_num': i + 1,
                    'teacher_utterance': utterance.text[:100],
                    'true_move': utterance.label,
                    'predicted_move': 'error',
                    'correct': False,
                    'reasoning': f'Error: {str(e)[:100]}',
                    'Technique': module_name,
                    'Model': 'GPT-5'
                })
            
            # Update conversation history
            conversation_so_far += f"Teacher: {utterance.text}\n"
            
            # Add student response if available  
            for j, (speaker, text, _) in enumerate(metadata['full_conversation'][i*2+1:i*2+2]):
                if speaker == 'Student':
                    conversation_so_far += f"Student: {text}\n"
                    break
            
            # Rate limiting for GPT-5
            time.sleep(1.0)  # Increased for GPT-5
    
    total_time = time.time() - start_time
    print(f"GPT-5 testing completed in {total_time:.1f} seconds ({api_calls} API calls)")
    
    return pd.DataFrame(results)

def calculate_mathdial_metrics(results_df):
    """Calculate metrics specific to MathDial classification"""
    
    if results_df.empty:
        print("Warning: Empty results DataFrame")
        return {
            'overall_accuracy': 0.0,
            'per_move_accuracy': {move: (0.0, 0) for move in TEACHER_MOVES},
            'confusion_matrix': {},
            'scaffolding_ratio': 0.0,
            'telling_rate': 0.0,
            'total_utterances': 0,
            'unique_conversations': 0
        }
    
    # Overall accuracy
    overall_accuracy = results_df['correct'].mean() * 100
    
    # Per-move accuracy
    move_accuracies = {}
    for move in TEACHER_MOVES:
        move_data = results_df[results_df['true_move'] == move]
        if len(move_data) > 0:
            move_accuracies[move] = (move_data['correct'].mean() * 100, len(move_data))
        else:
            move_accuracies[move] = (0.0, 0)
    
    # Confusion matrix
    try:
        confusion = pd.crosstab(
            results_df['true_move'], 
            results_df['predicted_move'],
            margins=True
        )
        confusion_dict = confusion.to_dict()
    except:
        confusion_dict = {}
    
    # Calculate scaffolding ratio (focus + probing)
    scaffolding_moves = results_df[results_df['predicted_move'].isin(['focus', 'probing'])]
    scaffolding_ratio = len(scaffolding_moves) / len(results_df) * 100 if len(results_df) > 0 else 0.0
    
    # Calculate telling rate
    telling_moves = results_df[results_df['predicted_move'] == 'telling']
    telling_rate = len(telling_moves) / len(results_df) * 100 if len(results_df) > 0 else 0.0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'per_move_accuracy': move_accuracies,
        'confusion_matrix': confusion_dict,
        'scaffolding_ratio': scaffolding_ratio,
        'telling_rate': telling_rate,
        'total_utterances': len(results_df),
        'unique_conversations': results_df['conversation_id'].nunique() if not results_df.empty else 0
    }
    
    return metrics

def test_api_connection():
    """Test if GPT-5 API is working properly with MathDial"""
    print("\n" + "="*60)
    print("TESTING GPT-5 API CONNECTION FOR MATHDIAL")
    print("="*60)
    
    test_utterance = "Let's think about what 'twice' means in this problem."
    test_problem = "Julia has twice as many apples as Tom. Tom has 5 apples."
    test_solution = "Julia has 2 apples because twice means divide by 2."
    
    print(f"Test utterance: {test_utterance}")
    print(f"Problem context: {test_problem}")
    
    try:
        classifier = MathDialClassifier()
        start = time.time()
        result = classifier(
            problem=test_problem,
            student_solution=test_solution,
            conversation_history="",
            teacher_utterance=test_utterance
        )
        elapsed = time.time() - start
        
        print(f"GPT-5 API call took: {elapsed:.2f} seconds")
        print(f"\nClassification result:")
        print(f"  Move type: {result.move_type}")
        print(f"  Reasoning: {result.reasoning}")
        
        if elapsed < 0.5:
            print("Note: Using cached responses for speed")
        else:
            print("SUCCESS: Fresh GPT-5 API call completed!")
            print("Note: GPT-5 is a reasoning model, so responses may take longer")
            
    except Exception as e:
        print(f"ERROR: GPT-5 API test failed - {e}")
        print("Check your API key and GPT-5 access in config.py")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    test_api_connection()