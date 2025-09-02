# mathdial_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for MathDial teacher move classification.
FIXED: Properly handles context for each utterance with multiple reasoning paths
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import json
import re
import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import Counter
from utils.mathdial_rubric import MathDialRubric

def get_single_reasoning_path(teacher_utterance: str,
                            conversation_context: str,
                            client,
                            question: str = None,
                            student_solution: str = None,
                            student_profile: str = None,
                            temperature: float = 0.7,
                            path_num: int = None) -> Optional[str]:
    """
    Get a single reasoning path for classifying ONE SPECIFIC utterance.
    NOW WITH: Few-shot examples + Auto-CoT
    
    Args:
        teacher_utterance: The SPECIFIC teacher's utterance to classify
        conversation_context: ALL previous conversation up to this utterance
        client: OpenAI client
        question: The math problem being discussed
        student_solution: The student's incorrect solution
        student_profile: The student's profile/characteristics
        temperature: Sampling temperature for diversity
        path_num: Which reasoning path this is (for debugging)
    
    Returns:
        Classification for THIS SPECIFIC utterance or None if failed
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    rubric = MathDialRubric()
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {rubric.get_move_definition(cat)}\n"

    # FEW-SHOT EXAMPLES (same as few_shot.py)
    examples_text = """EXAMPLES:
Teacher: "Hi ..., how are you doing with the word problem?"
Classification: generic

Teacher: "Good Job! Is there anything else I can help with?"
Classification: generic

Teacher: "So what should you do next?"
Classification: focus

Teacher: "Can you calculate ... ?"
Classification: focus

Teacher: "Why do you think you need to add these numbers?"
Classification: probing

Teacher: "Are you sure you need to add here?"
Classification: probing

Teacher: "You need to add ... to ... to get your answer."
Classification: telling

Teacher: "No, he had ... items."
Classification: telling

"""

    # BUILD FULL CONTEXT - This is critical!
    context_info = ""
    if question:
        context_info += f"Math Problem: {question}\n\n"
    if student_solution:
        context_info += f"Student's Incorrect Solution: {student_solution}\n\n"
    if student_profile:
        context_info += f"Student Profile: {student_profile}\n\n"

    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
{definitions_text}

{examples_text}

TASK: Classify ONE SPECIFIC teacher utterance using the examples and reasoning.
You have the full conversation context, but you must classify ONLY the specified utterance.

{context_info}

CONVERSATION CONTEXT (everything that happened before this utterance):
{conversation_context}

CURRENT TEACHER UTTERANCE TO CLASSIFY (classify ONLY this):
Teacher: "{teacher_utterance}"

Let's think step by step.

Think through this step-by-step and provide your classification:
1. What is the main purpose of THIS SPECIFIC teacher utterance?
2. What pedagogical intent does THIS utterance serve?
3. Which category best fits THIS utterance?

Provide your classification: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying individual teacher utterances. Learn from examples, think step by step about ONLY the specified utterance, and provide the category."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=400  # Increased for reasoning
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse classification
        classification = parse_classification(result, categories)
        return classification
            
    except Exception as e:
        if path_num:
            print(f"    Error in reasoning path {path_num}: {str(e)}")
        return None

def get_self_consistency_prediction(teacher_utterance: str,
                                   conversation_context: str,
                                   client,
                                   question: str = None,
                                   student_solution: str = None,
                                   student_profile: str = None,
                                   n_samples: int = 5,
                                   utterance_num: int = None) -> str:
    """
    Get Self-Consistency prediction for ONE SPECIFIC utterance using multiple reasoning paths.
    
    IMPORTANT: This generates multiple classifications for the SAME utterance
    and takes a majority vote for robustness.
    
    Args:
        teacher_utterance: The SPECIFIC teacher's utterance to classify
        conversation_context: ALL previous conversation up to this utterance
        client: OpenAI client
        question: The math problem being discussed
        student_solution: The student's incorrect solution
        student_profile: The student's profile/characteristics
        n_samples: Number of reasoning paths to sample
        utterance_num: Which teacher utterance this is (for debugging)
    
    Returns:
        Classification for THIS SPECIFIC utterance based on majority vote
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    # Debug output for verification
    if utterance_num is not None and utterance_num <= 2:  # Show for first 2 utterances
        print(f"\n  [DEBUG Self-Consistency] Classifying Teacher Utterance #{utterance_num}:")
        print(f"    Utterance: \"{teacher_utterance[:60]}...\"")
        print(f"    Sampling {n_samples} reasoning paths for this utterance")
    
    # Collect predictions from multiple reasoning paths
    all_predictions = []
    
    for i in range(n_samples):
        # Fixed temperature for all samples (same as original)
        temp = 0.7
        
        prediction = get_single_reasoning_path(
            teacher_utterance, conversation_context, client,
            question, student_solution, student_profile, temp,
            path_num=i+1
        )
        
        if prediction is not None:
            all_predictions.append(prediction)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not all_predictions:
        if utterance_num:
            print(f"    No valid predictions for utterance {utterance_num}")
        return 'generic'
    
    # Take majority vote
    vote_counts = Counter(all_predictions)
    majority_vote = vote_counts.most_common(1)[0][0]
    
    # Print voting details if there was disagreement (only for first few utterances)
    if utterance_num and utterance_num <= 2 and len(set(all_predictions)) > 1:
        print(f"    Votes: {dict(vote_counts)} → majority={majority_vote}")
    
    return majority_vote

def parse_classification(response_text: str, categories: list) -> str:
    """Parse classification from model response."""
    response_lower = response_text.strip().lower()
    
    # Try to find classification in the response
    for cat in categories:
        if cat in response_lower:
            return cat
    
    return 'generic'

# Alias for backward compatibility
def get_self_consistency_classification(teacher_utterance: str, client, 
                                       conversation_context: str = "", 
                                       n_samples: int = 5) -> str:
    """Alias for backward compatibility."""
    return get_self_consistency_prediction(teacher_utterance, conversation_context, client, n_samples=n_samples)