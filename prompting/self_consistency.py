# mathdial_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for MathDial teacher move classification.
Samples multiple reasoning paths and takes majority vote.
NOW INCLUDES: Few-shot examples + Auto-CoT
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import json
import re
import numpy as np
from typing import List, Optional, Dict
from collections import Counter
from utils.mathdial_rubric import MathDialRubric

def get_single_reasoning_path(teacher_utterance: str,
                            conversation_context: str,
                            client,
                            temperature: float = 0.7) -> Optional[str]:
    """
    Get a single reasoning path for classification.
    NOW WITH: Few-shot examples + Auto-CoT
    
    Args:
        teacher_utterance: Teacher's utterance
        conversation_context: Context
        client: OpenAI client
        temperature: Sampling temperature for diversity
    
    Returns:
        Classification or None if failed
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

    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
{definitions_text}

{examples_text}

Now work on this one:

Conversation Context:
{conversation_context}

Teacher: "{teacher_utterance}"

Let's think step by step.

Think through this step-by-step and provide your classification:
1. What is the main purpose of this teacher utterance?
2. What pedagogical intent does it serve?
3. Which category best fits?

Provide your classification: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying teacher moves. Learn from examples, think step by step, and provide the category."},
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
        print(f"Error in reasoning path: {str(e)}")
        return None

def get_self_consistency_prediction(teacher_utterance: str,
                                   conversation_context: str,
                                   client,
                                   n_samples: int = 5) -> str:
    """
    Get Self-Consistency prediction using multiple reasoning paths.
    NOW WITH: Few-shot examples + Auto-CoT in each attempt
    
    Args:
        teacher_utterance: Teacher's utterance
        conversation_context: Context
        client: OpenAI client
        n_samples: Number of reasoning paths to sample
    
    Returns:
        Classification based on majority vote
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    # Collect predictions from multiple reasoning paths
    all_predictions = []
    
    for i in range(n_samples):
        # Fixed temperature for all samples (same as Bloom)
        temp = 0.7
        
        prediction = get_single_reasoning_path(teacher_utterance, conversation_context, client, temp)
        
        if prediction is not None:
            all_predictions.append(prediction)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not all_predictions:
        print(f"No valid predictions obtained for self-consistency")
        return 'generic'
    
    # Take majority vote
    vote_counts = Counter(all_predictions)
    majority_vote = vote_counts.most_common(1)[0][0]
    
    # Print voting details if there was disagreement
    if len(set(all_predictions)) > 1:
        print(f"  Votes: {dict(vote_counts)} → majority={majority_vote}")
    
    print(f"Self-consistency: Used {len(all_predictions)} predictions")
    
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
    return get_self_consistency_prediction(teacher_utterance, conversation_context, client, n_samples)