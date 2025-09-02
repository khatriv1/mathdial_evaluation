# mathdial_evaluation/prompting/active_prompt.py

"""
Active Prompting for MathDial teacher move classification.
FIXED: New approach - 8 examples (1 uncertain + 1 wrong per category) WITHOUT self-consistency
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
import pandas as pd
from utils.mathdial_rubric import MathDialRubric

# Keep existing ActivePromptSelector class as-is...
class ActivePromptSelector:
    # ... existing code stays the same ...
    pass

# Keep existing prepare_active_prompting_data function as-is...
def prepare_active_prompting_data(df: pd.DataFrame, client) -> List[Tuple[str, str]]:
    # ... existing code stays the same ...
    pass

def get_active_prompt_prediction(teacher_utterance: str, conversation_context: str,
                                client,
                                question: str = None,
                                student_solution: str = None, 
                                student_profile: str = None,
                                uncertainty_data: List[Tuple[str, str]] = None) -> str:
    """Get active prompting prediction WITHOUT self-consistency"""
    
    rubric = MathDialRubric()
    
    # Build examples text from all 8 examples
    examples_text = ""
    if uncertainty_data:
        examples_text = "EXAMPLES:\n"
        for i, (utterance, classification) in enumerate(uncertainty_data, 1):
            examples_text += f'Example {i}:\n'
            examples_text += f'Teacher: "{utterance[:100]}..."\n'
            examples_text += f'{classification}\n\n'
    
    # ADD THE MISSING CONTEXT
    context_info = ""
    if question:
        context_info += f"Math Problem: {question}\n\n"
    if student_solution:
        context_info += f"Student's Incorrect Solution: {student_solution}\n\n"
    if student_profile:
        context_info += f"Student Profile: {student_profile}\n\n"
    
    # Build prompt with Auto-CoT addition
    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
GENERIC: {rubric.get_move_definition('generic')}
FOCUS: {rubric.get_move_definition('focus')}
PROBING: {rubric.get_move_definition('probing')}
TELLING: {rubric.get_move_definition('telling')}

TASK: Now analyze the following teacher utterance and classify it.

INSTRUCTIONS:
1. Learn from the examples provided above
2. Analyze the teacher utterance carefully
3. Consider the pedagogical intent
4. Respond with only one word: generic, focus, probing, or telling

{examples_text}

{context_info}

Now work on this one, please think step by step:

Conversation Context:
{conversation_context}

Teacher: "{teacher_utterance}"

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying teacher moves. Think step by step and provide the category."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Deterministic for final prediction
            max_tokens=400,
            timeout=15
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract classification
        categories = ['generic', 'focus', 'probing', 'telling']
        for cat in categories:
            if cat in result:
                return cat
        
        return 'generic'
                
    except Exception as e:
        print(f"Error in active prompting: {e}")
        return 'generic'

# Function to load active prompting pool
def load_active_prompting_pool(pool_path: str = "data/active_prompting_pool_20.csv") -> pd.DataFrame:
    """Load the active prompting pool data"""
    return pd.read_csv(pool_path)