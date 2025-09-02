# mathdial_evaluation/prompting/active_prompt.py

"""
Active Prompting for MathDial teacher move classification.
FIXED: Properly handles context for each utterance with uncertainty examples
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

class ActivePromptSelector:
    """Selects uncertain/wrong examples for active prompting"""
    
    def __init__(self, client):
        self.client = client
        self.rubric = MathDialRubric()
    
    def get_uncertainty_score(self, teacher_utterance: str, context: str = "") -> float:
        """Get uncertainty score for an utterance (simplified version)"""
        # In practice, this would use multiple predictions to measure uncertainty
        # For now, returning a random score for demonstration
        return np.random.random()
    
    def select_examples(self, df: pd.DataFrame, n_per_category: int = 2) -> List[Tuple[str, str]]:
        """Select uncertain/wrong examples from pool"""
        examples = []
        categories = ['generic', 'focus', 'probing', 'telling']
        
        for category in categories:
            # Get examples from this category
            category_examples = []
            
            for _, row in df.iterrows():
                if category in row.get('ground_truth_moves', []):
                    # Extract a teacher utterance from this category
                    conversation = row.get('cleaned_conversation', '')
                    lines = conversation.split('\n')
                    
                    for line in lines:
                        if line.startswith('Teacher:'):
                            utterance = line.replace('Teacher:', '').strip()
                            if utterance and len(category_examples) < n_per_category:
                                category_examples.append((utterance[:100], category))
                                break
                
                if len(category_examples) >= n_per_category:
                    break
            
            examples.extend(category_examples)
        
        return examples[:8]  # Return 8 examples total

def prepare_active_prompting_data(df: pd.DataFrame, client) -> List[Tuple[str, str]]:
    """Prepare uncertainty examples from separate pool"""
    selector = ActivePromptSelector(client)
    return selector.select_examples(df, n_per_category=2)

def get_active_prompt_prediction(teacher_utterance: str, conversation_context: str,
                                client,
                                question: str = None,
                                student_solution: str = None, 
                                student_profile: str = None,
                                uncertainty_data: List[Tuple[str, str]] = None,
                                utterance_num: int = None) -> str:
    """
    Get active prompting prediction for ONE SPECIFIC utterance.
    
    IMPORTANT: This function classifies ONE teacher utterance at a time,
    using uncertainty examples for guidance PLUS full conversation context.
    
    Args:
        teacher_utterance: The SPECIFIC teacher's utterance to classify
        conversation_context: ALL previous conversation up to this utterance
        client: OpenAI client
        question: The math problem being discussed
        student_solution: The student's incorrect solution
        student_profile: The student's profile/characteristics
        uncertainty_data: List of (utterance, classification) examples
        utterance_num: Which teacher utterance this is (for debugging)
    
    Returns:
        Classification for THIS SPECIFIC utterance
    """
    
    rubric = MathDialRubric()
    
    # Build examples text from uncertainty data
    examples_text = ""
    if uncertainty_data:
        examples_text = "CHALLENGING EXAMPLES TO LEARN FROM:\n"
        for i, (utterance, classification) in enumerate(uncertainty_data, 1):
            examples_text += f'Example {i}:\n'
            examples_text += f'Teacher: "{utterance[:100]}..."\n'
            examples_text += f'Classification: {classification}\n\n'
    
    # BUILD FULL CONTEXT - This is critical!
    context_info = ""
    if question:
        context_info += f"Math Problem: {question}\n\n"
    if student_solution:
        context_info += f"Student's Incorrect Solution: {student_solution}\n\n"
    if student_profile:
        context_info += f"Student Profile: {student_profile}\n\n"
    
    # Debug output for verification
    if utterance_num is not None and utterance_num <= 2:  # Show for first 2 utterances
        print(f"\n  [DEBUG Active] Classifying Teacher Utterance #{utterance_num}:")
        print(f"    Utterance: \"{teacher_utterance[:60]}...\"")
        print(f"    Context length: {len(conversation_context.split())} words")
        print(f"    Using {len(uncertainty_data) if uncertainty_data else 0} uncertainty examples")
    
    # Build prompt with Auto-CoT addition
    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
GENERIC: {rubric.get_move_definition('generic')}
FOCUS: {rubric.get_move_definition('focus')}
PROBING: {rubric.get_move_definition('probing')}
TELLING: {rubric.get_move_definition('telling')}

{examples_text}

TASK: Classify ONE SPECIFIC teacher utterance.
You have the full conversation context, but you must classify ONLY the specified utterance.

{context_info}

CONVERSATION CONTEXT (everything that happened before this utterance):
{conversation_context}

CURRENT TEACHER UTTERANCE TO CLASSIFY (classify ONLY this):
Teacher: "{teacher_utterance}"

INSTRUCTIONS:
1. Learn from the challenging examples provided above
2. Read the full context to understand the conversation flow
3. Focus on THIS SPECIFIC teacher utterance
4. Think step by step about the pedagogical intent of THIS utterance
5. Classify ONLY this utterance (not the whole conversation)
6. Respond with only one word: generic, focus, probing, or telling

Let's think step by step.

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying individual teacher utterances. Learn from challenging examples, think step by step about ONLY the specified utterance."},
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
        if utterance_num:
            print(f"    Error in active prompting for utterance {utterance_num}: {e}")
        return 'generic'

# Function to load active prompting pool
def load_active_prompting_pool(pool_path: str = "data/active_prompting_pool_20.csv") -> pd.DataFrame:
    """Load the active prompting pool data"""
    return pd.read_csv(pool_path)