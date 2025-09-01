# mathdial_evaluation/prompting/auto_cot.py

"""
Auto-CoT (Automatic Chain of Thought) prompting for MathDial teacher move classification.
Just Zero-Shot + "Let's think step by step"
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import json
import re
from typing import List, Optional, Dict
from utils.mathdial_rubric import MathDialRubric

def get_auto_cot_prediction(teacher_utterance: str, conversation_context: str, client) -> str:
    """
    Auto-CoT: Just Zero-Shot + "Let's think step by step"
    
    Args:
        teacher_utterance: The teacher's utterance to classify
        conversation_context: Previous conversation context
        client: OpenAI client
    
    Returns:
        One of: 'generic', 'focus', 'probing', 'telling'
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    rubric = MathDialRubric()
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {rubric.get_move_definition(cat)}\n"

    # EXACTLY like Zero-Shot but with "Let's think step by step" added
    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
{definitions_text}

TASK: Analyze the following teacher utterance and classify it into one of the four categories above.

Now work on this one:

Conversation Context:
{conversation_context}

Teacher Utterance: "{teacher_utterance}"

Let's think step by step.

INSTRUCTIONS:
1. Analyze the teacher utterance carefully
2. Consider the pedagogical intent of the teacher
3. Match it to one of the four categories
4. Respond with only one word: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying teacher moves in math tutoring. Think step by step and provide the category."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=400
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse classification
        classification = parse_classification(result, categories)
        
        return classification
        
    except Exception as e:
        print(f"Error in auto-CoT classification: {str(e)}")
        return 'generic'

def parse_classification(response_text: str, categories: list) -> str:
    """Parse classification from model response."""
    response_lower = response_text.strip().lower()
    
    # Extract the final classification
    for cat in categories:
        if cat in response_lower:
            return cat
    
    return 'generic'

# Legacy function
def get_auto_cot_classification(teacher_utterance: str, client, conversation_context: str = "") -> str:
    """Legacy function."""
    return get_auto_cot_prediction(teacher_utterance, conversation_context, client)