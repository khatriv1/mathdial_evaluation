# mathdial_evaluation/prompting/auto_cot.py

"""
Auto-CoT (Automatic Chain of Thought) prompting for MathDial teacher move classification.
FIXED: Properly handles context for each utterance classification with step-by-step reasoning
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import json
import re
from typing import List, Optional, Dict, Tuple
from utils.mathdial_rubric import MathDialRubric

def get_auto_cot_prediction(teacher_utterance: str, conversation_context: str, client,
                           question: str = None, student_solution: str = None,
                           student_profile: str = None, utterance_num: int = None) -> str:
    """
    Auto-CoT: Zero-Shot + "Let's think step by step" for a SINGLE utterance.
    
    IMPORTANT: This function classifies ONE teacher utterance at a time,
    but uses the FULL conversation context up to that point for reasoning.
    
    Args:
        teacher_utterance: The SPECIFIC teacher's utterance to classify
        conversation_context: ALL previous conversation up to this utterance
        client: OpenAI client
        question: The math problem being discussed
        student_solution: The student's incorrect solution
        student_profile: The student's profile/characteristics
        utterance_num: Which teacher utterance this is (for debugging)
    
    Returns:
        Classification for THIS SPECIFIC utterance: 'generic', 'focus', 'probing', or 'telling'
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    rubric = MathDialRubric()
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {rubric.get_move_definition(cat)}\n"

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
        print(f"\n  [DEBUG Auto-CoT] Classifying Teacher Utterance #{utterance_num}:")
        print(f"    Utterance: \"{teacher_utterance[:60]}...\"")
        print(f"    Context length: {len(conversation_context.split())} words")
        print(f"    Using chain-of-thought reasoning")

    # EXACTLY like Zero-Shot but with "Let's think step by step" added
    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
{definitions_text}

TASK: Analyze ONE SPECIFIC teacher utterance and classify it into one of the four categories above.
You have the full conversation context to understand the situation, but you must classify ONLY the specified utterance.

{context_info}

CONVERSATION CONTEXT (everything that happened before this utterance):
{conversation_context}

CURRENT TEACHER UTTERANCE TO CLASSIFY (classify ONLY this):
Teacher: "{teacher_utterance}"

Let's think step by step.

REASONING STEPS:
1. What is happening in the conversation at this point?
2. What is the teacher trying to accomplish with THIS specific utterance?
3. What pedagogical intent does THIS utterance serve?
4. Which category best matches THIS utterance's purpose?

After your reasoning, provide your final classification.
Respond with only one word at the end: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying individual teacher utterances. Think step by step about ONLY the specified utterance, then provide the category."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0,
            # max_tokens=400
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