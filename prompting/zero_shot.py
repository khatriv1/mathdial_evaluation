# mathdial_evaluation/prompting/zero_shot.py

"""
Zero-shot prompting for MathDial teacher move classification.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import json
import re
from typing import Optional, Dict, List, Tuple
from utils.mathdial_rubric import MathDialRubric

def parse_classification(response_text: str, categories: list) -> str:
    """
    Parse teacher move classification from model response.
    
    Args:
        response_text: Raw response from the model
        categories: List of category names
        
    Returns:
        The classified teacher move
    """
    response_lower = response_text.strip().lower()
    
    # Try direct match
    for cat in categories:
        if cat in response_lower:
            return cat
    
    # Default fallback
    return 'generic'

def get_zero_shot_prediction(teacher_utterance: str, conversation_context: str, client,
                            question: str = None, student_solution: str = None,
                            student_profile: str = None) -> str:
    """
    Get zero-shot prediction for teacher move classification.
    
    Args:
        teacher_utterance: The teacher's utterance to classify
        conversation_context: Previous conversation context
        client: OpenAI client
        question: The math problem being discussed
        student_solution: The student's incorrect solution
        student_profile: The student's profile/characteristics
    
    Returns:
        One of: 'generic', 'focus', 'probing', 'telling'
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    rubric = MathDialRubric()
    
    # Create comprehensive prompt for classification
    definitions_text = ""
    for cat in categories:
        definition = rubric.get_move_definition(cat)
        definitions_text += f"\n{cat.upper()}: {definition}\n"
        
        # Add intent examples from rubric
        intents = rubric.get_intent_examples(cat)
        if intents:
            for intent, example in intents.items():
                definitions_text += f"  - {intent}: {example}\n"
    
    # ADD THE MISSING CONTEXT
    context_info = ""
    if question:
        context_info += f"Math Problem: {question}\n\n"
    if student_solution:
        context_info += f"Student's Incorrect Solution: {student_solution}\n\n"
    if student_profile:
        context_info += f"Student Profile: {student_profile}\n\n"
    
    prompt = f"""Your expertise lies in categorizing teacher moves in math tutoring based on pedagogical intent.

TEACHER MOVE CATEGORIES:
{definitions_text}

TASK: Analyze the following teacher utterance and classify it into one of the four categories above.

{context_info}

Now work on this one:

Conversation Context:
{conversation_context}

Teacher Utterance: "{teacher_utterance}"

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
                {"role": "system", "content": "You are an expert at classifying teacher moves in math tutoring. Respond with only one category."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the classification
        classification = parse_classification(result, categories)
        
        return classification
        
    except Exception as e:
        print(f"Error in zero-shot classification: {str(e)}")
        # Return default
        return 'generic'

def analyze_conversation_moves(conversation: str, client, question: str = None,
                              student_solution: str = None, student_profile: str = None) -> List[Tuple[str, str]]:
    """
    Analyze all teacher moves in a complete conversation.
    """
    results = []
    
    # Parse conversation into turns
    lines = conversation.split('\n')
    context = ""
    
    for line in lines:
        if line.startswith('Teacher:'):
            # Extract teacher utterance
            utterance = line.replace('Teacher:', '').strip()
            
            # Remove any move labels if present (from ground truth)
            utterance = re.sub(r'\([^)]+\)', '', utterance)
            utterance = utterance.replace('|EOM|', '').strip()
            
            if utterance:
                # Classify this utterance WITH FULL CONTEXT
                classification = get_zero_shot_prediction(
                    utterance, context, client,
                    question=question,
                    student_solution=student_solution,
                    student_profile=student_profile
                )
                results.append((utterance, classification))
            
            # Add to context
            context += f"Teacher: {utterance}\n"
            
        elif line.startswith('Student:'):
            # Add student turn to context
            student_utterance = line.replace('Student:', '').strip()
            student_utterance = student_utterance.replace('|EOM|', '').strip()
            context += f"Student: {student_utterance}\n"
    
    return results

# Legacy function for backwards compatibility
def get_zero_shot_classification(teacher_utterance: str, client, conversation_context: str = "") -> str:
    """
    Legacy function for single utterance classification.
    """
    return get_zero_shot_prediction(teacher_utterance, conversation_context, client)