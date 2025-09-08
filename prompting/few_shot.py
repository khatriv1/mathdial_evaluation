# mathdial_evaluation/prompting/few_shot.py

"""
Few-shot prompting for MathDial teacher move classification.
FIXED: Properly handles context for each utterance classification
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

def get_few_shot_examples() -> str:
    """Get few-shot examples from MathDial Table 2."""
    examples = """
EXAMPLES:

Teacher: "Hi ..., how are you doing with the word problem?"
Classification: generic

Teacher: "Good Job! Is there anything else I can help with?"
Classification: generic

Teacher: "Can you go walk me through your solution?"
Classification: generic

Teacher: "So what should you do next?"
Classification: focus

Teacher: "Can you calculate ... ?"
Classification: focus

Teacher: "Can you reread the question and tell me what is ... ?"
Classification: focus

Teacher: "Why do you think you need to add these numbers?"
Classification: probing

Teacher: "Are you sure you need to add here?"
Classification: probing

Teacher: "How would things change if they had ... items instead?"
Classification: probing

Teacher: "How do you calculate the perimeter of a square?"
Classification: probing

Teacher: "You need to add ... to ... to get your answer."
Classification: telling

Teacher: "No, he had ... items."
Classification: telling
"""
    return examples

def get_few_shot_prediction(teacher_utterance: str, conversation_context: str, client,
                           question: str = None, student_solution: str = None,
                           student_profile: str = None, utterance_num: int = None) -> str:
    """
    Get few-shot prediction for a SINGLE teacher utterance with full context.
    
    IMPORTANT: This function classifies ONE teacher utterance at a time,
    but uses the FULL conversation context up to that point PLUS examples.
    
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
    
    examples_text = get_few_shot_examples()
    
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
        print(f"\n  [DEBUG Few-shot] Classifying Teacher Utterance #{utterance_num}:")
        print(f"    Utterance: \"{teacher_utterance[:60]}...\"")
        print(f"    Context length: {len(conversation_context.split())} words")
        print(f"    Using 12 examples for guidance")

    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
{definitions_text}

{examples_text}

TASK: Learn from the examples above, then classify ONE SPECIFIC teacher utterance.
You have the full conversation context to understand the situation, but you must classify ONLY the specified utterance.

{context_info}

CONVERSATION CONTEXT (everything that happened before this utterance):
{conversation_context}

CURRENT TEACHER UTTERANCE TO CLASSIFY (classify ONLY this):
Teacher: "{teacher_utterance}"

INSTRUCTIONS:
1. Learn from the 12 examples provided above
2. Read the full context to understand the conversation flow
3. Focus on THIS SPECIFIC teacher utterance
4. Analyze the pedagogical intent of THIS utterance
5. Classify ONLY this utterance (not the whole conversation)
6. Respond with only one word: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying individual teacher utterances. Learn from examples, then classify ONLY the specified utterance."},
                {"role": "user", "content": prompt}
            ],
            #temperature=0,
            #max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse classification
        classification = parse_classification(result, categories)
        
        return classification
        
    except Exception as e:
        print(f"Error in few-shot classification: {str(e)}")
        return 'generic'

def parse_classification(response_text: str, categories: list) -> str:
    """Parse classification from model response."""
    response_lower = response_text.strip().lower()
    
    for cat in categories:
        if cat in response_lower:
            return cat
    
    return 'generic'

# Legacy function for backwards compatibility
def get_few_shot_classification(teacher_utterance: str, client, conversation_context: str = "") -> str:
    """Legacy function."""
    return get_few_shot_prediction(teacher_utterance, conversation_context, client)