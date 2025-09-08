# mathdial_evaluation/prompting/zero_shot.py

"""
Zero-shot prompting for MathDial teacher move classification.
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

def parse_classification(response_text: str, categories: list) -> str:
    """Parse teacher move classification from model response."""
    response_lower = response_text.strip().lower()
    
    # Try direct match
    for cat in categories:
        if cat in response_lower:
            return cat
    
    # Default fallback
    return 'generic'

def get_zero_shot_prediction(teacher_utterance: str, conversation_context: str, client,
                            question: str = None, student_solution: str = None,
                            student_profile: str = None, utterance_num: int = None) -> str:
    """
    Get zero-shot prediction for a SINGLE teacher utterance with full context.
    
    IMPORTANT: This function classifies ONE teacher utterance at a time,
    but uses the FULL conversation context up to that point.
    
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
        print(f"\n  [DEBUG] Classifying Teacher Utterance #{utterance_num}:")
        print(f"    Utterance: \"{teacher_utterance[:60]}...\"")
        print(f"    Context length: {len(conversation_context.split())} words")
    
    prompt = f"""Your expertise lies in categorizing teacher moves in math tutoring based on pedagogical intent.

TEACHER MOVE CATEGORIES:
{definitions_text}

TASK: Analyze ONE SPECIFIC teacher utterance and classify it into one of the four categories above.
You have the full conversation context to understand the situation, but you must classify ONLY the specified utterance.

{context_info}

CONVERSATION CONTEXT (everything that happened before this utterance):
{conversation_context}

CURRENT TEACHER UTTERANCE TO CLASSIFY (classify ONLY this):
Teacher: "{teacher_utterance}"

INSTRUCTIONS:
1. Read the full context to understand the conversation flow
2. Focus on THIS SPECIFIC teacher utterance
3. Analyze the pedagogical intent of THIS utterance
4. Classify ONLY this utterance (not the whole conversation)
5. Respond with only one word: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying individual teacher utterances in math tutoring. Classify ONLY the specified utterance, using context for understanding."},
                {"role": "user", "content": prompt}
            ],
            #temperature=0,
            #max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the classification
        classification = parse_classification(result, categories)
        
        return classification
        
    except Exception as e:
        print(f"Error in zero-shot classification: {str(e)}")
        return 'generic'

def analyze_conversation_moves(conversation: str, client, question: str = None,
                              student_solution: str = None, student_profile: str = None) -> List[Tuple[str, str]]:
    """
    Analyze ALL teacher moves in a complete conversation.
    
    IMPORTANT: This function processes EACH teacher utterance individually,
    building up context as it goes through the conversation.
    
    For each teacher utterance:
    1. It uses all previous conversation as context
    2. Classifies ONLY that specific utterance
    3. Adds that utterance to the context for the next one
    
    Args:
        conversation: The full cleaned conversation
        client: OpenAI client
        question: Math problem
        student_solution: Student's incorrect solution
        student_profile: Student info
        
    Returns:
        List of (utterance, classification) tuples
    """
    results = []
    
    # Parse conversation into turns
    lines = conversation.split('\n')
    context = ""  # This will build up as we process the conversation
    utterance_num = 0
    
    print("\n  Processing conversation utterance by utterance...")
    print(f"  Total lines in conversation: {len(lines)}")
    
    for line in lines:
        if line.startswith('Teacher:'):
            utterance_num += 1
            
            # Extract teacher utterance
            utterance = line.replace('Teacher:', '').strip()
            
            # Remove any move labels if present (from ground truth)
            utterance = re.sub(r'\([^)]+\)', '', utterance)
            utterance = utterance.replace('|EOM|', '').strip()
            
            if utterance:
                # CRITICAL: Classify this utterance with ALL PREVIOUS context
                # but NOT including this utterance itself in the context yet
                classification = get_zero_shot_prediction(
                    utterance, 
                    context,  # Context up to BUT NOT INCLUDING this utterance
                    client,
                    question=question,
                    student_solution=student_solution,
                    student_profile=student_profile,
                    utterance_num=utterance_num
                )
                results.append((utterance, classification))
                
                # NOW add this utterance to context for the next classification
                context += f"Teacher: {utterance}\n"
            
        elif line.startswith('Student:'):
            # Add student turn to context
            student_utterance = line.replace('Student:', '').strip()
            student_utterance = student_utterance.replace('|EOM|', '').strip()
            context += f"Student: {student_utterance}\n"
    
    print(f"  Classified {len(results)} teacher utterances")
    
    return results

# Legacy function for backwards compatibility
def get_zero_shot_classification(teacher_utterance: str, client, conversation_context: str = "") -> str:
    """Legacy function for single utterance classification."""
    return get_zero_shot_prediction(teacher_utterance, conversation_context, client)