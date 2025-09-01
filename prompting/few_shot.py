# mathdial_evaluation/prompting/few_shot.py

"""
Few-shot prompting for MathDial teacher move classification.
Uses examples from Table 2 of the MathDial paper.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import json
import re
from typing import Optional, Dict
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

def get_few_shot_prediction(teacher_utterance: str, conversation_context: str, client) -> str:
    """
    Get few-shot prediction for teacher move classification.
    
    Args:
        teacher_utterance: The teacher's utterance to classify
        conversation_context: Previous conversation context
        client: OpenAI client
    
    Returns:
        One of: 'generic', 'focus', 'probing', 'telling'
    """
    categories = ['generic', 'focus', 'probing', 'telling']
    
    rubric = MathDialRubric()
    
    examples_text = get_few_shot_examples()
    
    definitions_text = ""
    for cat in categories:
        definitions_text += f"{cat.upper()}: {rubric.get_move_definition(cat)}\n"

    prompt = f"""You are an expert educator classifying teacher moves in math tutoring conversations.

TEACHER MOVE CATEGORIES:
{definitions_text}

{examples_text}

TASK: Now analyze the following teacher utterance and classify it based on the examples above.

Now work on this one:

Conversation Context:
{conversation_context}

Teacher: "{teacher_utterance}"

INSTRUCTIONS:
1. Learn from the examples provided above
2. Analyze the teacher utterance carefully  
3. Consider the pedagogical intent
4. Respond with only one word: generic, focus, probing, or telling

Classification:"""

    try:
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[
                {"role": "system", "content": "You are an expert at classifying teacher moves in math tutoring. Learn from the examples provided. Respond with only one category."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
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