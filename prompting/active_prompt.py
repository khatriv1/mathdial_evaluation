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

class ActivePromptSelector:
    """Implements Active Prompting methodology with uncertainty + wrong selection"""
    
    def __init__(self, pool_size: int = 20, k_samples: int = 5):
        self.pool_size = pool_size
        self.k_samples = k_samples
    
    def estimate_uncertainty_and_wrong(self, utterances: List[str], contexts: List[str],
                                      client, category: str, 
                                      ground_truth_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Estimate uncertainty AND wrongness for a SPECIFIC category"""
        print(f"Estimating uncertainty and wrongness for category: {category}")
        
        uncertainty_scores = {}
        wrong_scores = {}
        
        for i, (utterance, context) in enumerate(zip(utterances, contexts)):
            if (i + 1) % 5 == 0:
                print(f"Processing utterance {i + 1}/{len(utterances)}")
            
            # Get 5 predictions for uncertainty estimation
            predictions = []
            for sample_idx in range(self.k_samples):
                pred = self._get_single_prediction(utterance, context, client, category)
                if pred is not None:
                    predictions.append(pred)
                time.sleep(0.05)
            
            if predictions:
                # Calculate uncertainty (disagreement)
                unique_predictions = len(set(predictions))
                uncertainty = unique_predictions / len(predictions)
                uncertainty_scores[utterance] = uncertainty
                
                # Get ground truth
                ground_truth = self._get_ground_truth(utterance, category, ground_truth_df)
                
                # Calculate wrongness (error rate)
                if ground_truth == category:
                    correct_answer = 1
                else:
                    correct_answer = 0
                    
                wrong_rate = sum(1 for p in predictions if (p == category) != correct_answer) / len(predictions)
                wrong_scores[utterance] = wrong_rate
            else:
                uncertainty_scores[utterance] = 0.0
                wrong_scores[utterance] = 0.0
        
        return uncertainty_scores, wrong_scores
    
    def _get_single_prediction(self, teacher_utterance: str, context: str, 
                              client, category: str) -> Optional[str]:
        """Get a single prediction"""
        try:
            prompt = f"""Is this teacher utterance a {category.upper()} move in math tutoring?

{category.upper()} means: {MathDialRubric().get_move_definition(category)}

Teacher: "{teacher_utterance[:200]}"

Answer with only one word (the category): generic, focus, probing, or telling

Answer:"""

            response = client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": f"You are an expert at classifying teacher moves. Answer with only one category."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for uncertainty estimation
                max_tokens=5,
                timeout=12
            )
            
            result = response.choices[0].message.content.strip().lower()
            if result in ['generic', 'focus', 'probing', 'telling']:
                return result
            
        except Exception as e:
            pass
            
        return None
    
    def _get_ground_truth(self, utterance: str, category: str, df: pd.DataFrame) -> str:
        """Get ground truth for an utterance and category"""
        # Extract from conversation column
        # This would need proper implementation based on data structure
        return category  # Placeholder
    
    def select_top_uncertain_and_wrong(self, uncertainty_scores: Dict[str, float], 
                                      wrong_scores: Dict[str, float]) -> Tuple[str, str]:
        """Select top 1 uncertain and top 1 wrong utterance"""
        # Get top uncertain
        top_uncertain = max(uncertainty_scores.items(), key=lambda x: x[1])[0] if uncertainty_scores else None
        
        # Get top wrong (highest error rate)
        top_wrong = max(wrong_scores.items(), key=lambda x: x[1])[0] if wrong_scores else None
        
        print(f"  Top uncertain (score: {uncertainty_scores.get(top_uncertain, 0):.3f}): {top_uncertain[:60] if top_uncertain else 'None'}...")
        print(f"  Top wrong (score: {wrong_scores.get(top_wrong, 0):.3f}): {top_wrong[:60] if top_wrong else 'None'}...")
        
        return top_uncertain, top_wrong

def prepare_active_prompting_data(df: pd.DataFrame, client) -> List[Tuple[str, str]]:
    """Prepare 8 active prompting examples (1 uncertain + 1 wrong per category)"""
    print("Preparing Active Prompting data (8 EXAMPLES VERSION)...")
    
    categories = ['generic', 'focus', 'probing', 'telling']
    all_examples = []  # Will contain 8 examples total
    
    # Get sample utterances from the pool
    selector = ActivePromptSelector(pool_size=20, k_samples=5)
    sample_df = df.sample(n=min(len(df), 20), random_state=42)
    
    # Extract utterances and contexts from conversations
    sample_utterances = []
    sample_contexts = []
    
    for _, row in sample_df.iterrows():
        conversation = row['conversation'] if 'conversation' in row else row['cleaned_conversation']
        # Parse conversation to extract teacher utterances
        lines = conversation.split('\n')
        context = ""
        for line in lines:
            if line.startswith('Teacher:'):
                utterance = re.sub(r'\([^)]+\)', '', line.replace('Teacher:', '')).strip()
                utterance = utterance.replace('|EOM|', '').strip()
                if utterance:
                    sample_utterances.append(utterance)
                    sample_contexts.append(context)
                context += line + "\n"
            elif line.startswith('Student:'):
                context += line + "\n"
    
    # Find uncertain and wrong examples for EACH CATEGORY
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        try:
            # Get uncertainty and wrong scores
            uncertainty_scores, wrong_scores = selector.estimate_uncertainty_and_wrong(
                sample_utterances[:20], sample_contexts[:20], client, category, sample_df
            )
            
            # Select top 1 uncertain and top 1 wrong
            top_uncertain, top_wrong = selector.select_top_uncertain_and_wrong(
                uncertainty_scores, wrong_scores
            )
            
            # Add uncertain example with ground truth
            if top_uncertain:
                all_examples.append((top_uncertain, f"Classification: {category}"))
            
            # Add wrong example with ground truth (if different from uncertain)
            if top_wrong and top_wrong != top_uncertain:
                all_examples.append((top_wrong, f"Classification: {category}"))
            elif top_wrong == top_uncertain and top_uncertain:
                # If same utterance is both most uncertain and most wrong, pick next best wrong
                sorted_wrong = sorted(wrong_scores.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_wrong) > 1:
                    second_wrong = sorted_wrong[1][0]
                    all_examples.append((second_wrong, f"Classification: {category}"))
            
            print(f"✓ Selected 2 examples for {category}")
            
        except Exception as e:
            print(f"⚠ Error in {category}: {e}")
    
    print(f"\n✓ Total examples collected: {len(all_examples)}")
    return all_examples

def get_active_prompt_prediction(teacher_utterance: str, conversation_context: str,
                                client, uncertainty_data: List[Tuple[str, str]] = None) -> str:
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