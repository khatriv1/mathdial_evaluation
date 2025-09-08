# mathdial_apo_system.py - RUBRIC OPTIMIZATION FOR MATHDIAL

import pandas as pd
import numpy as np
import openai
from typing import Dict, List, Tuple, Any, Optional
import json
import time
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys
import os
import shutil
from collections import Counter

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

@dataclass
class RubricCandidate:
    """Represents a candidate rubric with its performance metrics"""
    rubric_definitions: Dict[str, str]
    performance_scores: Dict[str, float]  # Score for each technique
    average_score: float
    detailed_metrics: Dict[str, Dict[str, float]]  # Metrics for each technique

class MathDialRubricAPO:
    """
    Automatic Prompt Optimization for MathDial Teacher Move Classification
    Optimizes RUBRIC definitions instead of prompt instructions
    """
    
    def __init__(self, api_key: str, 
                 validation_sample_size: int = 100,
                 evaluation_sample_size: int = 50):
        """
        Initialize Rubric APO system for MathDial
        Args:
            api_key: OpenAI API key
            validation_sample_size: Not used in current implementation
            evaluation_sample_size: Number of samples to use (10, 20, or 50)
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set OpenAI API key
        openai.api_key = api_key
        self.client = self._get_openai_client()
        
        self.validation_sample_size = validation_sample_size
        self.evaluation_sample_size = evaluation_sample_size
        self.teacher_moves = ['generic', 'focus', 'probing', 'telling']
        
        # Initialize optimization results cache
        self._optimization_results = None
        self._optimization_completed = False
        
        # Initialize Active Prompting cache
        self._active_prompting_uncertainty_cache = None
        
        # Directory setup
        self.prompting_dir = "../prompting"  
        self.utils_dir = "../utils"  
        self.apo_copies_dir = "./apo_copies"  
        
        # Create APO copies directory
        self._setup_apo_copies()
        
        # Get baseline rubric definitions
        self.baseline_rubric = self._get_baseline_rubric()
        
        # Techniques to test
        self.techniques = ['zero_shot', 'few_shot', 'auto_cot', 'self_consistency', 'active_prompting']
        
        # Load and prepare data - ONLY uses the two required files
        self.logger.info("Loading APO data...")
        
        # Define the ONLY TWO files we need - NO hold_out150.csv!
        apo_file = '../data/apo_training_100.csv'
        active_pool_file = '../data/active_prompting_pool_20.csv'
        
        # Check for APO training set
        if not os.path.exists(apo_file):
            self.logger.error(f"ERROR: APO training file not found at {apo_file}")
            self.logger.error("This file is required for APO optimization.")
            self.logger.error("Please ensure apo_training_100.csv exists in ../data/ folder")
            sys.exit(1)
        
        # Check for Active Prompting pool
        if not os.path.exists(active_pool_file):
            self.logger.error(f"ERROR: Active Prompting pool not found at {active_pool_file}")
            self.logger.error("This file is required for active prompting technique.")
            self.logger.error("Please ensure active_prompting_pool_20.csv exists in ../data/ folder")
            sys.exit(1)
        
        # Both files exist, load them properly
        self.logger.info("✓ Found both required data files")
        
        # Load APO training set using MathDialDataLoader
        from utils.data_loader import MathDialDataLoader
        loader = MathDialDataLoader(apo_file)
        self.apo_set, _ = loader.load_data()
        self.logger.info(f"  Loaded {len(self.apo_set)} conversations from APO training set")
        
        # Load Active Prompting pool using MathDialDataLoader
        active_loader = MathDialDataLoader(active_pool_file)
        self.active_pool_df, _ = active_loader.load_data()
        self.logger.info(f"  Loaded {len(self.active_pool_df)} conversations from active prompting pool")
        
        # Use nested sampling from APO set based on evaluation_sample_size
        if evaluation_sample_size <= len(self.apo_set):
            self.validation_data = self.apo_set.iloc[:evaluation_sample_size]
        else:
            self.validation_data = self.apo_set
            self.logger.info(f"  Note: Requested {evaluation_sample_size} samples but only {len(self.apo_set)} available, using all")
        
        self.logger.info(f"Using {len(self.validation_data)} conversations for validation")
        
        # Count total utterances
        total_utterances = sum(len(row.get('ground_truth_moves', [])) for _, row in self.validation_data.iterrows())
        self.logger.info(f"Total teacher utterances to evaluate: {total_utterances}")
        
        # Store original mathdial_rubric.py content
        self.original_rubric_content = self._store_original_rubric()
        
        # Generate rubric variations
        self.rubric_variations = []  # Will be populated during optimization
    
    def _get_openai_client(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        return OpenAI(api_key=openai.api_key)
    
    def _setup_apo_copies(self):
        """Create APO copies directory with prompting and utils folders"""
        
        # Create apo_copies directory
        if os.path.exists(self.apo_copies_dir):
            shutil.rmtree(self.apo_copies_dir)
        os.makedirs(self.apo_copies_dir)
        
        # Copy entire prompting folder
        src_prompting = self.prompting_dir
        dst_prompting = os.path.join(self.apo_copies_dir, "prompting")
        if os.path.exists(src_prompting):
            shutil.copytree(src_prompting, dst_prompting)
            self.logger.info(f"Copied prompting folder to {dst_prompting}")
        
        # Copy entire utils folder
        src_utils = self.utils_dir
        dst_utils = os.path.join(self.apo_copies_dir, "utils")
        if os.path.exists(src_utils):
            shutil.copytree(src_utils, dst_utils)
            self.logger.info(f"Copied utils folder to {dst_utils}")
        
        self.logger.info(f"APO copies created in {self.apo_copies_dir}")
    
    def _store_original_rubric(self):
        """Store original mathdial_rubric.py content"""
        rubric_file = os.path.join(self.apo_copies_dir, "utils", "mathdial_rubric.py")
        if os.path.exists(rubric_file):
            with open(rubric_file, 'r') as f:
                return f.read()
        return None
    
    def _get_baseline_rubric(self) -> Dict[str, str]:
        """Get baseline rubric definitions for MathDial"""
        return {
            'generic': "The teachers intention includes Greeting/Fairwell and General inquiry",
            'focus': "The teachers intention includes seeking a strategy, guiding student focus, and recalling relevant information",
            'probing': "The teachers intention includes Asking for Explanation, Seeking Self Correction, Perturbing the Question, and Seeking World Knowledge",
            'telling': "The teachers intention includes Revealing Strategy and Revealing Answer"
        }
    
    def _generate_rubric_variations(self) -> List[Dict[str, str]]:
        """Generate 5 variations of the rubric definitions using the EXACT prompt provided"""
        
        prompt = f"""You are a research assistant refining a teacher's moves codebook for LLMs. I will give you a baseline codebook (JSON). Using it as a reference for scope and label meanings, produce FIVE alternative codebooks that help an LLM classify teachers' moves more accurately.

Insert the provided baseline here:
{json.dumps(self.baseline_rubric, indent=2)}

Examples:
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

Each definition should be 1-2 sentences and create meaningfully different classification boundaries for borderline cases.
Return EXACTLY 5 variations as a JSON array. Each variation should contain all 4 categories.

Return in this EXACT format:
[
  {{
    "generic": "definition here",
    "focus": "definition here",
    "probing": "definition here",
    "telling": "definition here"
  }},
  {{
    "generic": "definition here",
    "focus": "definition here",
    "probing": "definition here",
    "telling": "definition here"
  }},
  {{
    "generic": "definition here",
    "focus": "definition here",
    "probing": "definition here",
    "telling": "definition here"
  }},
  {{
    "generic": "definition here",
    "focus": "definition here",
    "probing": "definition here",
    "telling": "definition here"
  }},
  {{
    "generic": "definition here",
    "focus": "definition here",
    "probing": "definition here",
    "telling": "definition here"
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            import config
            model_name = getattr(config, 'MODEL_ID', 'gpt-3.5-turbo')
            
            self.logger.info(f"Generating rubric variations using {model_name}...")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at refining classification rubrics. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000
            )
            
            result = response.choices[0].message.content.strip()
            
            # LOG THE RAW RESPONSE TO SEE WHAT WE GET
            self.logger.info("="*60)
            self.logger.info("Model RAW RESPONSE:")
            self.logger.info(result[:500] + "..." if len(result) > 500 else result)
            self.logger.info("="*60)
            
            # Try to parse as JSON
            try:
                # Remove markdown code blocks if present
                if '```json' in result.lower():
                    result = result.split('```json')[1].split('```')[0]
                elif '```' in result:
                    result = result.split('```')[1].split('```')[0]
                
                variations = json.loads(result)
                
                if isinstance(variations, list) and len(variations) >= 5:
                    self.logger.info(f"✓ Successfully parsed {len(variations)} variations!")
                    
                    # LOG EACH VARIATION SO WE CAN SEE THEM
                    for i, var in enumerate(variations[:5], 1):
                        self.logger.info(f"\nVARIATION {i}:")
                        self.logger.info(f"  Generic: {var.get('generic', 'N/A')[:50]}...")
                        self.logger.info(f"  Focus: {var.get('focus', 'N/A')[:50]}...")
                    
                    return variations[:5]
                else:
                    self.logger.warning(f"JSON had {len(variations) if isinstance(variations, list) else 0} variations, expected 5")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
            
            # Fallback variations
            self.logger.warning("Using fallback variations")
            return self._get_fallback_variations()
        
        except Exception as e:
            self.logger.error(f"Error generating variations: {e}")
            return self._get_fallback_variations()
    
    def _get_fallback_variations(self):
        """Provide fallback variations if generation fails"""
        return [
            {
                "generic": "Teacher utterances that establish rapport, greet students, or make general conversational remarks without mathematical content",
                "focus": "Teacher utterances directing student attention to specific problem aspects or guiding their problem-solving approach",
                "probing": "Teacher utterances requesting explanations, justifications, or encouraging students to reflect on their reasoning",
                "telling": "Teacher utterances directly providing solutions, answers, or explicit problem-solving steps"
            },
            {
                "generic": "Social interactions and general classroom management utterances unrelated to the mathematical task",
                "focus": "Questions or statements that orient students toward the next step or relevant problem information",
                "probing": "Utterances that challenge student thinking or ask them to explain their mathematical reasoning",
                "telling": "Direct instruction providing specific mathematical information or correcting student errors"
            },
            {
                "generic": "Non-instructional utterances including greetings, praise, and general encouragement",
                "focus": "Guidance that helps students identify what to do next without providing the answer",
                "probing": "Questions that require students to justify, explain, or reconsider their approach",
                "telling": "Explicit provision of mathematical facts, procedures, or corrections"
            },
            {
                "generic": "Utterances for social interaction and general classroom discourse without mathematical content",
                "focus": "Scaffolding utterances that guide student thinking toward productive problem-solving strategies",
                "probing": "Utterances prompting students to articulate their reasoning or evaluate their solutions",
                "telling": "Direct communication of mathematical answers or solution methods"
            },
            {
                "generic": "General conversational utterances that maintain social interaction but don't advance problem-solving",
                "focus": "Strategic questions or hints that direct attention to important problem elements",
                "probing": "Metacognitive prompts asking students to explain, justify, or reflect on their mathematical thinking",
                "telling": "Explicit instruction providing mathematical information, answers, or procedures"
            }
        ]
    
    def _update_rubric_file(self, rubric_definitions: Dict[str, str]):
        """Update the mathdial_rubric.py file with new definitions"""
        
        rubric_file = os.path.join(self.apo_copies_dir, "utils", "mathdial_rubric.py")
        
        # Read the current content
        with open(rubric_file, 'r') as f:
            content = f.read()
        
        # Update each category's description
        for category, definition in rubric_definitions.items():
            # Escape special characters in the definition
            escaped_definition = definition.replace('"', '\\"').replace("'", "\\'")
            
            # Find and replace the description for each category
            # Look for the pattern in move_definitions dictionary
            pattern = f"'{category}':\\s*['\"][^'\"]*['\"]"
            replacement = f"'{category}': '{escaped_definition}'"
            content = re.sub(pattern, replacement, content)
        
        # Write back
        with open(rubric_file, 'w') as f:
            f.write(content)
        
        self.logger.debug(f"Updated rubric definitions in {rubric_file}")
    
    def _restore_original_rubric(self):
        """Restore the original rubric file"""
        rubric_file = os.path.join(self.apo_copies_dir, "utils", "mathdial_rubric.py")
        if self.original_rubric_content:
            with open(rubric_file, 'w') as f:
                f.write(self.original_rubric_content)
    
    def evaluate_rubric_with_all_techniques(self, rubric_definitions: Dict[str, str], sample_data: pd.DataFrame) -> Dict[str, Dict]:
        """Test a rubric with ALL prompting techniques - LIMITED TO EXACT UTTERANCE COUNTS"""
        
        # Update the rubric file
        self._update_rubric_file(rubric_definitions)
        
        # Clear module cache
        modules_to_remove = []
        for mod_name in list(sys.modules.keys()):
            if 'prompting' in mod_name or 'utils' in mod_name or 'mathdial_rubric' in mod_name:
                modules_to_remove.append(mod_name)
        
        for mod_name in modules_to_remove:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        # Ensure apo_copies is first in path
        copies_path = self.apo_copies_dir
        if copies_path in sys.path:
            sys.path.remove(copies_path)
        sys.path.insert(0, copies_path)
        
        # DETERMINE TARGET UTTERANCE COUNT BASED ON CONVERSATIONS
        # 2 conversations -> 10 utterances
        # 4 conversations -> 20 utterances  
        # 9 conversations -> 50 utterances
        target_utterances = {2: 10, 4: 20, 9: 50}
        max_utterances = target_utterances.get(len(sample_data), 50)
        
        # FIRST, COLLECT ALL UTTERANCES FROM ALL CONVERSATIONS
        all_utterance_data = []
        
        for idx, row in sample_data.iterrows():
            full_conversation = row.get('cleaned_conversation', '')
            ground_truth_moves = row.get('ground_truth_moves', [])
            question = row.get('question', '')
            student_solution = row.get('student_incorrect_solution', '')
            student_profile = row.get('student_profile', '')
            
            lines = full_conversation.split('\n')
            conversation_context = ""
            teacher_utterance_idx = 0
            
            for line in lines:
                if line.startswith('Teacher:'):
                    teacher_utterance = line.replace('Teacher:', '').strip()
                    
                    if teacher_utterance and teacher_utterance_idx < len(ground_truth_moves):
                        # Store this utterance data
                        all_utterance_data.append({
                            'teacher_utterance': teacher_utterance,
                            'conversation_context': conversation_context,
                            'true_label': ground_truth_moves[teacher_utterance_idx],
                            'question': question,
                            'student_solution': student_solution,
                            'student_profile': student_profile
                        })
                        teacher_utterance_idx += 1
                        
                    conversation_context += f"Teacher: {teacher_utterance}\n"
                    
                elif line.startswith('Student:'):
                    student_utterance = line.replace('Student:', '').strip()
                    conversation_context += f"Student: {student_utterance}\n"
        
        # LIMIT TO EXACT NUMBER OF UTTERANCES
        utterances_to_evaluate = all_utterance_data[:max_utterances]
        
        self.logger.info(f"  Collected {len(all_utterance_data)} total utterances")
        self.logger.info(f"  Evaluating exactly {len(utterances_to_evaluate)} utterances (target: {max_utterances})")
        
        results = {}
        
        # Test each technique
        for technique in self.techniques:
            self.logger.info(f"  Testing {technique} with current rubric...")
            
            try:
                import config
                
                # Import the appropriate function
                if technique == 'zero_shot':
                    from prompting.zero_shot import get_zero_shot_prediction
                    get_prediction_func = get_zero_shot_prediction
                elif technique == 'few_shot':
                    from prompting.few_shot import get_few_shot_prediction
                    get_prediction_func = get_few_shot_prediction
                elif technique == 'auto_cot':
                    from prompting.auto_cot import get_auto_cot_prediction
                    get_prediction_func = get_auto_cot_prediction
                elif technique == 'self_consistency':
                    from prompting.self_consistency import get_self_consistency_prediction
                    get_prediction_func = get_self_consistency_prediction
                elif technique == 'active_prompting':
                    from prompting.active_prompt import get_active_prompt_prediction
                    get_prediction_func = get_active_prompt_prediction
                
                # Prepare active prompting data if needed
                uncertainty_data = None
                if technique == 'active_prompting' and self.active_pool_df is not None:
                    if self._active_prompting_uncertainty_cache is None:
                        self.logger.info("    Preparing Active Prompting uncertainty examples...")
                        from prompting.active_prompt import prepare_active_prompting_data
                        self._active_prompting_uncertainty_cache = prepare_active_prompting_data(self.active_pool_df, self.client)
                        self.logger.info(f"    Cached {len(self._active_prompting_uncertainty_cache)} uncertainty examples")
                    uncertainty_data = self._active_prompting_uncertainty_cache
                
                # Process ONLY THE LIMITED UTTERANCES
                all_predictions = []
                all_true_labels = []
                
                for utterance_info in utterances_to_evaluate:
                    # Get prediction for this utterance
                    if technique == 'self_consistency':
                        prediction = get_prediction_func(
                            teacher_utterance=utterance_info['teacher_utterance'],
                            conversation_context=utterance_info['conversation_context'],
                            client=self.client,
                            question=utterance_info['question'],
                            student_solution=utterance_info['student_solution'],
                            student_profile=utterance_info['student_profile'],
                            n_samples=3
                        )
                    elif technique == 'active_prompting' and uncertainty_data:
                        prediction = get_prediction_func(
                            teacher_utterance=utterance_info['teacher_utterance'],
                            conversation_context=utterance_info['conversation_context'],
                            client=self.client,
                            question=utterance_info['question'],
                            student_solution=utterance_info['student_solution'],
                            student_profile=utterance_info['student_profile'],
                            uncertainty_data=uncertainty_data
                        )
                    else:
                        prediction = get_prediction_func(
                            teacher_utterance=utterance_info['teacher_utterance'],
                            conversation_context=utterance_info['conversation_context'],
                            client=self.client,
                            question=utterance_info['question'],
                            student_solution=utterance_info['student_solution'],
                            student_profile=utterance_info['student_profile']
                        )
                    
                    # Validate prediction
                    if prediction not in self.teacher_moves:
                        prediction = 'generic'
                    
                    all_predictions.append(prediction)
                    all_true_labels.append(utterance_info['true_label'])
                    
                    time.sleep(0.05)  # Small delay between API calls
                
                # Calculate metrics on EXACTLY the target number of utterances
                if all_predictions and all_true_labels:
                    accuracy = accuracy_score(all_true_labels, all_predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_true_labels, all_predictions, average='weighted', zero_division=0
                    )
                else:
                    accuracy = precision = recall = f1 = 0
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'total_utterances': len(all_predictions)
                }
                
                results[technique] = metrics
                self.logger.info(f"    {technique} accuracy: {accuracy:.3f} ({len(all_predictions)} utterances)")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {technique}: {e}")
                import traceback
                traceback.print_exc()
                results[technique] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'total_utterances': 0}
        
        # Restore original rubric
        self._restore_original_rubric()
        
        # Clear cache
        for mod_name in modules_to_remove:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        
        return results
    
    def optimize_rubrics(self) -> RubricCandidate:
        """Main optimization function - test baseline + 5 variations"""
        
        self.logger.info("="*60)
        self.logger.info("STARTING MATHDIAL RUBRIC APO OPTIMIZATION")
        self.logger.info("Testing 6 rubrics (baseline + 5 variations)")
        self.logger.info("="*60)
        
        # Generate 5 rubric variations
        self.logger.info("\nGenerating 5 rubric variations...")
        self.rubric_variations = self._generate_rubric_variations()
        self.logger.info(f"✓ Generated {len(self.rubric_variations)} variations")
        
        # Store all results
        all_candidates = []
        
        # Test baseline rubric
        self.logger.info("\n" + "="*40)
        self.logger.info("Testing BASELINE rubric...")
        self.logger.info("="*40)
        baseline_results = self.evaluate_rubric_with_all_techniques(self.baseline_rubric, self.validation_data)
        
        baseline_candidate = RubricCandidate(
            rubric_definitions=self.baseline_rubric,
            performance_scores={tech: metrics.get('accuracy', 0) for tech, metrics in baseline_results.items()},
            average_score=np.mean([metrics.get('accuracy', 0) for metrics in baseline_results.values()]),
            detailed_metrics=baseline_results
        )
        all_candidates.append(baseline_candidate)
        
        self.logger.info(f"Baseline average score: {baseline_candidate.average_score:.3f}")
        
        # Test each variation
        for i, variation in enumerate(self.rubric_variations, 1):
            self.logger.info("\n" + "="*40)
            self.logger.info(f"Testing VARIATION {i}/5...")
            self.logger.info("="*40)
            
            variation_results = self.evaluate_rubric_with_all_techniques(variation, self.validation_data)
            
            variation_candidate = RubricCandidate(
                rubric_definitions=variation,
                performance_scores={tech: metrics.get('accuracy', 0) for tech, metrics in variation_results.items()},
                average_score=np.mean([metrics.get('accuracy', 0) for metrics in variation_results.values()]),
                detailed_metrics=variation_results
            )
            all_candidates.append(variation_candidate)
            
            self.logger.info(f"Variation {i} average score: {variation_candidate.average_score:.3f}")
        
        # Find best candidate
        best_candidate = max(all_candidates, key=lambda x: x.average_score)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("OPTIMIZATION COMPLETE!")
        self.logger.info(f"Best average score: {best_candidate.average_score:.3f}")
        self.logger.info("="*60)
        
        return best_candidate
    
    def save_results(self, filename: str):
        """Save optimization results to JSON"""
        
        best_candidate = self.optimize_rubrics()
        
        output_data = {
            'evaluation_sample_size': self.evaluation_sample_size,
            'baseline_rubric': self.baseline_rubric,
            'best_rubric': best_candidate.rubric_definitions,
            'all_variations': self.rubric_variations,
            'performance_by_technique': best_candidate.performance_scores,
            'average_score': best_candidate.average_score,
            'detailed_metrics': best_candidate.detailed_metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
        return best_candidate

# Main execution
if __name__ == "__main__":
    print("MathDial Rubric APO System")
    print("Run this through run_apo.py for full automation")