"""
Teacher moves rubric for MathDial dataset evaluation
"""

from typing import List, Dict, Tuple

class MathDialRubric:
    """
    Rubric for MathDial teacher moves classification
    """
    
    def __init__(self):
        self.teacher_moves = ['generic', 'focus', 'probing', 'telling']
        
        # Teacher move definitions based on the categories
        self.move_definitions = {
            # 'focus': 'The teachers intention includes seeking a strategy, guiding student focus, and recalling relevant information',
            # 'probing': 'The teachers intention includes Asking for Explanation, Seeking Self Correction, Perturbing the Question, and Seeking World Knowledge',
            # 'telling': 'The teachers intention includes Revealing Strategy and Revealing Answer',
            # 'generic': 'The teachers intention includes Greeting/Fairwell and General inquiry'
            "generic": "Utterances for social interaction and general classroom discourse without mathematical content",
            "focus": "Scaffolding utterances that guide student thinking toward productive problem-solving strategies",
            "probing": "Utterances prompting students to articulate their reasoning or evaluate their solutions",
            "telling": "Direct communication of mathematical answers or solution methods"
       
           
        }
        
        # Detailed intents for each move category - FROM THE TABLE
        self.move_intents = {
            'focus': {
                'Seek Strategy': "So what should you do next?",
                'Guiding Student Focus': "Can you calculate ... ?",
                'Recall Relevant Information': "Can you reread the question and tell me what is ... ?"
            },
            'probing': {
                'Asking for Explanation': "Why do you think you need to add these numbers?",
                'Seeking Self Correction': "Are you sure you need to add here?",
                'Perturbing the Question': "How would things change if they had ... items instead?",
                'Seeking World Knowledge': "How do you calculate the perimeter of a square?"
            },
            'telling': {
                'Revealing Strategy': "You need to add ... to ... to get your answer.",
                'Revealing Answer': "No, he had ... items."
            },
            'generic': {
                'Greeting/Fairwell': "Hi ..., how are you doing with the word problem?",
                'General inquiry': "Can you go walk me through your solution?"
            }
        }
        
        # Example utterances for each move - EXACT examples from the table
        self.move_examples = {
            'focus': [
                "So what should you do next?",
                "Can you calculate ... ?", 
                "Can you reread the question and tell me what is ... ?"
            ],
            'probing': [
                "Why do you think you need to add these numbers?",
                "Are you sure you need to add here?",
                "How would things change if they had ... items instead?",
                "How do you calculate the perimeter of a square?"
            ],
            'telling': [
                "You need to add ... to ... to get your answer.",
                "No, he had ... items."
            ],
            'generic': [
                "Hi ..., how are you doing with the word problem?",
                "Good Job! Is there anything else I can help with?",
                "Can you go walk me through your solution?"
            ]
        }
    
    def get_move_definition(self, move: str) -> str:
        """Get the definition of a specific teacher move"""
        return self.move_definitions.get(move.lower(), "Unknown move")
    
    def get_all_moves(self) -> List[str]:
        """Get list of all teacher moves"""
        return self.teacher_moves.copy()
    
    def format_moves_for_prompt(self) -> str:
        """Format teacher moves for use in prompts"""
        prompt_text = "Teacher moves:\n"
        for move in self.teacher_moves:
            prompt_text += f"- {move.upper()}: {self.move_definitions[move]}\n"
            if move in self.move_intents:
                for intent, example in self.move_intents[move].items():
                    prompt_text += f"  - {intent}: {example}\n"
        return prompt_text
    
    def get_move_examples(self, move: str, num_examples: int = 2) -> List[str]:
        """Get example utterances for a specific move"""
        if move.lower() not in self.move_examples:
            return []
        
        examples = self.move_examples[move.lower()]
        return examples[:min(num_examples, len(examples))]
    
    def get_intent_examples(self, move: str) -> Dict[str, str]:
        """Get intent-example pairs for a specific move"""
        return self.move_intents.get(move.lower(), {})
    
    def validate_move(self, move: str) -> bool:
        """Check if a teacher move is valid"""
        return move.lower() in self.teacher_moves
    
    def parse_move_response(self, response_text: str) -> List[str]:
        """
        Parse teacher move response from LLM output
        Tries to extract a list of moves from various response formats
        """
        import re
        
        response_text = response_text.strip().lower()
        
        # Look for moves mentioned in the text
        found_moves = []
        
        for move in self.teacher_moves:
            if move in response_text:
                found_moves.append(move)
        
        return found_moves
    
    def validate_moves(self, moves: List[str]) -> List[str]:
        """Validate that moves are in the valid move set"""
        valid_moves = []
        for move in moves:
            if move.lower() in self.teacher_moves:
                valid_moves.append(move.lower())
        return valid_moves
    
    def get_scaffolding_moves(self) -> List[str]:
        """Get moves that are considered scaffolding (Focus + Probing)"""
        return ['focus', 'probing']
    
    def is_scaffolding(self, move: str) -> bool:
        """Check if a move is a scaffolding move"""
        return move.lower() in self.get_scaffolding_moves()
    
    def get_prompt_descriptions(self) -> Dict[str, str]:
        """Get prompt descriptions for all moves"""
        prompt_descriptions = {}
        
        for move in self.teacher_moves:
            prompt_descriptions[move] = (
                f"Teacher uses {move}: {self.move_definitions[move]}"
            )
        
        return prompt_descriptions
    
    def get_all_intents(self) -> List[str]:
        """Get list of all intents across all moves"""
        all_intents = []
        for move_intents in self.move_intents.values():
            all_intents.extend(move_intents.keys())
        return all_intents