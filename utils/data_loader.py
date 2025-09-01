"""
Data loader for MathDial dataset with teacher move classification
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class MathDialDataLoader:
    """Data loader for MathDial tutoring dialogue dataset"""
    
    def __init__(self, data_path: str = "data/hold_out150.csv"):
        self.data_path = Path(data_path)
        self.teacher_moves = ['generic', 'focus', 'probing', 'telling']
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and process MathDial data from hold_out150.csv
        
        Args:
            sample_size: Number of samples to return (None for all 287 rows)
            
        Returns:
            Tuple of (processed_dataframe, teacher_moves_list)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load the hold_out150.csv file
        df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"Loaded {len(df)} rows from hold_out150.csv")
        
        # Process the data
        df = self._process_mathdial_format(df)
        
        # Apply sampling if requested
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {len(df)} conversations for evaluation")
        
        return df, self.teacher_moves
    
    def _extract_student_name(self, student_profile: str) -> str:
        """Extract student name from profile string"""
        # Extract first word/name from student profile
        if pd.isna(student_profile):
            return "Unknown"
        name_match = re.search(r'(\w+)', str(student_profile))
        return name_match.group(1) if name_match else "Unknown"
    
    def _extract_teacher_moves(self, conversation: str) -> List[str]:
        """
        Extract ground truth teacher moves from conversation
        Returns list of moves in order: ['generic', 'focus', 'probing', ...]
        """
        if pd.isna(conversation):
            return []
        
        # Pattern to find teacher moves in parentheses
        pattern = r'Teacher:\s*\((\w+)\)'
        moves = re.findall(pattern, str(conversation))
        
        # Validate moves are in our expected set
        valid_moves = []
        for move in moves:
            if move.lower() in self.teacher_moves:
                valid_moves.append(move.lower())
        
        return valid_moves
    
    def _clean_conversation(self, conversation: str) -> str:
        """
        Remove ground truth labels and |EOM| markers from conversation
        """
        if pd.isna(conversation):
            return ""
        
        # Remove parenthetical labels
        cleaned = re.sub(r'\(generic\)|\(focus\)|\(probing\)|\(telling\)', '', str(conversation))
        
        # Remove |EOM| markers
        cleaned = cleaned.replace('|EOM|', '\n')
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _process_mathdial_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process MathDial dataset format"""
        
        # Create unique conversation ID
        df['student_name'] = df['student_profile'].apply(self._extract_student_name)
        df['conversation_id'] = df['qid'].astype(str) + '_' + df['student_name']
        
        # Extract ground truth teacher moves
        df['ground_truth_moves'] = df['conversation'].apply(self._extract_teacher_moves)
        
        # Clean conversation text
        df['cleaned_conversation'] = df['conversation'].apply(self._clean_conversation)
        
        # Count teacher moves
        df['num_teacher_moves'] = df['ground_truth_moves'].apply(len)
        
        # Filter out conversations with no teacher moves
        initial_count = len(df)
        df = df[df['num_teacher_moves'] > 0]
        
        if len(df) < initial_count:
            self.logger.info(f"Filtered from {initial_count} to {len(df)} conversations with teacher moves")
            print(f"Filtered to {len(df)} conversations with valid teacher moves")
        
        # Print statistics
        self._print_move_statistics(df)
        
        return df
    
    def _print_move_statistics(self, df: pd.DataFrame):
        """Print statistics about teacher move distribution"""
        print("\nTeacher Move Statistics:")
        
        # Count total moves
        all_moves = []
        for moves_list in df['ground_truth_moves']:
            all_moves.extend(moves_list)
        
        total_moves = len(all_moves)
        print(f"Total teacher moves: {total_moves}")
        
        # Count each move type
        for move in self.teacher_moves:
            count = all_moves.count(move)
            percentage = (count / total_moves) * 100 if total_moves > 0 else 0
            print(f"  {move:<10}: {count}/{total_moves} ({percentage:.1f}%)")
    
    def get_conversation_by_id(self, df: pd.DataFrame, conversation_id: str) -> Dict:
        """Get a specific conversation by ID"""
        conv_df = df[df['conversation_id'] == conversation_id]
        
        if conv_df.empty:
            return None
        
        row = conv_df.iloc[0]
        return {
            'conversation_id': row['conversation_id'],
            'question': row['question'],
            'ground_truth': row['ground_truth'],
            'student_incorrect_solution': row['student_incorrect_solution'],
            'conversation': row['cleaned_conversation'],
            'teacher_moves': row['ground_truth_moves']
        }


# Compatibility functions for existing evaluation code
def load_and_preprocess_mathdial_data(data_path: str = "data/hold_out150.csv", 
                                      sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess MathDial data (compatibility function)"""
    loader = MathDialDataLoader(data_path)
    return loader.load_data(sample_size)

def extract_teacher_moves(conversation: str) -> List[str]:
    """Extract teacher moves from conversation (compatibility function)"""
    pattern = r'Teacher:\s*\((\w+)\)'
    moves = re.findall(pattern, str(conversation))
    valid_moves = ['generic', 'focus', 'probing', 'telling']
    return [move.lower() for move in moves if move.lower() in valid_moves]

def clean_mathdial_conversation(conversation: str) -> str:
    """Clean conversation text (compatibility function)"""
    if pd.isna(conversation):
        return ""
    cleaned = re.sub(r'\(generic\)|\(focus\)|\(probing\)|\(telling\)', '', str(conversation))
    cleaned = cleaned.replace('|EOM|', '\n')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()