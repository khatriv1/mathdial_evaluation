"""
COMPLETE FIXED Data loader for MathDial dataset
- Groups rows by QID + Student Name to form unique conversations
- Preserves original CSV order
- Properly handles |EOM| markers
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class MathDialDataLoader:
    """Data loader for MathDial tutoring dialogue dataset"""
    
    def __init__(self, data_path: str):  # FIXED: No default path
        self.data_path = Path(data_path)
        self.teacher_moves = ['generic', 'focus', 'probing', 'telling']
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and process MathDial data from CSV
        Groups by QID + Student Name to form unique conversations
        Preserves original order from CSV
        
        Args:
            sample_size: Number of CONVERSATIONS to return
            
        Returns:
            Tuple of (processed_dataframe with one row per conversation, teacher_moves_list)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load the CSV file
        df_raw = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(df_raw)} rows with {len(df_raw.columns)} columns")
        print(f"Loaded {len(df_raw)} rows from {self.data_path.name}")  # FIXED: Use actual filename
        
        # Extract student names from each row
        df_raw['student_name'] = df_raw['student_profile'].apply(self._extract_student_name)
        
        # Create unique conversation identifier
        df_raw['conversation_id'] = df_raw['qid'].astype(str) + '_' + df_raw['student_name']
        
        # Show the actual structure
        unique_qids = df_raw['qid'].nunique()
        unique_conversations = df_raw['conversation_id'].nunique()
        
        print(f"Found {unique_qids} unique QIDs (questions)")
        print(f"Found {unique_conversations} unique conversations (QID + Student combinations)")
        print(f"Average {len(df_raw)/unique_conversations:.1f} utterances per conversation")
        
        # Group rows by conversation_id to reconstruct full conversations
        df_conversations = self._reconstruct_conversations(df_raw)
        
        print(f"Reconstructed {len(df_conversations)} complete conversations")
        
        # Apply sampling if requested - use head() to preserve order
        if sample_size is not None and sample_size < len(df_conversations):
            df_conversations = df_conversations.head(sample_size)
            print(f"Selected first {sample_size} conversations for evaluation")
        
        # Print statistics
        self._print_move_statistics(df_conversations)
        
        return df_conversations, self.teacher_moves
    
    def _extract_student_name(self, student_profile: str) -> str:
        """Extract student name from profile string"""
        if pd.isna(student_profile):
            return "Unknown"
        # Extract first word/name from student profile
        name_match = re.search(r'(\w+)', str(student_profile))
        return name_match.group(1) if name_match else "Unknown"
    
    def _reconstruct_conversations(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct complete conversations by grouping rows with the same conversation_id
        Preserves original order from CSV
        """
        conversations = []
        
        # Get unique conversation IDs in order of first appearance
        conversation_order = []
        seen = set()
        for conv_id in df_raw['conversation_id']:
            if conv_id not in seen:
                conversation_order.append(conv_id)
                seen.add(conv_id)
        
        # Process conversations in original order
        for conversation_id in conversation_order:
            group = df_raw[df_raw['conversation_id'] == conversation_id]
            
            # Sort by index to maintain utterance order within conversation
            group = group.sort_index()
            
            # Collect all utterances for this conversation
            utterances = []
            teacher_moves = []
            
            for _, row in group.iterrows():
                utterance = str(row['conversation'])
                
                # Remove |EOM| marker
                utterance = utterance.replace('|EOM|', '').strip()
                
                if utterance:
                    # Check if it's a teacher utterance and extract move
                    if utterance.startswith('Teacher:'):
                        # Extract the move label if present
                        move_match = re.match(r'Teacher:\s*\((\w+)\)', utterance)
                        if move_match:
                            move = move_match.group(1).lower()
                            if move in self.teacher_moves:
                                teacher_moves.append(move)
                        
                        # Clean the utterance (remove label)
                        clean_utterance = re.sub(r'\((generic|focus|probing|telling)\)', '', utterance)
                        utterances.append(clean_utterance.strip())
                    else:
                        # Student utterance - just add as is
                        utterances.append(utterance)
            
            # Join all utterances with newlines to form the complete conversation
            full_conversation = '\n'.join(utterances)
            
            # Get metadata from first row of group
            first_row = group.iloc[0]
            
            # Create a single row for this complete conversation
            conversation_data = {
                'qid': first_row['qid'],
                'conversation_id': conversation_id,
                'student_name': first_row['student_name'],
                'question': first_row['question'],
                'ground_truth': first_row['ground_truth'],  # Math solution
                'student_incorrect_solution': first_row.get('student_incorrect_solution', ''),
                'student_profile': first_row.get('student_profile', ''),
                'cleaned_conversation': full_conversation,
                'ground_truth_moves': teacher_moves,
                'num_teacher_moves': len(teacher_moves),
                'actual_teacher_count': full_conversation.count('Teacher:'),
                'actual_student_count': full_conversation.count('Student:'),
                'total_utterances': len(group)
            }
            
            conversations.append(conversation_data)
        
        # Create DataFrame from reconstructed conversations
        df_conversations = pd.DataFrame(conversations)
        
        # Verify reconstruction
        mismatches = df_conversations[
            df_conversations['num_teacher_moves'] != df_conversations['actual_teacher_count']
        ]
        if len(mismatches) > 0:
            print(f"\n⚠️  WARNING: {len(mismatches)} conversations have mismatch between extracted moves and teacher count")
            for _, row in mismatches.head(3).iterrows():
                print(f"  Conv {row['conversation_id']}: {row['num_teacher_moves']} moves vs {row['actual_teacher_count']} teachers")
        
        # Show distribution of students per question
        students_per_qid = df_conversations.groupby('qid')['student_name'].nunique()
        print(f"\nStudents per question (QID):")
        print(f"  Average: {students_per_qid.mean():.1f}")
        print(f"  Min: {students_per_qid.min()}")
        print(f"  Max: {students_per_qid.max()}")
        
        return df_conversations
    
    def _print_move_statistics(self, df: pd.DataFrame):
        """Print statistics about teacher move distribution"""
        print("\nTeacher Move Statistics:")
        
        # Count total moves across all conversations
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
        
        # Show average moves per conversation
        avg_moves = df['num_teacher_moves'].mean()
        print(f"\nAverage teacher moves per conversation: {avg_moves:.1f}")
        
        # Show sample conversations
        print(f"\nTotal complete conversations: {len(df)}")
        
        # Show how many unique questions and students
        unique_qids = df['qid'].nunique()
        unique_students = df['student_name'].nunique()
        print(f"Unique questions (QIDs): {unique_qids}")
        print(f"Unique students: {unique_students}")
        
        print("\nSample conversation structure (first 3):")
        for idx, row in df.head(3).iterrows():
            print(f"\n  Conversation {row['conversation_id']}:")
            print(f"    QID: {row['qid']}, Student: {row['student_name']}")
            print(f"    Teacher moves: {row['ground_truth_moves']}")
            print(f"    Number of moves: {row['num_teacher_moves']}")
            print(f"    Total utterances: {row['total_utterances']}")
            
            # Show first few lines of conversation
            lines = row['cleaned_conversation'].split('\n')
            print(f"    First few lines:")
            for i, line in enumerate(lines[:6]):
                if line:
                    print(f"      {i+1}. {line[:80]}...")
            if len(lines) > 6:
                print(f"      ... {len(lines)-6} more lines")
    
    def get_conversation_by_id(self, df: pd.DataFrame, conversation_id: str) -> Dict:
        """Get a specific conversation by ID"""
        conv_df = df[df['conversation_id'] == conversation_id]
        
        if conv_df.empty:
            return None
        
        row = conv_df.iloc[0]
        return {
            'conversation_id': row['conversation_id'],
            'qid': row['qid'],
            'student_name': row['student_name'],
            'question': row['question'],
            'ground_truth': row['ground_truth'],
            'student_incorrect_solution': row.get('student_incorrect_solution', ''),
            'student_profile': row.get('student_profile', ''),
            'conversation': row['cleaned_conversation'],
            'teacher_moves': row['ground_truth_moves']
        }


# Compatibility functions for existing evaluation code
def load_and_preprocess_mathdial_data(data_path: str,  # FIXED: No default
                                      sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess MathDial data (compatibility function)"""
    loader = MathDialDataLoader(data_path)
    return loader.load_data(sample_size)