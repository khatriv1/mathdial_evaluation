"""
Metrics for MathDial teacher move classification evaluation
FIXED VERSION: Uses position-by-position accuracy as the main metric
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import logging

# Try to import krippendorff
try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    logging.warning("krippendorff package not found. Install with: pip install krippendorff")

class MathDialMetrics:
    """Metrics for MathDial teacher move classification"""
    
    def __init__(self, teacher_moves: List[str] = None):
        self.teacher_moves = teacher_moves or ['generic', 'focus', 'probing', 'telling']
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_by_position_accuracy(self, y_true: List[List[str]], y_pred: List[List[str]]) -> dict:
        """
        Calculate position-by-position accuracy (THE MAIN METRIC)
        Each position is evaluated independently
        
        Example:
        Ground truth: ['probing', 'focus', 'telling', 'generic']
        Predicted:    ['focus', 'probing', 'telling', 'generic']
        Position 1: probing vs focus → ✗
        Position 2: focus vs probing → ✗  
        Position 3: telling vs telling → ✓
        Position 4: generic vs generic → ✓
        Accuracy = 2/4 = 50%
        """
        total_correct = 0
        total_positions = 0
        per_conversation_accuracy = []
        detailed_results = []
        
        for conv_idx, (true_moves, pred_moves) in enumerate(zip(y_true, y_pred)):
            # For this conversation, count position matches
            conversation_correct = 0
            conversation_total = len(true_moves)
            
            # Compare position by position
            min_len = min(len(true_moves), len(pred_moves))
            position_results = []
            
            for i in range(min_len):
                if true_moves[i] == pred_moves[i]:
                    conversation_correct += 1
                    total_correct += 1
                    position_results.append({'position': i+1, 'correct': True})
                else:
                    position_results.append({'position': i+1, 'correct': False})
                total_positions += 1
            
            # Handle length mismatch (penalize missing/extra predictions)
            if len(true_moves) > len(pred_moves):
                # Missing predictions
                for i in range(len(pred_moves), len(true_moves)):
                    total_positions += 1
                    position_results.append({'position': i+1, 'correct': False, 'missing': True})
            elif len(pred_moves) > len(true_moves):
                # Extra predictions (don't count towards total_positions)
                for i in range(len(true_moves), len(pred_moves)):
                    position_results.append({'position': i+1, 'correct': False, 'extra': True})
            
            # Calculate per-conversation accuracy
            conv_accuracy = (conversation_correct / conversation_total) * 100 if conversation_total > 0 else 0
            per_conversation_accuracy.append(conv_accuracy)
            
            detailed_results.append({
                'conversation_idx': conv_idx,
                'correct_positions': conversation_correct,
                'total_positions': conversation_total,
                'accuracy': conv_accuracy,
                'position_results': position_results
            })
        
        # Overall position-by-position accuracy
        overall_accuracy = (total_correct / total_positions) * 100 if total_positions > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_positions': total_positions,
            'per_conversation_accuracy': per_conversation_accuracy,
            'mean_conversation_accuracy': np.mean(per_conversation_accuracy) if per_conversation_accuracy else 0,
            'detailed_results': detailed_results
        }
    
    def calculate_exact_match_accuracy(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """
        Calculate exact match accuracy (entire sequence must match)
        This is what was being used before - too strict!
        """
        matches = 0
        total = len(y_true)
        
        for true_moves, pred_moves in zip(y_true, y_pred):
            # Check if sequences are exactly the same
            if len(true_moves) == len(pred_moves) and all(t == p for t, p in zip(true_moves, pred_moves)):
                matches += 1
        
        return (matches / total) * 100 if total > 0 else 0.0
    
    def moves_to_binary(self, moves_lists: List[List[str]]) -> np.ndarray:
        """Convert move sequences to binary matrix for metrics calculation"""
        max_len = max(len(moves) for moves in moves_lists) if moves_lists else 1
        binary_matrix = np.zeros((len(moves_lists), max_len * len(self.teacher_moves)))
        
        for i, moves in enumerate(moves_lists):
            for j, move in enumerate(moves):
                if move in self.teacher_moves:
                    move_idx = self.teacher_moves.index(move)
                    binary_matrix[i, j * len(self.teacher_moves) + move_idx] = 1
        
        return binary_matrix
    
    def flatten_for_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Tuple[List[str], List[str]]:
        """
        Flatten the lists for sklearn metrics calculation
        Aligns predictions position by position
        """
        y_true_flat = []
        y_pred_flat = []
        
        for true_moves, pred_moves in zip(y_true, y_pred):
            min_len = min(len(true_moves), len(pred_moves))
            
            # Add aligned predictions
            for i in range(min_len):
                y_true_flat.append(true_moves[i])
                y_pred_flat.append(pred_moves[i])
            
            # Handle length mismatch
            if len(true_moves) > len(pred_moves):
                # Missing predictions - add as 'generic' (default)
                for i in range(len(pred_moves), len(true_moves)):
                    y_true_flat.append(true_moves[i])
                    y_pred_flat.append('generic')  # Default prediction for missing
        
        return y_true_flat, y_pred_flat
    
    def calculate_cohens_kappa(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Calculate Cohen's Kappa for move classification"""
        # First convert to binary matrices
        y_true_binary = self.moves_to_binary(y_true)
        y_pred_binary = self.moves_to_binary(y_pred)
        
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        try:
            kappa = cohen_kappa_score(y_true_flat, y_pred_flat)
            return kappa * 100  # Multiply by 100 as shown in the chart
        except:
            return 0.0
    
    def calculate_krippendorffs_alpha(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Calculate Krippendorff's Alpha for reliability"""
        if not HAS_KRIPPENDORFF:
            return 0.0
            
        try:
            # Convert to binary matrices
            y_true_binary = self.moves_to_binary(y_true)
            y_pred_binary = self.moves_to_binary(y_pred)
            
            data = np.vstack([y_true_binary.flatten(), y_pred_binary.flatten()])
            alpha = krippendorff.alpha(data, level_of_measurement='nominal')
            return (alpha * 100) if not np.isnan(alpha) else 0.0  # Multiply by 100
        except:
            return 0.0
    
    def calculate_icc(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Calculate Intraclass Correlation Coefficient"""
        try:
            # Convert to binary matrices
            y_true_binary = self.moves_to_binary(y_true)
            y_pred_binary = self.moves_to_binary(y_pred)
            
            correlations = []
            
            # Calculate correlation for each position
            min_len = min(y_true_binary.shape[1], y_pred_binary.shape[1])
            for i in range(min_len):
                true_col = y_true_binary[:, i]
                pred_col = y_pred_binary[:, i] if i < y_pred_binary.shape[1] else np.zeros_like(true_col)
                
                if np.var(true_col) > 0 and np.var(pred_col) > 0:
                    corr, _ = pearsonr(true_col, pred_col)
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            icc = np.mean(correlations) if correlations else 0.0
            return icc * 100  # Multiply by 100
        except:
            return 0.0
    
    def calculate_confusion_matrix(self, y_true: List[List[str]], y_pred: List[List[str]]) -> np.ndarray:
        """Calculate confusion matrix for detailed analysis"""
        # Flatten the predictions
        y_true_flat, y_pred_flat = self.flatten_for_metrics(y_true, y_pred)
        return confusion_matrix(y_true_flat, y_pred_flat, labels=self.teacher_moves)
    
    def comprehensive_evaluation(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        Perform comprehensive evaluation of teacher move classification
        
        IMPORTANT: The main 'accuracy' metric is now position-by-position accuracy,
        NOT exact sequence match!
        
        Args:
            y_true: List of true move sequences
            y_pred: List of predicted move sequences
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Calculate position-by-position accuracy (THE MAIN METRIC)
        position_metrics = self.calculate_position_by_position_accuracy(y_true, y_pred)
        
        # Also calculate exact match for comparison
        exact_match = self.calculate_exact_match_accuracy(y_true, y_pred)
        
        # Calculate other metrics using binary matrices
        cohens_kappa = self.calculate_cohens_kappa(y_true, y_pred)
        krippendorffs_alpha = self.calculate_krippendorffs_alpha(y_true, y_pred)
        icc = self.calculate_icc(y_true, y_pred)
        
        # Calculate confusion matrix
        conf_matrix = self.calculate_confusion_matrix(y_true, y_pred)
        
        # Calculate per-category accuracy
        per_category = {}
        for i, category in enumerate(self.teacher_moves):
            true_positives = conf_matrix[i, i]
            total_actual = conf_matrix[i, :].sum()
            accuracy = (true_positives / total_actual * 100) if total_actual > 0 else 0
            per_category[category] = accuracy
        
        # Compile results - USE POSITION ACCURACY AS MAIN METRIC
        results = {
            # Main accuracy metric (position-by-position) - THIS IS THE KEY CHANGE!
            'accuracy': position_metrics['overall_accuracy'],  # THIS IS NOW POSITION ACCURACY
            
            # Detailed position metrics
            'position_accuracy': position_metrics['overall_accuracy'],
            'total_correct': position_metrics['total_correct'],
            'total_positions': position_metrics['total_positions'],
            'mean_conversation_accuracy': position_metrics['mean_conversation_accuracy'],
            'per_conversation_accuracy': position_metrics['per_conversation_accuracy'],
            
            # Exact match (for reference only - this was the old way)
            'exact_match_accuracy': exact_match,
            
            # Agreement metrics
            'cohens_kappa': cohens_kappa,
            'krippendorffs_alpha': krippendorffs_alpha,
            'icc': icc,
            
            # Additional info
            'num_samples': len(y_true),
            'confusion_matrix': conf_matrix.tolist(),
            'per_category_accuracy': per_category
        }
        
        return results
    
    def print_results(self, results: Dict, technique_name: str = ""):
        """Print formatted results"""
        if technique_name:
            print(f"\n=== {technique_name} Results ===")
        
        print("MathDial Teacher Move Classification Metrics:")
        
        # Main metric - position-by-position accuracy
        print(f"  Accuracy: {results['accuracy']:.1f}%")
        print(f"  Cohen's κ (*100): {results['cohens_kappa']:.1f}")
        print(f"  Krippendorff's α (*100): {results['krippendorffs_alpha']:.1f}")
        print(f"  ICC (*100): {results['icc']:.1f}")
        print(f"  Number of samples: {results['num_samples']}")
        
        # Additional details
        if 'total_correct' in results:
            print(f"\n  Position-by-position details:")
            print(f"    Correct positions: {results['total_correct']}/{results['total_positions']}")
            print(f"    Mean per-conversation accuracy: {results['mean_conversation_accuracy']:.1f}%")
        
        if 'exact_match_accuracy' in results:
            print(f"\n  Exact sequence match: {results['exact_match_accuracy']:.1f}%")
        
        # Interpretation
        kappa = results['cohens_kappa'] / 100  # Convert back to 0-1 scale
        if kappa >= 0.8:
            agreement = "Almost Perfect Agreement"
        elif kappa >= 0.6:
            agreement = "Substantial Agreement"
        elif kappa >= 0.4:
            agreement = "Moderate Agreement"
        elif kappa >= 0.2:
            agreement = "Fair Agreement"
        else:
            agreement = "Slight Agreement"
        
        print(f"\n  Agreement Level: {agreement}")


# Compatibility functions for existing evaluation code
def calculate_mathdial_metrics(y_true: List[List[str]], y_pred: List[List[str]], 
                               teacher_moves: List[str] = None) -> Dict:
    """Calculate MathDial metrics (compatibility function)"""
    metrics = MathDialMetrics(teacher_moves)
    return metrics.comprehensive_evaluation(y_true, y_pred)

def print_mathdial_results(results: Dict, technique_name: str = ""):
    """Print MathDial results (compatibility function)"""
    metrics = MathDialMetrics()
    metrics.print_results(results, technique_name)

def evaluate_teacher_moves(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
    """Evaluate teacher move predictions (compatibility function)"""
    metrics = MathDialMetrics()
    return metrics.comprehensive_evaluation(y_true, y_pred)