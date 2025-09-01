"""
Metrics for MathDial teacher move classification evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import cohen_kappa_score, accuracy_score
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
    
    def calculate_exact_match_accuracy(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Calculate exact match accuracy (all moves in sequence must match)"""
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
    
    def calculate_cohens_kappa(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Cohen's Kappa for move classification"""
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        try:
            kappa = cohen_kappa_score(y_true_flat, y_pred_flat)
            return kappa * 100  # Multiply by 100 as shown in the chart
        except:
            return 0.0
    
    def calculate_krippendorffs_alpha(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Krippendorff's Alpha for reliability"""
        if not HAS_KRIPPENDORFF:
            return 0.0
            
        try:
            data = np.vstack([y_true_binary.flatten(), y_pred_binary.flatten()])
            alpha = krippendorff.alpha(data, level_of_measurement='nominal')
            return (alpha * 100) if not np.isnan(alpha) else 0.0  # Multiply by 100
        except:
            return 0.0
    
    def calculate_icc(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calculate Intraclass Correlation Coefficient"""
        try:
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
    
    def comprehensive_evaluation(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        Perform comprehensive evaluation of teacher move classification
        
        Args:
            y_true: List of true move sequences
            y_pred: List of predicted move sequences
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Convert to binary matrices
        y_true_binary = self.moves_to_binary(y_true)
        y_pred_binary = self.moves_to_binary(y_pred)
        
        # Calculate the 4 main metrics
        accuracy = self.calculate_exact_match_accuracy(y_true, y_pred)
        cohens_kappa = self.calculate_cohens_kappa(y_true_binary, y_pred_binary)
        krippendorffs_alpha = self.calculate_krippendorffs_alpha(y_true_binary, y_pred_binary)
        icc = self.calculate_icc(y_true_binary, y_pred_binary)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'cohens_kappa': cohens_kappa,
            'krippendorffs_alpha': krippendorffs_alpha,
            'icc': icc,
            'num_samples': len(y_true)
        }
        
        return results
    
    def print_results(self, results: Dict, technique_name: str = ""):
        """Print formatted results"""
        if technique_name:
            print(f"\n=== {technique_name} Results ===")
        
        print("MathDial Teacher Move Classification Metrics:")
        print(f"  Accuracy: {results['accuracy']:.1f}%")
        print(f"  Cohen's κ (*100): {results['cohens_kappa']:.1f}")
        print(f"  Krippendorff's α (*100): {results['krippendorffs_alpha']:.1f}")
        print(f"  ICC (*100): {results['icc']:.1f}")
        print(f"  Number of samples: {results['num_samples']}")
        
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


# Compatibility functions
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