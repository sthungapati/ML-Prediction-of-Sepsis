"""
Model Evaluation and Verification Module
Includes comprehensive metrics and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation and verification"""
    
    def __init__(self, model, threshold: float = 0.5):
        """
        Args:
            model: Trained Keras model
            threshold: Classification threshold
        """
        self.model = model
        self.threshold = threshold
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }
        
        # Calculate PR-AUC
        if len(np.unique(y_true)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        else:
            metrics['pr_auc'] = 0.0
        
        if verbose:
            print("\n" + "="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
            print("="*50)
        
        return metrics
    
    def confusion_matrix_plot(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        save_path: str = None
    ):
        """Plot confusion matrix"""
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Sepsis', 'Sepsis'],
                   yticklabels=['No Sepsis', 'Sepsis'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # In script mode we avoid blocking on GUI windows
        plt.close()
    
    def roc_curve_plot(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        save_path: str = None
    ):
        """Plot ROC curve"""
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        
        if len(np.unique(y_true)) < 2:
            print("Cannot plot ROC curve: need both classes in test set")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def precision_recall_curve_plot(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        save_path: str = None
    ):
        """Plot Precision-Recall curve"""
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        
        if len(np.unique(y_true)) < 2:
            print("Cannot plot PR curve: need both classes in test set")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def classification_report_print(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ):
        """Print detailed classification report"""
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, 
                                   target_names=['No Sepsis', 'Sepsis']))
        print("="*50)
    
    def comprehensive_evaluation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_plots: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation on all datasets
        
        Returns:
            Dictionary with metrics for train, val, and test sets
        """
        results = {}
        
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Evaluate on training set
        print("\n[TRAINING SET]")
        results['train'] = self.evaluate(X_train, y_train, verbose=True)
        
        # Evaluate on validation set
        print("\n[VALIDATION SET]")
        results['val'] = self.evaluate(X_val, y_val, verbose=True)
        
        # Evaluate on test set
        print("\n[TEST SET]")
        results['test'] = self.evaluate(X_test, y_test, verbose=True)
        
        # Print classification report
        print("\n[TEST SET - Classification Report]")
        self.classification_report_print(X_test, y_test)
        
        # Generate plots
        if save_plots:
            print("\n[Generating evaluation plots...]")
            self.confusion_matrix_plot(X_test, y_test, 'confusion_matrix.png')
            self.roc_curve_plot(X_test, y_test, 'roc_curve.png')
            self.precision_recall_curve_plot(X_test, y_test, 'pr_curve.png')
        
        return results

