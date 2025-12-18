"""
NumPy Utilities for Sequence Batching, Padding, and Masking
Bottom-up implementation showing data pipeline structure
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SequenceBatcher:
    """
    Utility class for creating sliding-window sequences from time series data
    """
    
    @staticmethod
    def create_sequences(
        features: np.ndarray, 
        labels: np.ndarray, 
        sequence_length: int,
        stride: int = 1,
        prediction_offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding-window sequences from time series data
        
        Args:
            features: Array of shape (n_timesteps, n_features)
            labels: Array of shape (n_timesteps,)
            sequence_length: Length of input sequences
            stride: Step size for sliding window
            prediction_offset: Offset for prediction (0 = predict next timestep)
            
        Returns:
            X: Sequences of shape (n_sequences, sequence_length, n_features)
            y: Labels of shape (n_sequences,)
        """
        if len(features) < sequence_length + prediction_offset:
            return np.array([]), np.array([])
        
        n_samples = len(features) - sequence_length - prediction_offset + 1
        n_features = features.shape[1]
        
        X = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(0, n_samples, stride):
            end_idx = i + sequence_length
            pred_idx = end_idx + prediction_offset - 1
            
            if pred_idx >= len(labels):
                break
                
            X[i // stride] = features[i:end_idx]
            y[i // stride] = labels[pred_idx]
        
        # Only return sequences created with the stride
        actual_samples = (n_samples + stride - 1) // stride
        return X[:actual_samples], y[:actual_samples]
    
    @staticmethod
    def pad_sequences(
        sequences: List[np.ndarray],
        max_length: Optional[int] = None,
        padding_value: float = 0.0,
        padding_side: str = 'pre'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad sequences to the same length
        
        Args:
            sequences: List of sequences, each of shape (seq_len, n_features)
            max_length: Maximum sequence length (None = use longest sequence)
            padding_value: Value to use for padding
            padding_side: 'pre' or 'post' padding
            
        Returns:
            padded_sequences: Array of shape (n_sequences, max_length, n_features)
            sequence_lengths: Array of actual lengths for masking
        """
        if not sequences:
            return np.array([]), np.array([])
        
        # Find max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        n_features = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
        n_sequences = len(sequences)
        
        # Handle 1D sequences
        if len(sequences[0].shape) == 1:
            padded = np.full((n_sequences, max_length), padding_value, dtype=np.float32)
            lengths = np.zeros(n_sequences, dtype=np.int32)
            
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                lengths[i] = seq_len
                
                if padding_side == 'pre':
                    if seq_len <= max_length:
                        padded[i, -seq_len:] = seq
                    else:
                        padded[i] = seq[-max_length:]
                else:  # post
                    if seq_len <= max_length:
                        padded[i, :seq_len] = seq
                    else:
                        padded[i] = seq[:max_length]
            
            return padded, lengths
        
        # Handle 2D sequences (time series)
        padded = np.full((n_sequences, max_length, n_features), padding_value, dtype=np.float32)
        lengths = np.zeros(n_sequences, dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            lengths[i] = min(seq_len, max_length)
            
            if padding_side == 'pre':
                if seq_len <= max_length:
                    padded[i, -seq_len:] = seq
                else:
                    padded[i] = seq[-max_length:]
            else:  # post
                if seq_len <= max_length:
                    padded[i, :seq_len] = seq
                else:
                    padded[i] = seq[:max_length]
        
        return padded, lengths
    
    @staticmethod
    def create_mask(sequence_lengths: np.ndarray, max_length: int) -> np.ndarray:
        """
        Create binary mask for padded sequences
        
        Args:
            sequence_lengths: Array of actual sequence lengths
            max_length: Maximum sequence length
            
        Returns:
            mask: Binary mask of shape (n_sequences, max_length)
                  1 for valid timesteps, 0 for padding
        """
        n_sequences = len(sequence_lengths)
        mask = np.zeros((n_sequences, max_length), dtype=np.float32)
        
        for i, length in enumerate(sequence_lengths):
            mask[i, :length] = 1.0
        
        return mask
    
    @staticmethod
    def batch_patient_sequences(
        patient_data: List[Tuple[np.ndarray, np.ndarray]],
        sequence_length: int,
        max_sequence_length: Optional[int] = None,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create batched sequences from multiple patients
        
        Args:
            patient_data: List of (features, labels) tuples
            sequence_length: Desired sequence length
            max_sequence_length: Maximum length for padding (None = use sequence_length)
            stride: Step size for sliding window
            
        Returns:
            X: Padded sequences of shape (n_total_sequences, max_length, n_features)
            y: Labels of shape (n_total_sequences,)
            mask: Binary mask of shape (n_total_sequences, max_length)
        """
        if max_sequence_length is None:
            max_sequence_length = sequence_length
        
        all_sequences = []
        all_labels = []
        
        for features, labels in patient_data:
            if len(features) == 0:
                continue
            
            # Create sequences for this patient
            X_patient, y_patient = SequenceBatcher.create_sequences(
                features, labels, sequence_length, stride
            )
            
            if len(X_patient) > 0:
                all_sequences.append(X_patient)
                all_labels.append(y_patient)
        
        if not all_sequences:
            return np.array([]), np.array([]), np.array([])
        
        # Flatten sequences from all patients
        sequences_flat = []
        labels_flat = []
        
        for seq_batch, label_batch in zip(all_sequences, all_labels):
            for seq, label in zip(seq_batch, label_batch):
                sequences_flat.append(seq)
                labels_flat.append(label)
        
        # Pad sequences
        X_padded, lengths = SequenceBatcher.pad_sequences(
            sequences_flat, 
            max_length=max_sequence_length,
            padding_side='pre'
        )
        
        # Create mask
        mask = SequenceBatcher.create_mask(lengths, max_sequence_length)
        
        y = np.array(labels_flat, dtype=np.int32)
        
        return X_padded, y, mask
    
    @staticmethod
    def split_sequences(
        X: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split sequences into train/validation/test sets
        
        Args:
            X: Sequences
            y: Labels
            mask: Optional mask
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            random_seed: Random seed for shuffling
            
        Returns:
            (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test)
        """
        np.random.seed(random_seed)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        if mask is not None:
            mask_train = mask[train_idx]
            mask_val = mask[val_idx]
            mask_test = mask[test_idx]
            return (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test)
        else:
            return (X_train, y_train, None), (X_val, y_val, None), (X_test, y_test, None)

