"""
Training Script for Sepsis Prediction Model
Includes parallel processing support for data loading and training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import pickle
import time
from typing import List, Tuple

from data_preprocessing import SepsisDataPreprocessor
from sequence_utils import SequenceBatcher
from sepsis_lstm_model import SepsisLSTMModel
from model_evaluation import ModelEvaluator


class ParallelDataLoader:
    """Parallel data loading for faster preprocessing"""
    
    @staticmethod
    def load_patient_batch(args):
        """Load and process a batch of patients (for multiprocessing)"""
        file_paths, prediction_horizon = args
        preprocessor = SepsisDataPreprocessor(prediction_horizon=prediction_horizon)
        
        patient_data = []
        for file_path in file_paths:
            df = preprocessor.load_patient_data(file_path)
            if df is None or df.empty:
                continue
            
            df_clean = preprocessor.clean_data(df)
            if df_clean is None or df_clean.empty:
                continue
            
            df_labeled = preprocessor.create_labels(df_clean)
            features, labels = preprocessor.extract_features(df_labeled)
            
            if features is not None and len(features) > 0:
                patient_data.append((features, labels))
        
        return patient_data


def load_data_parallel(
    data_dir: str,
    prediction_horizon: int = 6,
    max_patients: int = None,
    n_workers: int = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load patient data in parallel
    
    Args:
        data_dir: Directory containing .psv files
        prediction_horizon: Hours ahead to predict
        max_patients: Maximum patients to load
        n_workers: Number of parallel workers (None = use all CPUs)
        
    Returns:
        List of (features, labels) tuples
    """
    data_dir = Path(data_dir)
    psv_files = list(data_dir.glob('*.psv'))
    
    if max_patients:
        psv_files = psv_files[:max_patients]
    
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Limit to 8 to avoid memory issues
    
    # Split files into batches for parallel processing
    batch_size = max(1, len(psv_files) // n_workers)
    file_batches = [psv_files[i:i+batch_size] for i in range(0, len(psv_files), batch_size)]
    
    print(f"Loading {len(psv_files)} patient files using {n_workers} workers...")
    
    # Process in parallel
    with Pool(n_workers) as pool:
        args = [(batch, prediction_horizon) for batch in file_batches]
        results = pool.map(ParallelDataLoader.load_patient_batch, args)
    
    # Flatten results
    all_patient_data = []
    for result in results:
        all_patient_data.extend(result)
    
    print(f"Successfully loaded {len(all_patient_data)} patients")
    return all_patient_data


def normalize_all_patients(
    patient_data: List[Tuple[np.ndarray, np.ndarray]],
    fit_scaler: bool = True
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], SepsisDataPreprocessor]:
    """
    Normalize features across all patients
    
    Args:
        patient_data: List of (features, labels) tuples
        fit_scaler: Whether to fit scaler (True for training, False for inference)
        
    Returns:
        Normalized patient data and preprocessor with fitted scaler
    """
    preprocessor = SepsisDataPreprocessor()
    
    if fit_scaler:
        # Collect all features to fit scaler
        print("Fitting feature scaler...")
        all_features = np.vstack([features for features, _ in patient_data])
        preprocessor.normalize_features(all_features, fit=True)
    
    # Normalize each patient's features
    print("Normalizing patient features...")
    normalized_data = []
    for features, labels in patient_data:
        features_norm = preprocessor.normalize_features(features, fit=False)
        normalized_data.append((features_norm, labels))
    
    return normalized_data, preprocessor


def train_sepsis_model(
    training_dir: str,
    sequence_length: int = 24,
    prediction_horizon: int = 6,
    max_patients: int = None,
    batch_size: int = 32,
    epochs: int = 50,
    lstm_units: list = [128, 64],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    use_parallel_loading: bool = True,
    n_workers: int = None,
    # Use modern Keras v3 native format instead of legacy H5
    save_model_path: str = 'sepsis_lstm_model.keras',
    save_preprocessor_path: str = 'preprocessor.pkl'
):
    """
    Main training function
    
    Args:
        training_dir: Directory containing training .psv files
        sequence_length: Length of input sequences (hours)
        prediction_horizon: Hours ahead to predict sepsis
        max_patients: Maximum patients to use (None for all)
        batch_size: Training batch size
        epochs: Number of training epochs
        lstm_units: LSTM layer units
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        use_parallel_loading: Use parallel data loading
        n_workers: Number of parallel workers
        save_model_path: Path to save trained model
        save_preprocessor_path: Path to save preprocessor
    """
    print("="*70)
    print("SEPSIS PREDICTION MODEL TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    # Step 1: Load data
    print("\n[Step 1/5] Loading and preprocessing data...")
    if use_parallel_loading:
        patient_data = load_data_parallel(
            training_dir,
            prediction_horizon=prediction_horizon,
            max_patients=max_patients,
            n_workers=n_workers
        )
    else:
        preprocessor = SepsisDataPreprocessor(prediction_horizon=prediction_horizon)
        patient_data = preprocessor.process_all_patients(training_dir, max_patients=max_patients)
    
    if len(patient_data) == 0:
        raise ValueError("No patient data loaded!")
    
    # Step 2: Normalize features
    print("\n[Step 2/5] Normalizing features...")
    patient_data_norm, preprocessor = normalize_all_patients(patient_data, fit_scaler=True)
    
    # Save preprocessor for inference
    with open(save_preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {save_preprocessor_path}")
    
    # Step 3: Create sequences
    print("\n[Step 3/5] Creating sequences...")
    X, y, mask = SequenceBatcher.batch_patient_sequences(
        patient_data_norm,
        sequence_length=sequence_length,
        max_sequence_length=sequence_length,
        stride=1
    )
    
    print(f"Created {len(X)} sequences")
    print(f"Sequence shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Step 4: Split data
    print("\n[Step 4/5] Splitting data into train/val/test sets...")
    (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test) = \
        SequenceBatcher.split_sequences(X, y, mask, train_ratio=0.7, val_ratio=0.15)
    
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Step 5: Build and train model
    print("\n[Step 5/5] Building and training model...")
    input_shape = (sequence_length, X_train.shape[2])
    
    model = SepsisLSTMModel(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        use_masking=True
    )
    
    model.build_model()
    print("\nModel Architecture:")
    model.model.summary()
    
    # Train model
    print("\nStarting training...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    # Save model
    model.save_model(save_model_path)
    
    # Step 6: Evaluate model
    print("\n[Evaluation] Evaluating model performance...")
    evaluator = ModelEvaluator(model.model, threshold=0.5)
    results = evaluator.comprehensive_evaluation(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        save_plots=True
    )
    
    # Training summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Model saved to: {save_model_path}")
    print(f"Preprocessor saved to: {save_preprocessor_path}")
    print("="*70)
    
    return model, preprocessor, results, history


if __name__ == "__main__":
    # Configuration
    config = {
        'training_dir': 'training_setA/training',  # Change to your training directory
        'sequence_length': 24,  # 24 hours of history
        'prediction_horizon': 6,  # Predict 6 hours ahead
        'max_patients': 5000,  # Limit for faster testing (None for all)
        'batch_size': 32,
        'epochs': 50,
        'lstm_units': [128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'use_parallel_loading': True,
        'n_workers': None,  # Auto-detect
    }
    
    # Train model
    model, preprocessor, results, history = train_sepsis_model(**config)

