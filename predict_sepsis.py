"""
Inference Script for Sepsis Prediction
Use trained model to predict sepsis probability for new patient data
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from tensorflow import keras

from data_preprocessing import SepsisDataPreprocessor
from sequence_utils import SequenceBatcher


def predict_sepsis(
    patient_file: str,
    # Default to modern Keras format produced by training script
    model_path: str = 'sepsis_lstm_model.keras',
    preprocessor_path: str = 'preprocessor.pkl',
    sequence_length: int = 24,
    threshold: float = 0.5
):
    """
    Predict sepsis probability for a patient
    
    Args:
        patient_file: Path to patient .psv file
        model_path: Path to trained model
        preprocessor_path: Path to saved preprocessor
        sequence_length: Sequence length used during training
        threshold: Probability threshold for binary classification
        
    Returns:
        Dictionary with predictions and probabilities
    """
    # Load model
    print(f"Loading model from {model_path}...")
    # Use keras.saving.load_model for Keras v3 native format
    try:
        model = keras.saving.load_model(model_path)
    except Exception as e:
        print(f"Standard load failed with: {e}")
        print("Retrying with legacy loader (for older .h5 models)...")
        model = keras.models.load_model(model_path, compile=False)
    
    # Load preprocessor
    print(f"Loading preprocessor from {preprocessor_path}...")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load patient data
    print(f"Loading patient data from {patient_file}...")
    df = preprocessor.load_patient_data(patient_file)
    if df is None or df.empty:
        raise ValueError(f"Could not load data from {patient_file}")
    
    # Clean data
    print("Cleaning data...")
    df_clean = preprocessor.clean_data(df)
    
    # Extract features
    features, _ = preprocessor.extract_features(df_clean)
    
    # Normalize features
    print("Normalizing features...")
    features_norm = preprocessor.normalize_features(features, fit=False)
    
    # Create sequences
    print(f"Creating sequences (length={sequence_length})...")
    if len(features_norm) < sequence_length:
        print(f"Warning: Patient has only {len(features_norm)} timesteps, "
              f"need at least {sequence_length}. Padding...")
        # Pad if needed
        padding = np.zeros((sequence_length - len(features_norm), features_norm.shape[1]))
        features_norm = np.vstack([padding, features_norm])
    
    X, _, _ = SequenceBatcher.batch_patient_sequences(
        [(features_norm, np.zeros(len(features_norm)))],
        sequence_length=sequence_length,
        max_sequence_length=sequence_length,
        stride=1
    )
    
    if len(X) == 0:
        raise ValueError("Could not create sequences from patient data")
    
    # Predict
    print("Making predictions...")
    probabilities = model.predict(X, verbose=0)
    predictions = (probabilities >= threshold).astype(int)
    
    # Results
    results = {
        'probabilities': probabilities.flatten().tolist(),
        'predictions': predictions.flatten().tolist(),
        'latest_probability': float(probabilities[-1][0]),
        'latest_prediction': int(predictions[-1][0]),
        'n_sequences': len(X),
        'threshold': threshold
    }
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Predict sepsis probability for a patient'
    )
    
    parser.add_argument(
        'patient_file',
        type=str,
        help='Path to patient .psv file'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='sepsis_lstm_model.h5',
        help='Path to trained model (default: sepsis_lstm_model.h5)'
    )
    
    parser.add_argument(
        '--preprocessor_path',
        type=str,
        default='preprocessor.pkl',
        help='Path to preprocessor (default: preprocessor.pkl)'
    )
    
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=24,
        help='Sequence length (default: 24)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for classification (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.patient_file).exists():
        print(f"Error: Patient file '{args.patient_file}' not found!")
        return
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Please train the model first using main.py")
        return
    
    if not Path(args.preprocessor_path).exists():
        print(f"Error: Preprocessor file '{args.preprocessor_path}' not found!")
        print("Please train the model first using main.py")
        return
    
    # Make prediction
    try:
        results = predict_sepsis(
            args.patient_file,
            args.model_path,
            args.preprocessor_path,
            args.sequence_length,
            args.threshold
        )
        
        # Print results
        print("\n" + "="*70)
        print("SEPSIS PREDICTION RESULTS")
        print("="*70)
        print(f"Patient File:        {args.patient_file}")
        print(f"Number of Sequences: {results['n_sequences']}")
        print(f"Threshold:           {results['threshold']}")
        print(f"\nLatest Prediction:")
        print(f"  Probability: {results['latest_probability']:.4f}")
        print(f"  Prediction:  {'SEPSIS' if results['latest_prediction'] == 1 else 'NO SEPSIS'}")
        print("="*70)
        
        # Show all sequence predictions
        if results['n_sequences'] > 1:
            print("\nAll Sequence Predictions:")
            for i, (prob, pred) in enumerate(zip(results['probabilities'], results['predictions'])):
                status = "SEPSIS" if pred == 1 else "NO SEPSIS"
                print(f"  Sequence {i+1}: {prob:.4f} -> {status}")
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

