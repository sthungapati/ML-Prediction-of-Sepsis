"""
Main Execution Script for Sepsis Prediction Pipeline
End-to-end ML model for predicting sepsis onset
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Set TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("Using CPU")

from train_sepsis_model import train_sepsis_model


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Train Sepsis Prediction LSTM Model'
    )
    
    parser.add_argument(
        '--training_dir',
        type=str,
        default='training_setA/training',
        help='Directory containing training .psv files'
    )
    
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=24,
        help='Length of input sequences in hours (default: 24)'
    )
    
    parser.add_argument(
        '--prediction_horizon',
        type=int,
        default=6,
        help='Hours ahead to predict sepsis (default: 6)'
    )
    
    parser.add_argument(
        '--max_patients',
        type=int,
        default=None,
        help='Maximum number of patients to use (None for all)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--lstm_units',
        type=str,
        default='128,64',
        help='Comma-separated LSTM units (default: 128,64)'
    )
    
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of parallel workers (None for auto)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='sepsis_lstm_model.keras',
        help='Path to save trained model (use .keras format for best compatibility)'
    )
    
    parser.add_argument(
        '--preprocessor_path',
        type=str,
        default='preprocessor.pkl',
        help='Path to save preprocessor'
    )
    
    args = parser.parse_args()
    
    # Parse LSTM units
    lstm_units = [int(x.strip()) for x in args.lstm_units.split(',')]
    
    # Check if training directory exists
    if not Path(args.training_dir).exists():
        print(f"Error: Training directory '{args.training_dir}' not found!")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*70)
    print("SEPSIS PREDICTION MODEL - CONFIGURATION")
    print("="*70)
    print(f"Training Directory:    {args.training_dir}")
    print(f"Sequence Length:       {args.sequence_length} hours")
    print(f"Prediction Horizon:    {args.prediction_horizon} hours")
    print(f"Max Patients:          {args.max_patients or 'All'}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Epochs:                {args.epochs}")
    print(f"LSTM Units:            {lstm_units}")
    print(f"Dropout Rate:          {args.dropout_rate}")
    print(f"Learning Rate:         {args.learning_rate}")
    print(f"Parallel Workers:      {args.n_workers or 'Auto'}")
    print("="*70 + "\n")
    
    # Train model
    try:
        model, preprocessor, results, history = train_sepsis_model(
            training_dir=args.training_dir,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            max_patients=args.max_patients,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lstm_units=lstm_units,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            use_parallel_loading=True,
            n_workers=args.n_workers,
            save_model_path=args.model_path,
            save_preprocessor_path=args.preprocessor_path
        )
        
        print("\n✓ Training completed successfully!")
        print(f"✓ Model saved to: {args.model_path}")
        print(f"✓ Preprocessor saved to: {args.preprocessor_path}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

