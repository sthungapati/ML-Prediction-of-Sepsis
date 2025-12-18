"""
TensorFlow/Keras LSTM Model for Sepsis Prediction
Stacked LSTM layers with masking and dense sigmoid head
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import numpy as np


class SepsisLSTMModel:
    """
    Stacked LSTM model for predicting sepsis onset probability
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: list = [128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        use_masking: bool = True
    ):
        """
        Args:
            input_shape: (sequence_length, n_features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate between layers
            learning_rate: Learning rate for optimizer
            use_masking: Whether to use masking layer for padded sequences
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_masking = use_masking
        self.model = None
        
    def build_model(self) -> keras.Model:
        """Build the stacked LSTM model"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='sequence_input')
        x = inputs
        
        # Masking layer for padded sequences
        if self.use_masking:
            x = layers.Masking(mask_value=0.0)(x)
        
        # Stacked LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)  # Last LSTM doesn't return sequences
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                name=f'lstm_{i+1}'
            )(x)
            
            # Batch normalization
            if return_sequences:
                x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Dense layers before output
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name='dropout_2')(x)
        
        # Output layer: sigmoid for binary classification (probability)
        outputs = layers.Dense(1, activation='sigmoid', name='sepsis_probability')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='SepsisLSTM')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet. Call build_model() first."
        return self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 50,
        verbose: int = 1,
        callbacks: Optional[list] = None
    ) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            verbose: Verbosity level
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    # Save in Keras v3 native format for better compatibility
                    'best_sepsis_model.keras',
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            shuffle=True
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sepsis probability
        
        Args:
            X: Input sequences
            
        Returns:
            Probabilities of shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        probabilities = self.model.predict(X, verbose=0)
        return probabilities.flatten()
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary classes
        
        Args:
            X: Input sequences
            threshold: Probability threshold for classification
            
        Returns:
            Binary predictions
        """
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def save_model(self, filepath: str):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

