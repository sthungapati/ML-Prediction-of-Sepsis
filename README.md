# Sepsis Prediction Model - LSTM-based Clinical Data Analysis

A comprehensive end-to-end machine learning pipeline for predicting sepsis onset in ICU patients using Long Short-Term Memory (LSTM) neural networks.

## Overview

This project implements a supervised learning model that predicts the probability of sepsis onset within the next 6 hours using clinical time-series data. The model uses stacked LSTM layers with masking to handle variable-length sequences and missing data.

## Features

- **Data Preprocessing**: Comprehensive cleaning and normalization of clinical data
- **Sequence Batching**: NumPy-based utilities for sliding-window sequence creation, padding, and masking
- **LSTM Model**: Stacked LSTM architecture with masking layer and sigmoid output for probability prediction
- **Parallel Processing**: Multi-core data loading and training acceleration
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, PR-AUC, confusion matrix, and classification reports
- **Mathematical Methods**: 
  - Z-score normalization
  - Forward/backward filling for missing values
  - Sliding window sequence generation
  - Binary cross-entropy loss with sigmoid activation
  - Adam optimizer with learning rate scheduling

## Project Structure

```
.
├── data_preprocessing.py      # Data cleaning and feature extraction
├── sequence_utils.py          # NumPy utilities for sequence batching
├── sepsis_lstm_model.py        # TensorFlow/Keras LSTM model
├── model_evaluation.py         # Model verification and metrics
├── train_sepsis_model.py       # Training script with parallel processing
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the training data in the following structure:
```
training_setA/
  └── training/
      ├── p000001.psv
      ├── p000002.psv
      └── ...
```

## Usage

### Basic Training

```bash
python main.py --training_dir training_setA/training
```

### Advanced Training with Custom Parameters

```bash
python main.py \
  --training_dir training_setA/training \
  --sequence_length 24 \
  --prediction_horizon 6 \
  --max_patients 10000 \
  --batch_size 64 \
  --epochs 100 \
  --lstm_units 256,128,64 \
  --dropout_rate 0.4 \
  --learning_rate 0.0001 \
  --n_workers 8
```

### Using Training Script Directly

```python
from train_sepsis_model import train_sepsis_model

model, preprocessor, results, history = train_sepsis_model(
    training_dir='training_setA/training',
    sequence_length=24,
    prediction_horizon=6,
    max_patients=5000,
    batch_size=32,
    epochs=50
)
```

## Model Architecture

### Input
- **Shape**: `(batch_size, sequence_length, n_features)`
- **Features**: 40+ clinical variables (vitals, lab values, demographics)
- **Sequence Length**: 24 hours (configurable)

### Architecture
1. **Input Layer**: Accepts variable-length sequences
2. **Masking Layer**: Handles padded sequences
3. **Stacked LSTM Layers**: 
   - LSTM 1: 128 units (returns sequences)
   - Batch Normalization
   - LSTM 2: 64 units (returns final state)
4. **Dense Layers**:
   - Dense 1: 64 units (ReLU)
   - Dropout (30%)
   - Batch Normalization
   - Dense 2: 32 units (ReLU)
   - Dropout (15%)
5. **Output Layer**: 1 unit (Sigmoid) - Probability of sepsis

### Training
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate scheduling
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Metrics**: Accuracy, Precision, Recall, AUC

## Data Pipeline

### 1. Data Loading
- Loads `.psv` (pipe-separated values) files for each patient
- Each file contains time-series clinical data

### 2. Data Cleaning
- Forward fill missing values (carry forward last known value)
- Backward fill remaining NaN values
- Fill with median for remaining missing values
- Handle categorical features (Gender, Unit1, Unit2)

### 3. Label Creation
- Creates binary labels: 1 if sepsis occurs within next 6 hours, else 0
- Uses sliding window approach to check future timesteps

### 4. Feature Normalization
- Z-score normalization: `(x - mean) / std`
- Fitted on training data, applied to all sets

### 5. Sequence Generation
- Creates sliding-window sequences of fixed length
- Pads shorter sequences with zeros
- Generates binary masks for valid timesteps

### 6. Data Splitting
- Training: 70%
- Validation: 15%
- Test: 15%

## Model Verification

The evaluation module provides:

1. **Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
   - PR-AUC

2. **Visualizations**:
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve

3. **Reports**:
   - Classification Report
   - Per-class metrics

## Parallel Processing

The pipeline supports parallel processing for:

1. **Data Loading**: Multi-process loading of patient files
2. **Training**: TensorFlow automatically uses available GPUs
3. **Batch Processing**: Efficient batch creation and padding

## Mathematical Methods

### Normalization
```
z = (x - μ) / σ
```
where μ is the mean and σ is the standard deviation.

### LSTM Cell
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # Hidden state
```

### Loss Function (Binary Cross-Entropy)
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Output Activation (Sigmoid)
```
P(sepsis) = 1 / (1 + e^(-z))
```

## Performance Optimization

1. **Parallel Data Loading**: Uses multiprocessing for faster data preprocessing
2. **GPU Acceleration**: Automatically uses GPU if available
3. **Batch Processing**: Efficient batching with masking
4. **Early Stopping**: Prevents overfitting and reduces training time
5. **Learning Rate Scheduling**: Adaptive learning rate reduction

## Output Files

After training, the following files are generated:

- `sepsis_lstm_model.keras`: Trained Keras model (Keras v3 native format)
- `preprocessor.pkl`: Fitted preprocessor for inference
- `best_sepsis_model.keras`: Best model checkpoint (highest validation AUC)
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `pr_curve.png`: Precision-Recall curve plot

## Inference

To use the trained model for predictions:

```python
import pickle
import numpy as np
from tensorflow import keras
from data_preprocessing import SepsisDataPreprocessor
from sequence_utils import SequenceBatcher

# Load model and preprocessor (Keras v3 format)
model = keras.saving.load_model('sepsis_lstm_model.keras')
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load and preprocess new patient data
df = preprocessor.load_patient_data('new_patient.psv')
df_clean = preprocessor.clean_data(df)
features, _ = preprocessor.extract_features(df_clean)
features_norm = preprocessor.normalize_features(features, fit=False)

# Create sequences
X, _, _ = SequenceBatcher.batch_patient_sequences(
    [(features_norm, np.zeros(len(features_norm)))],
    sequence_length=24
)

# Predict
probabilities = model.predict(X)
print(f"Sepsis probability: {probabilities[0][0]:.4f}")
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.21+
- Pandas 1.3+
- scikit-learn 1.0+
- matplotlib 3.5+
- seaborn 0.11+

## Notes

- The model uses masking to handle variable-length sequences and missing data
- Training time depends on dataset size and hardware (GPU recommended)
- For large datasets, consider using `max_patients` parameter for initial testing
- Model performance improves with more training data

## License

This project is for educational and research purposes.

