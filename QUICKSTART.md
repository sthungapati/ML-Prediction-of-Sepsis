# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Train the Model

```bash
python main.py --training_dir training_setA/training --max_patients 1000
```

This will:
- Load and preprocess patient data
- Create sequences
- Train the LSTM model
- Evaluate performance
- Save the model and preprocessor

### 2. Make Predictions

```bash
python predict_sepsis.py training_setA/training/p000001.psv --model_path sepsis_lstm_model.keras
```

## Configuration Examples

### Small Dataset (Testing)
```bash
python main.py \
  --training_dir training_setA/training \
  --max_patients 500 \
  --sequence_length 12 \
  --epochs 20 \
  --batch_size 16
```

### Full Dataset (Production)
```bash
python main.py \
  --training_dir training_setA/training \
  --sequence_length 24 \
  --prediction_horizon 6 \
  --batch_size 64 \
  --epochs 100 \
  --lstm_units 256,128,64 \
  --dropout_rate 0.4 \
  --learning_rate 0.0001
```

### Using Both Training Sets
```bash
# Train on set A
python main.py --training_dir training_setA/training --model_path model_setA.h5

# Train on set B
python main.py --training_dir training_setB/training_setB --model_path model_setB.h5
```

## Expected Output

After training, you'll see:
- Model architecture summary
- Training progress with metrics
- Evaluation results on train/val/test sets
- Saved model files
- Visualization plots (ROC curve, confusion matrix, etc.)

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `max_patients`
- Reduce `sequence_length`
- Reduce `lstm_units`

### Slow Training
- Enable GPU if available
- Reduce `max_patients` for testing
- Use `--n_workers` to control parallel processing

### Poor Performance
- Increase training data (`max_patients`)
- Increase `sequence_length`
- Adjust `lstm_units` and `dropout_rate`
- Train for more `epochs`

