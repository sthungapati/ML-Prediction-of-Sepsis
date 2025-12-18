# Mathematical Methods in Sepsis Prediction Model

This document describes the mathematical methods and algorithms used in the sepsis prediction pipeline.

## 1. Data Preprocessing

### 1.1 Missing Value Imputation

**Forward Fill (Carry Forward)**
```
x[t] = x[t-1] if x[t] is NaN, else x[t]
```
Carries forward the last known value, appropriate for time-series clinical data.

**Backward Fill**
```
x[t] = x[t+1] if x[t] is NaN, else x[t]
```
Fills missing values from future timesteps when forward fill is not available.

**Median Imputation**
```
x[t] = median(x) if x[t] is still NaN after forward/backward fill
```
Uses column median for remaining missing values.

### 1.2 Z-Score Normalization

Standardizes features to have zero mean and unit variance:

```
z = (x - μ) / σ
```

where:
- `μ` = mean of the feature across all training samples
- `σ` = standard deviation of the feature
- `z` = normalized feature value

This ensures all features are on the same scale, which is crucial for neural network training.

## 2. Sequence Generation

### 2.1 Sliding Window

Creates sequences of fixed length using a sliding window approach:

```
X[i] = [x[t], x[t+1], ..., x[t+L-1]]
y[i] = label[t+L+offset-1]
```

where:
- `L` = sequence length (e.g., 24 hours)
- `offset` = prediction horizon (e.g., 6 hours)
- `t` = starting timestep

### 2.2 Padding

For sequences shorter than the maximum length, padding is applied:

**Pre-padding (default)**
```
X_padded[i] = [0, 0, ..., 0, x[0], x[1], ..., x[n]]
```

**Post-padding**
```
X_padded[i] = [x[0], x[1], ..., x[n], 0, 0, ..., 0]
```

### 2.3 Masking

Binary mask indicates valid (non-padded) timesteps:

```
mask[i, t] = {
    1 if t < actual_length[i]
    0 if t >= actual_length[i]
}
```

This mask is used by the LSTM to ignore padded timesteps during computation.

## 3. LSTM Architecture

### 3.1 LSTM Cell Equations

The Long Short-Term Memory cell processes sequences using the following equations:

**Forget Gate**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
Determines what information to discard from the cell state.

**Input Gate**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```
Determines what new information to store in the cell state.

**Cell State Update**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```
Updates the cell state by forgetting old information and adding new information.

**Output Gate**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```
Determines what parts of the cell state to output.

where:
- `σ` = sigmoid activation function
- `tanh` = hyperbolic tangent activation
- `⊙` = element-wise multiplication (Hadamard product)
- `W_f, W_i, W_C, W_o` = weight matrices
- `b_f, b_i, b_C, b_o` = bias vectors
- `h_t` = hidden state at time t
- `C_t` = cell state at time t
- `x_t` = input at time t

### 3.2 Stacked LSTM

Multiple LSTM layers are stacked:

```
h_t^(1) = LSTM_1(x_t, h_{t-1}^(1))
h_t^(2) = LSTM_2(h_t^(1), h_{t-1}^(2))
...
```

Each layer processes the output of the previous layer, allowing the model to learn hierarchical temporal patterns.

### 3.3 Dropout Regularization

During training, randomly sets a fraction of inputs to zero:

```
h_dropout = h ⊙ mask_dropout
mask_dropout ~ Bernoulli(1 - p)
```

where `p` is the dropout rate (e.g., 0.3 = 30% dropout).

This prevents overfitting by preventing the model from relying too heavily on specific neurons.

### 3.4 Batch Normalization

Normalizes activations within a batch:

```
μ_B = (1/m) Σ x_i
σ_B² = (1/m) Σ (x_i - μ_B)²
x̂_i = (x_i - μ_B) / √(σ_B² + ε)
y_i = γ · x̂_i + β
```

where:
- `m` = batch size
- `ε` = small constant (e.g., 1e-5) to prevent division by zero
- `γ, β` = learnable parameters

This stabilizes training and allows for higher learning rates.

## 4. Loss Function

### 4.1 Binary Cross-Entropy Loss

For binary classification (sepsis vs. no sepsis):

```
L = -[y · log(ŷ) + (1-y) · log(1-ŷ)]
```

where:
- `y` = true label (0 or 1)
- `ŷ` = predicted probability (0 to 1)

**Gradient:**
```
∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
```

This loss function is appropriate for binary classification and works well with sigmoid activation.

## 5. Activation Functions

### 5.1 Sigmoid (Output Layer)

```
σ(z) = 1 / (1 + e^(-z))
```

Outputs values between 0 and 1, representing probability of sepsis.

**Derivative:**
```
σ'(z) = σ(z) · (1 - σ(z))
```

### 5.2 ReLU (Hidden Layers)

```
ReLU(z) = max(0, z)
```

Provides non-linearity while being computationally efficient.

**Derivative:**
```
ReLU'(z) = {
    1 if z > 0
    0 if z ≤ 0
}
```

### 5.3 Hyperbolic Tangent (LSTM Gates)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

Outputs values between -1 and 1, used in LSTM cell state updates.

## 6. Optimization

### 6.1 Adam Optimizer

Adaptive Moment Estimation combines momentum and RMSprop:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

where:
- `g_t` = gradient at time t
- `β₁, β₂` = decay rates (typically 0.9 and 0.999)
- `α` = learning rate
- `ε` = small constant (e.g., 1e-8)

Adam adapts the learning rate for each parameter, making it robust to different hyperparameter settings.

### 6.2 Learning Rate Scheduling

**Reduce on Plateau:**
```
lr_new = lr_old · factor if no improvement for patience epochs
```

Reduces learning rate when validation loss plateaus, allowing fine-tuning.

## 7. Evaluation Metrics

### 7.1 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Proportion of correct predictions.

### 7.2 Precision

```
Precision = TP / (TP + FP)
```

Proportion of positive predictions that are correct.

### 7.3 Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

Proportion of actual positives that are correctly identified.

### 7.4 F1-Score

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall.

### 7.5 ROC-AUC

Area under the Receiver Operating Characteristic curve:

```
AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
```

where:
- `TPR` = True Positive Rate (Recall)
- `FPR` = False Positive Rate

Measures the model's ability to distinguish between classes.

### 7.6 PR-AUC

Area under the Precision-Recall curve:

```
PR-AUC = ∫₀¹ Precision(Recall⁻¹(x)) dx
```

More informative than ROC-AUC for imbalanced datasets.

## 8. Parallel Processing

### 8.1 Data Loading Parallelization

Using multiprocessing, patient files are processed in parallel:

```
Process 1: [file_1, file_2, ..., file_n]
Process 2: [file_{n+1}, file_{n+2}, ..., file_{2n}]
...
```

Each process independently loads and preprocesses its assigned files, then results are combined.

### 8.2 GPU Acceleration

TensorFlow automatically parallelizes operations across GPU cores:

```
Matrix multiplication: O(n³) operations distributed across cores
```

GPU parallelization significantly speeds up training for large models and datasets.

## 9. Label Creation

### 9.1 Future Window Labeling

For each timestep `t`, check if sepsis occurs within the next `H` hours:

```
y[t] = {
    1 if ∃ s ∈ [t+1, t+H] : SepsisLabel[s] == 1
    0 otherwise
}
```

where `H` is the prediction horizon (e.g., 6 hours).

This creates a supervised learning problem where the model learns to predict future sepsis onset.

## 10. Mathematical Properties

### 10.1 Gradient Flow

LSTM's gating mechanism helps with gradient flow:

```
∂L/∂C_{t-k} = ∂L/∂C_t · ∏_{i=t-k}^{t-1} f_i
```

The forget gate allows gradients to flow through time, enabling learning of long-term dependencies.

### 10.2 Vanishing Gradient Mitigation

LSTM addresses vanishing gradients through:
- Additive cell state updates (instead of multiplicative)
- Gating mechanisms that can learn to preserve gradients
- Skip connections through cell state

### 10.3 Overfitting Prevention

Multiple regularization techniques:
- **Dropout**: Randomly zeroes activations
- **Batch Normalization**: Reduces internal covariate shift
- **Early Stopping**: Stops training when validation performance plateaus
- **Weight Decay**: Implicitly through Adam optimizer

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

2. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

3. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML.

4. Srivastava, N., et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR.

