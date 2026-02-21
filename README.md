# Predictive alerting for cloud metrics

## Problem

Predict whether an **incident will occur within the next H time steps** using the previous W time steps of a time-series metric.

This is formulated as a **binary classification problem**:

- **Input:** Sliding window of past W values  
- **Target:** 1 if any incident occurs in the next H steps, else 0  

The task mimics a real-world alerting system where early detection of anomalies and failures are critical.

## Dataset

To keep the focus on modeling rather than dataset complexity, a **synthetic time series** is generated.

- `T = 2000` time steps  
- Base signal: sinusoid + Gaussian noise  
- Fixed random seed (89) for reproducibility
- 40 injected incidents (5 steps each)  
- Each incident is preceded by a small upward drift and followed by a spike  

This simulates gradual degradation followed by failure, which is similar to the real-world scenario.

## Sliding Window Formulation

Parameters:
- **W = 20** - look-back window size
- **H = 5** - prediction horizon  

For each time index `i`:
- Features: `data[i-W : i]`
- Label: `1` if any incident in `incidents[i : i+H]`

## Feature Engineering

Features include:
- Raw window values
- First-order differences (`np.diff`)  

Differences help capture early drift patterns before incidents, which helps model to learn pattern, particularly in case of **Logistic Regression**.
First-order differences didn't improve model performance in this case, but it was included to showcase feature engineering process.

## Models

Two models were trained to compare linear vs nonlinear decision boundaries:

### Logistic Regression
- Linear baseline  
- `class_weight='balanced'`   

### Random Forest
- 100 trees  
- Nonlinear  
- `class_weight='balanced'`  

## Training Setup

- 75% / 25% train-test split  
  - First 75% of the data for training
  - Last 25% of the data for testing

## Evaluation

Models output incident probabilities.  
The alert threshold is optimized over:

{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}

For each threshold, we compute precision and recall, and select the one that maximizes: `Score = 0.5 × Recall + 0.5 × Precision`
This allows us to prioritize detecting incidents (recall) or prediction correctness (precision).

Using the best threshold, we report:
- F1-score
- Recall
- Precision
- ROC-AUC

## Results Summary

- **Logistic Regression**  
  Best threshold: 0.5 – F1 ≈ 0.700, Recall ≈ 0.673, Precision ≈ 0.729, AUC ≈ 0.888  
  Captures consistent drift patterns well, but slightly lower recall for detecting incidents.

- **Random Forest**  
  Best threshold: 0.2 – F1 ≈ 0.813, Recall ≈ 0.837, Precision ≈ 0.791, AUC ≈ 0.921 
  Outperforms Logistic Regression on all metrics for this dataset.
  Better at modeling nonlinear spikes and identifying incidents early.

- **Threshold effect**  
  Increasing the alert threshold generally increases precision but reduces recall.

## Future improvements

- Real-world dataset
- Cross-validation 
- Hyperparameter tuning  

## How to Run

```bash
python main.py
