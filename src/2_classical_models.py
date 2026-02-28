"""
SCRIPT 2: CLASSICAL MACHINE LEARNING MODELS
=============================================

WHAT THIS SCRIPT DOES:
This script trains three classical machine learning models on the preprocessed data.
These will serve as our baseline to compare against the quantum classifier.

WHY WE NEED THIS:
We need to know how well traditional methods perform before we can evaluate
whether quantum computing offers any advantages. Right now, classical ML is 
still more accurate than quantum ML, but we're exploring the quantum approach
to understand its potential for the future.

MODELS WE'LL TRAIN:
1. Support Vector Machine (SVM) - Good for high-dimensional data
2. Logistic Regression - Simple, interpretable linear model
3. Neural Network - Non-linear model that can learn complex patterns

STEPS:
1. Load preprocessed data
2. Train each classical model
3. Evaluate performance (accuracy, precision, recall, F1-score)
4. Save models and results
5. Create visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import joblib
import time
import os

print("="*70)
print("QUANTUM STROKE PREDICTION - CLASSICAL MODELS")
print("="*70)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("\n[STEP 1] Loading preprocessed data...")

try:
    X_train = np.load('results/X_train.npy')
    X_test = np.load('results/X_test.npy')
    y_train = np.load('results/y_train.npy')
    y_test = np.load('results/y_test.npy')
    selected_features = joblib.load('results/selected_features.pkl')
    
    print(f"✓ Loaded training data: {X_train.shape}")
    print(f"✓ Loaded test data: {X_test.shape}")
    print(f"✓ Selected features: {selected_features}")
except FileNotFoundError:
    print("ERROR: Preprocessed data not found!")
    print("Please run '1_data_preprocessing.py' first.")
    exit(1)

# ============================================================================
# STEP 2: DEFINE MODELS
# ============================================================================
print("\n[STEP 2] Defining classical models...")

"""
MODEL EXPLANATIONS:

1. Support Vector Machine (SVM):
   - Finds the best boundary (hyperplane) that separates the two classes
   - Works well with high-dimensional data
   - 'rbf' kernel allows it to learn non-linear patterns
   - Think of it as drawing the best possible line between stroke/non-stroke patients

2. Logistic Regression:
   - Simple linear model that predicts probability of stroke
   - Fast to train and easy to interpret
   - Good baseline model
   - Assumes a linear relationship between features and outcome

3. Neural Network (Multi-Layer Perceptron):
   - Can learn complex, non-linear patterns
   - Has hidden layers that transform the data
   - More powerful but also more prone to overfitting
   - Similar architecture to deep learning models
"""

models = {
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), 
                                    max_iter=1000, 
                                    random_state=42)
}

print("✓ Defined 3 classical models:")
for name in models.keys():
    print(f"  - {name}")

# ============================================================================
# STEP 3: TRAIN AND EVALUATE MODELS
# ============================================================================
print("\n[STEP 3] Training and evaluating models...")

results = {}

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    # Train the model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'predictions': y_pred,
        'model': model
    }
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    """
    METRIC EXPLANATIONS:
    
    Accuracy: (TP + TN) / Total
    - Overall correctness: What % of predictions were correct?
    
    Precision: TP / (TP + FP)
    - When we predict stroke, how often are we right?
    - Important if false alarms are costly
    
    Recall: TP / (TP + FN)
    - Of all actual stroke cases, how many did we catch?
    - Important in medical diagnosis - we don't want to miss strokes!
    
    F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    - Balance between precision and recall
    - Good overall metric when classes are balanced
    """
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"  True Negatives (TN):  {cm[0,0]}")
    print(f"  False Positives (FP): {cm[0,1]}")
    print(f"  False Negatives (FN): {cm[1,0]}")
    print(f"  True Positives (TP):  {cm[1,1]}")

# ============================================================================
# STEP 4: SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "="*70)
print("[STEP 4] Saving models and results...")
print("="*70)

# Save each trained model
for model_name, result in results.items():
    safe_name = model_name.replace(' ', '_').lower()
    joblib.dump(result['model'], f'results/{safe_name}_model.pkl')
    print(f"✓ Saved {model_name} model")

# Save results summary
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1_score'] for r in results.values()],
    'Training Time (s)': [r['training_time'] for r in results.values()]
})

results_df.to_csv('results/classical_results.csv', index=False)
print("✓ Saved results summary to classical_results.csv")

# ============================================================================
# STEP 5: CREATE VISUALIZATIONS
# ============================================================================
print("\n[STEP 5] Creating visualizations...")

# 1. Performance Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Classical Models - Performance Comparison', fontsize=16)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    row = idx // 2
    col = idx % 2
    
    values = [results[model][metric.lower().replace('-', '_')] for model in models.keys()]
    bars = axes[row, col].bar(models.keys(), values, color=colors)
    axes[row, col].set_ylabel(metric)
    axes[row, col].set_ylim(0, 1)
    axes[row, col].set_title(f'{metric} Comparison')
    axes[row, col].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/classical_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved performance comparison plot")

# 2. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices', fontsize=16)

for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(model_name)
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('results/classical_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrices plot")

# 3. Training Time Comparison
plt.figure(figsize=(10, 6))
times = [results[model]['training_time'] for model in models.keys()]
bars = plt.bar(models.keys(), times, color=colors)
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/classical_training_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved training time plot")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("CLASSICAL MODELS TRAINING COMPLETE!")
print("="*70)

print("\nFinal Results Summary:")
print(results_df.to_string(index=False))

print("\nKey Observations:")
best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_accuracy = results_df['Accuracy'].max()
print(f"✓ Best performing model: {best_model} with {best_accuracy:.4f} accuracy")

fastest_model = results_df.loc[results_df['Training Time (s)'].idxmin(), 'Model']
fastest_time = results_df['Training Time (s)'].min()
print(f"✓ Fastest training: {fastest_model} at {fastest_time:.2f} seconds")

print("\nFiles saved:")
print("  - results/classical_results.csv")
print("  - results/classical_performance_comparison.png")
print("  - results/classical_confusion_matrices.png")
print("  - results/classical_training_time.png")
print("  - results/*_model.pkl (trained models)")

print("\nNext step: Run '3_quantum_classifier.py' to train the quantum model")