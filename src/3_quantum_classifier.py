"""
SCRIPT 3: QUANTUM CLASSIFIER
==============================

WHAT THIS SCRIPT DOES:
This script implements a Variational Quantum Classifier (VQC) using Qiskit.
This is the core of your project - where classical data meets quantum computing!

HOW IT WORKS:
1. Classical data is encoded into quantum states (superposition)
2. A parameterized quantum circuit transforms these states
3. We measure the output to get a classification
4. A classical optimizer adjusts the circuit parameters to improve accuracy
5. This hybrid quantum-classical loop continues until convergence

KEY QUANTUM CONCEPTS USED:
- Superposition: Qubits can be 0, 1, or both simultaneously
- Entanglement: Qubits become correlated with each other
- Measurement: Collapses quantum state to classical output
- Parameterized gates: Quantum gates with adjustable angles (like neural network weights)

STEPS:
1. Load preprocessed data
2. Design the quantum circuit (feature map + ansatz)
3. Set up the quantum classifier
4. Train the model
5. Evaluate performance
6. Visualize the quantum circuit
7. Save results
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import time

print("="*70)
print("QUANTUM STROKE PREDICTION - QUANTUM CLASSIFIER")
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
    print(f"✓ Number of features (qubits needed): {X_train.shape[1]}")
except FileNotFoundError:
    print("ERROR: Preprocessed data not found!")
    print("Please run '1_data_preprocessing.py' first.")
    exit(1)

# Use a subset for faster training (quantum simulators are slow!)
# In practice, you might want to train on more data, but it will take hours
TRAIN_SAMPLES = 200  # Reduce for faster testing, increase for better accuracy
TEST_SAMPLES = 100

print(f"\n⚠ Note: Using {TRAIN_SAMPLES} training samples for reasonable computation time")
print(f"  (Full training would take 2-3 hours on a simulator)")

X_train_subset = X_train[:TRAIN_SAMPLES]
y_train_subset = y_train[:TRAIN_SAMPLES]
X_test_subset = X_test[:TEST_SAMPLES]
y_test_subset = y_test[:TEST_SAMPLES]

num_features = X_train.shape[1]  # Number of qubits = number of features

# ============================================================================
# STEP 2: DESIGN THE QUANTUM CIRCUIT
# ============================================================================
print("\n[STEP 2] Designing the quantum circuit...")

"""
QUANTUM CIRCUIT ARCHITECTURE:

Our quantum circuit has two main parts:

1. FEATURE MAP (Data Encoding):
   - Takes classical data and encodes it into quantum states
   - We use ZZFeatureMap which creates entanglement between qubits
   - Each data point becomes a unique quantum state |ψ(x)⟩
   
   Example: If patient has age=0.5, glucose=0.8, this becomes:
   |ψ⟩ = cos(0.5)|0⟩ + sin(0.5)|1⟩ ⊗ cos(0.8)|0⟩ + sin(0.8)|1⟩ ⊗ ...
   (This is simplified - actual encoding is more complex)

2. ANSATZ (Parameterized Circuit):
   - Acts like the "model" - has trainable parameters
   - RealAmplitudes ansatz applies rotation gates with adjustable angles
   - These parameters are optimized during training (like neural network weights)
   - Creates entanglement between qubits to capture correlations
   
The combination looks like:
Data → Feature Map → Ansatz → Measurement → Prediction
"""

# Create feature map (encodes classical data into quantum states)
feature_map = ZZFeatureMap(feature_dimension=num_features, 
                           reps=2,  # How many times to repeat the encoding
                           entanglement='linear')

print("✓ Created ZZFeatureMap for data encoding")
print(f"  - Features (qubits): {num_features}")
print(f"  - Encoding repetitions: 2")
print(f"  - Entanglement pattern: linear (qubit i connects to i+1)")

# Create ansatz (the parameterized part we'll train)
ansatz = RealAmplitudes(num_qubits=num_features, 
                        reps=3,  # Depth of the circuit
                        entanglement='full')

print("✓ Created RealAmplitudes ansatz")
print(f"  - Qubits: {num_features}")
print(f"  - Layers: 3")
print(f"  - Trainable parameters: {ansatz.num_parameters}")
print(f"  - Entanglement: full (all qubits can interact)")

# Visualize the circuit structure
print("\n[Visualizing circuit structure...]")
full_circuit = QuantumCircuit(num_features)
full_circuit.compose(feature_map, inplace=True)
full_circuit.compose(ansatz, inplace=True)

# Save circuit diagram
try:
    circuit_fig = full_circuit.draw(output='mpl', style='clifford')
    plt.savefig('results/quantum_circuit_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Saved circuit diagram to results/quantum_circuit_diagram.png")
except Exception as e:
    print(f"⚠ Could not save circuit diagram: {e}")

print(f"\nCircuit Statistics:")
print(f"  - Total qubits: {full_circuit.num_qubits}")
print(f"  - Circuit depth: {full_circuit.depth()}")
print(f"  - Total gates: {sum(full_circuit.count_ops().values())}")

# ============================================================================
# STEP 3: SET UP THE QUANTUM CLASSIFIER
# ============================================================================
print("\n[STEP 3] Setting up the Variational Quantum Classifier...")

"""
VARIATIONAL QUANTUM CLASSIFIER (VQC):

This is a hybrid quantum-classical algorithm:

1. Quantum part: Runs the circuit and measures output
2. Classical part: Computes loss and updates parameters
3. Repeat until convergence

The "variational" means we vary the parameters to minimize loss,
just like training a neural network with gradient descent.

OPTIMIZER:
We use COBYLA (Constrained Optimization BY Linear Approximation)
- Doesn't need gradients (good for noisy quantum circuits)
- Robust but can be slow
- Alternative: SPSA, Adam, etc.
"""

# Create the sampler (handles quantum measurements)
sampler = Sampler()

# Create optimizer
optimizer = COBYLA(maxiter=100)  # Maximum 100 iterations
print("✓ Created COBYLA optimizer")
print(f"  - Max iterations: 100")

# Create the VQC
print("\n✓ Initializing Variational Quantum Classifier...")
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
)

print("✓ VQC created successfully!")

# ============================================================================
# STEP 4: TRAIN THE QUANTUM MODEL
# ============================================================================
print("\n[STEP 4] Training the quantum model...")
print("="*70)
print("⏰ WARNING: This will take 10-20 minutes depending on your computer!")
print("The quantum simulator needs to calculate complex quantum states.")
print("You'll see the loss decreasing as training progresses.")
print("="*70)

start_time = time.time()

# Train the model
print("\nStarting training...")
vqc.fit(X_train_subset, y_train_subset)

training_time = time.time() - start_time

print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ============================================================================
# STEP 5: EVALUATE PERFORMANCE
# ============================================================================
print("\n[STEP 5] Evaluating quantum classifier performance...")

# Make predictions on test set
print("Making predictions on test set...")
y_pred_train = vqc.predict(X_train_subset)
y_pred_test = vqc.predict(X_test_subset)

# Calculate metrics
train_accuracy = accuracy_score(y_train_subset, y_pred_train)
test_accuracy = accuracy_score(y_test_subset, y_pred_test)
test_precision = precision_score(y_test_subset, y_pred_test)
test_recall = recall_score(y_test_subset, y_pred_test)
test_f1 = f1_score(y_test_subset, y_pred_test)

print("\n" + "="*70)
print("QUANTUM CLASSIFIER RESULTS")
print("="*70)
print(f"\nTraining Set:")
print(f"  Accuracy: {train_accuracy:.4f}")

print(f"\nTest Set:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

print(f"\nTraining Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Confusion Matrix
cm = confusion_matrix(y_test_subset, y_pred_test)
print(f"\nConfusion Matrix:")
print(cm)
print(f"  True Negatives (TN):  {cm[0,0]}")
print(f"  False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}")
print(f"  True Positives (TP):  {cm[1,1]}")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
print("\n[STEP 6] Saving quantum classifier results...")

# Save the trained model
joblib.dump(vqc, 'results/quantum_classifier.pkl')
print("✓ Saved quantum classifier model")

# Save results
quantum_results = {
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'training_time': training_time,
    'num_parameters': ansatz.num_parameters,
    'circuit_depth': full_circuit.depth(),
    'train_samples': TRAIN_SAMPLES,
    'test_samples': TEST_SAMPLES
}

joblib.dump(quantum_results, 'results/quantum_results.pkl')
print("✓ Saved quantum results")

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================
print("\n[STEP 7] Creating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Quantum Classifier - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/quantum_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrix")

# 2. Performance Metrics Bar Chart
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [test_accuracy, test_precision, test_recall, test_f1]
colors = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71']

bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Quantum Classifier - Performance Metrics')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/quantum_performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved performance metrics chart")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("QUANTUM CLASSIFIER TRAINING COMPLETE!")
print("="*70)

print("\nKey Statistics:")
print(f"  - Test Accuracy: {test_accuracy:.4f}")
print(f"  - Training Time: {training_time/60:.2f} minutes")
print(f"  - Circuit Parameters: {ansatz.num_parameters}")
print(f"  - Circuit Depth: {full_circuit.depth()}")
print(f"  - Qubits Used: {num_features}")

print("\nWhat makes this quantum?")
print("  ✓ Data encoded in quantum superposition")
print("  ✓ Qubits entangled to capture correlations")
print("  ✓ Quantum gates process information")
print("  ✓ Measurement collapses to classical prediction")

print("\nFiles saved:")
print("  - results/quantum_classifier.pkl")
print("  - results/quantum_results.pkl")
print("  - results/quantum_circuit_diagram.png")
print("  - results/quantum_confusion_matrix.png")
print("  - results/quantum_performance_metrics.png")

print("\nNext step: Run '4_comparison_analysis.py' to compare quantum vs classical")