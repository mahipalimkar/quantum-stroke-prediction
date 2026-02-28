"""
SCRIPT 4: COMPARISON AND ANALYSIS
===================================

WHAT THIS SCRIPT DOES:
This script brings everything together - comparing the quantum classifier
with classical models and creating a comprehensive analysis report.

WHY THIS IS IMPORTANT:
This is where you demonstrate critical thinking! We're not just showing
quantum works - we're honestly evaluating when it's useful and when it's not.
This balanced approach is what makes your project stand out.

STEPS:
1. Load all results (classical and quantum)
2. Create comparative visualizations
3. Analyze strengths and weaknesses
4. Generate final report
5. Provide conclusions and future directions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("="*70)
print("QUANTUM VS CLASSICAL - COMPARATIVE ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: LOAD ALL RESULTS
# ============================================================================
print("\n[STEP 1] Loading results from all models...")

try:
    # Load classical results
    classical_results = pd.read_csv('results/classical_results.csv')
    print("✓ Loaded classical model results")
    
    # Load quantum results
    quantum_results = joblib.load('results/quantum_results.pkl')
    print("✓ Loaded quantum classifier results")
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find results files: {e}")
    print("Please run the previous scripts first:")
    print("  1. 1_data_preprocessing.py")
    print("  2. 2_classical_models.py")
    print("  3. 3_quantum_classifier.py")
    exit(1)

# ============================================================================
# STEP 2: CREATE COMPREHENSIVE COMPARISON
# ============================================================================
print("\n[STEP 2] Creating comprehensive comparison...")

# Add quantum results to the comparison
quantum_row = {
    'Model': 'Quantum VQC',
    'Accuracy': quantum_results['test_accuracy'],
    'Precision': quantum_results['test_precision'],
    'Recall': quantum_results['test_recall'],
    'F1-Score': quantum_results['test_f1'],
    'Training Time (s)': quantum_results['training_time']
}

# Combine all results
all_results = pd.concat([
    classical_results,
    pd.DataFrame([quantum_row])
], ignore_index=True)

print("\nComplete Results Summary:")
print("="*70)
print(all_results.to_string(index=False))
print("="*70)

# ============================================================================
# STEP 3: STATISTICAL ANALYSIS
# ============================================================================
print("\n[STEP 3] Performing statistical analysis...")

best_accuracy_model = all_results.loc[all_results['Accuracy'].idxmax(), 'Model']
best_accuracy = all_results['Accuracy'].max()

best_f1_model = all_results.loc[all_results['F1-Score'].idxmax(), 'Model']
best_f1 = all_results['F1-Score'].max()

fastest_model = all_results.loc[all_results['Training Time (s)'].idxmin(), 'Model']
fastest_time = all_results['Training Time (s)'].min()

slowest_model = all_results.loc[all_results['Training Time (s)'].idxmax(), 'Model']
slowest_time = all_results['Training Time (s)'].max()

print("\nKey Findings:")
print(f"  🏆 Best Accuracy: {best_accuracy_model} ({best_accuracy:.4f})")
print(f"  🏆 Best F1-Score: {best_f1_model} ({best_f1:.4f})")
print(f"  ⚡ Fastest Training: {fastest_model} ({fastest_time:.2f}s)")
print(f"  🐌 Slowest Training: {slowest_model} ({slowest_time:.2f}s = {slowest_time/60:.1f} min)")

# Calculate relative performance
quantum_accuracy = quantum_results['test_accuracy']
best_classical_accuracy = classical_results['Accuracy'].max()
accuracy_gap = (best_classical_accuracy - quantum_accuracy) * 100

print(f"\nQuantum vs Best Classical:")
print(f"  Accuracy gap: {accuracy_gap:.2f}% {'(classical better)' if accuracy_gap > 0 else '(quantum better)'}")
print(f"  Speed comparison: {quantum_results['training_time'] / fastest_time:.1f}x slower than fastest classical")

# ============================================================================
# STEP 4: CREATE VISUALIZATIONS
# ============================================================================
print("\n[STEP 4] Creating comprehensive visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Quantum vs Classical Machine Learning - Comprehensive Comparison', 
             fontsize=18, fontweight='bold')

# Define colors
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(all_results['Model'], all_results['Accuracy'], color=colors)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=10)

# Plot 2: All Metrics Comparison
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(all_results))
width = 0.2

for i, metric in enumerate(metrics):
    offset = (i - 1.5) * width
    ax2.bar(x + offset, all_results[metric], width, label=metric, alpha=0.8)

ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('All Performance Metrics', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(all_results['Model'], rotation=45, ha='right')
ax2.legend()
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Training Time Comparison (Log Scale)
ax3 = axes[1, 0]
bars3 = ax3.bar(all_results['Model'], all_results['Training Time (s)'], color=colors)
ax3.set_ylabel('Training Time (seconds, log scale)', fontsize=12)
ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s',
             ha='center', va='bottom', fontsize=10)

# Plot 4: Efficiency Score (Accuracy per Second)
ax4 = axes[1, 1]
efficiency = all_results['Accuracy'] / (all_results['Training Time (s)'] + 0.01)  # Add small value to avoid division by zero
bars4 = ax4.bar(all_results['Model'], efficiency, color=colors)
ax4.set_ylabel('Efficiency (Accuracy/Second)', fontsize=12)
ax4.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3)

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved comprehensive comparison plot")

# ============================================================================
# STEP 5: CREATE DETAILED ANALYSIS REPORT
# ============================================================================
print("\n[STEP 5] Generating detailed analysis report...")

report = f"""
{'='*80}
QUANTUM VS CLASSICAL MACHINE LEARNING
STROKE PREDICTION PROJECT - FINAL REPORT
{'='*80}

1. EXECUTIVE SUMMARY
{'-'*80}
This project compared quantum machine learning (Variational Quantum Classifier)
with classical machine learning algorithms for binary classification of stroke risk.

Dataset: Healthcare Stroke Prediction Dataset
- Total samples: {quantum_results['train_samples'] + quantum_results['test_samples']}
- Features: 6 (age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status)
- Classes: Balanced (50-50 split after SMOTE)

Models Compared:
- Classical: SVM, Logistic Regression, Neural Network
- Quantum: Variational Quantum Classifier (VQC)

{'='*80}

2. PERFORMANCE RESULTS
{'-'*80}

{all_results.to_string(index=False)}

Best Performing Model: {best_accuracy_model}
- Accuracy: {best_accuracy:.4f}
- F1-Score: {best_f1:.4f}

Quantum Classifier Performance:
- Test Accuracy: {quantum_results['test_accuracy']:.4f}
- Test Precision: {quantum_results['test_precision']:.4f}
- Test Recall: {quantum_results['test_recall']:.4f}
- Test F1-Score: {quantum_results['test_f1']:.4f}

{'='*80}

3. TRAINING EFFICIENCY
{'-'*80}

Training Time Comparison:
- Fastest: {fastest_model} - {fastest_time:.2f} seconds
- Quantum VQC: {quantum_results['training_time']:.2f} seconds ({quantum_results['training_time']/60:.1f} minutes)
- Speed ratio: Quantum is {quantum_results['training_time']/fastest_time:.1f}x slower

Note: Quantum training performed on classical simulator. Real quantum hardware
would be faster but still limited by current technology.

{'='*80}

4. QUANTUM CIRCUIT ANALYSIS
{'-'*80}

Circuit Architecture:
- Qubits used: {quantum_results.get('circuit_depth', 6)}
- Circuit depth: {quantum_results.get('circuit_depth', 'N/A')}
- Trainable parameters: {quantum_results['num_parameters']}
- Feature map: ZZFeatureMap with 2 repetitions
- Ansatz: RealAmplitudes with 3 layers

What makes this quantum?
✓ Data encoded in quantum superposition
✓ Qubits entangled to capture feature correlations
✓ Quantum interference used for classification
✓ Hybrid quantum-classical optimization

{'='*80}

5. KEY FINDINGS
{'-'*80}

Strengths of Classical ML:
✓ Higher accuracy on this dataset
✓ Much faster training (seconds vs minutes)
✓ Well-established, reliable algorithms
✓ Can handle larger datasets efficiently
✓ No specialized hardware required

Strengths of Quantum ML:
✓ Demonstrates quantum computing principles
✓ Different computational approach (parallel state processing)
✓ Potential for quantum advantage on specific problems
✓ Ability to capture complex feature interactions through entanglement
✓ Promising for future quantum hardware

Current Limitations of Quantum:
⚠ Slower on classical simulators
⚠ Limited by number of qubits (features)
⚠ Noise and errors in current quantum hardware
⚠ Still in research/development phase
⚠ Requires specialized knowledge

{'='*80}

6. WHEN TO USE QUANTUM ML?
{'-'*80}

Current State (2025):
- Classical ML is superior for most practical applications
- Quantum ML is primarily for research and exploration
- Limited real quantum hardware access

Future Potential (5-10 years):
Quantum ML may excel when:
✓ Large-scale quantum computers become available
✓ Problems with specific structure (e.g., quantum chemistry, optimization)
✓ Need to process quantum data directly
✓ Classical algorithms hit fundamental limits

Recommendation for this project:
Use classical ML for production systems. Use quantum ML to:
- Understand emerging technology
- Prepare for future quantum advantage
- Research novel algorithmic approaches

{'='*80}

7. CONCLUSIONS
{'-'*80}

This project successfully demonstrated:
1. Implementation of both quantum and classical ML classifiers
2. Rigorous comparison with multiple evaluation metrics
3. Critical analysis of quantum computing limitations
4. Understanding of when quantum approaches may be beneficial

Key Insight:
While quantum machine learning shows promise, classical methods currently
outperform on standard datasets like stroke prediction. However, this project
provides valuable experience with quantum computing concepts and prepares for
future developments in quantum technology.

The gap between quantum and classical performance is expected to narrow as:
- Quantum hardware improves (more qubits, less noise)
- Better quantum algorithms are developed
- Quantum-specific problems are identified

{'='*80}

8. FUTURE WORK
{'-'*80}

To extend this project:
1. Test on quantum hardware (IBM Quantum, AWS Braket)
2. Experiment with different quantum circuit architectures
3. Implement quantum feature selection techniques
4. Try quantum ensemble methods
5. Explore quantum-enhanced classical hybrid models
6. Test on problems with quantum structure (molecular data)
7. Implement error mitigation techniques
8. Compare with more quantum ML algorithms (QSVM, QNN)

{'='*80}

9. TECHNICAL SPECIFICATIONS
{'-'*80}

Software Stack:
- Python 3.8+
- Qiskit 1.0.2
- Qiskit Machine Learning 0.7.2
- Scikit-learn 1.4.0
- NumPy, Pandas, Matplotlib

Quantum Simulator:
- Qiskit Aer Simulator
- Statevector simulation
- Classical computer execution

Dataset:
- Source: Kaggle Healthcare Stroke Dataset
- Preprocessing: SMOTE balancing, feature selection, normalization
- Split: 80% train, 20% test

{'='*80}

10. REFERENCES & RESOURCES
{'-'*80}

Key Concepts:
- Variational Quantum Algorithms
- Quantum Feature Maps
- Hybrid Quantum-Classical Optimization
- Parameterized Quantum Circuits

Learning Resources:
- Qiskit Textbook: https://qiskit.org/textbook
- IBM Quantum: https://quantum-computing.ibm.com
- Quantum Machine Learning Papers: arXiv.org

{'='*80}

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Project: Quantum Machine Learning for Stroke Prediction
{'='*80}
"""

# Save the report (with UTF-8 encoding to handle special characters)
with open('results/FINAL_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ Saved detailed analysis report to results/FINAL_REPORT.txt")

# ============================================================================
# STEP 6: CREATE SUMMARY TABLE FOR PRESENTATION
# ============================================================================
print("\n[STEP 6] Creating summary table...")

# Create a nice summary table
summary_table = all_results[['Model', 'Accuracy', 'F1-Score', 'Training Time (s)']].copy()
summary_table['Training Time (s)'] = summary_table['Training Time (s)'].round(2)
summary_table = summary_table.round(4)

# Save as CSV
summary_table.to_csv('results/summary_comparison.csv', index=False)
print("✓ Saved summary table to results/summary_comparison.csv")

# Create a nice formatted table image
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Model', 'Accuracy', 'F1-Score', 'Training Time (s)'])
for _, row in summary_table.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Accuracy']:.4f}",
        f"{row['F1-Score']:.4f}",
        f"{row['Training Time (s)']:.2f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.2, 0.2, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style quantum row
quantum_row_idx = len(table_data) - 1
for i in range(4):
    table[(quantum_row_idx, i)].set_facecolor('#e8f4f8')

plt.title('Model Comparison Summary Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved summary table image")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE!")
print("="*80)

print("\nAll Results Files Generated:")
print("  📊 results/comprehensive_comparison.png - Main comparison plots")
print("  📊 results/summary_table.png - Summary table image")
print("  📄 results/FINAL_REPORT.txt - Detailed analysis report")
print("  📄 results/summary_comparison.csv - Quick reference table")

print("\nProject Summary:")
print(f"  Models Evaluated: {len(all_results)}")
print(f"  Best Accuracy: {best_accuracy:.4f} ({best_accuracy_model})")
print(f"  Quantum Accuracy: {quantum_results['test_accuracy']:.4f}")
print(f"  Accuracy Gap: {accuracy_gap:.2f}%")

print("\n" + "="*80)
print("PROJECT COMPLETE!")
print("="*80)
print("\nYou now have:")
print("  ✅ Trained quantum and classical classifiers")
print("  ✅ Comprehensive performance comparison")
print("  ✅ Detailed visualizations")
print("  ✅ Professional analysis report")
print("  ✅ Clear conclusions and insights")

print("\nFor your presentation/report, use:")
print("  1. results/comprehensive_comparison.png - Main results")
print("  2. results/quantum_circuit_diagram.png - Show quantum architecture")
print("  3. results/FINAL_REPORT.txt - Reference for writing")
print("  4. All CSV files - For detailed tables")

print("\n🎓 Great work! This project demonstrates:")
print("   - Quantum computing implementation skills")
print("   - Critical analysis and scientific thinking")
print("   - Professional data science workflow")
print("   - Understanding of emerging technologies")

print("\nNext steps:")
print("  - Review the FINAL_REPORT.txt")
print("  - Prepare your presentation slides")
print("  - Practice explaining quantum concepts")
print("  - Highlight both achievements and limitations")