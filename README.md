# Quantum Machine Learning for Stroke Prediction

A comparative study of quantum and classical machine learning approaches for binary classification of stroke risk using a Variational Quantum Classifier (VQC).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Qiskit](https://img.shields.io/badge/qiskit-1.0.2-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Project Overview

This project implements and compares quantum machine learning against classical ML algorithms for healthcare prediction. Using the Healthcare Stroke Prediction dataset, we evaluate the current capabilities and limitations of quantum computing in practical machine learning tasks.

### Key Features
- ⚛️ Implementation of Variational Quantum Classifier using Qiskit
- 📊 Comprehensive comparison with classical ML models (SVM, Logistic Regression, Neural Network)
- 🔬 Rigorous data preprocessing with SMOTE balancing
- 📈 Detailed performance analysis and visualizations
- 📝 Honest evaluation of quantum ML's current state vs future potential

## 🎯 Motivation

While quantum computing promises revolutionary advances, realistic assessment of its current capabilities is crucial. This project:
- Demonstrates practical quantum ML implementation
- Provides honest comparison with mature classical methods
- Explores when quantum approaches might be beneficial
- Serves as educational resource for quantum computing concepts

## 📊 Results Summary

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Neural Network | 88.12% | 0.8812 | 14.52s |
| SVM | 85.60% | 0.8549 | 8.25s |
| Logistic Regression | 77.02% | 0.7739 | 0.02s |
| **Quantum VQC** | **54.00%** | **0.5741** | **221.20s** |

### Key Findings
- Classical models currently outperform quantum approaches for this task
- Quantum training is ~9,600x slower (due to simulation constraints)
- Gap expected to narrow with real quantum hardware and algorithm improvements
- Project provides valuable insights into quantum ML's current capabilities

## 🛠️ Technologies Used

- **Quantum Computing**: Qiskit 1.0.2, Qiskit Machine Learning 0.7.2
- **Classical ML**: Scikit-learn 1.4.0
- **Data Processing**: Pandas, NumPy, Imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn
- **Python**: 3.8+

## 📁 Project Structure

```
quantum-stroke-prediction/

├── src/
│   ├── 1_data_preprocessing.py
│   ├── 2_classical_models.py
│   ├── 3_quantum_classifier.py
│   └── 4_comparison_analysis.py
├── requirements.txt
├── README.md
└── QUICK_START.md

```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2-3 GB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/quantum-stroke-prediction.git
cd quantum-stroke-prediction
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Visit [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Download `healthcare-dataset-stroke-data.csv`
- Place in `data/` directory

### Running the Project

Execute scripts in order:

```bash
# 1. Preprocess data (~1 minute)
python src/1_data_preprocessing.py

# 2. Train classical models (~3 minutes)
python src/2_classical_models.py

# 3. Train quantum classifier (~15 minutes)
python src/3_quantum_classifier.py

# 4. Generate comparison analysis (~1 minute)
python src/4_comparison_analysis.py
```

Total runtime: ~20 minutes

## 🧪 Methodology

### Data Preprocessing
1. Handle missing values (BMI imputation)
2. Encode categorical variables
3. Balance classes using SMOTE (Synthetic Minority Over-sampling)
4. Feature selection (top 6 features)
5. Normalization (StandardScaler)

### Quantum Classifier Architecture
- **Feature Map**: ZZFeatureMap with 2 repetitions
- **Ansatz**: RealAmplitudes with 3 layers
- **Optimizer**: COBYLA (100 iterations)
- **Qubits**: 6 (one per feature)
- **Trainable Parameters**: Variable based on circuit depth

### Classical Baselines
- Support Vector Machine (RBF kernel)
- Logistic Regression
- Multi-Layer Perceptron (64-32 hidden layers)

## 📈 Visualizations

The project generates comprehensive visualizations:

- Performance comparison charts
- Confusion matrices
- Quantum circuit diagrams
- Feature distribution analysis
- Training time comparisons
- Efficiency metrics

Sample outputs saved in `results/` directory.

## 🔍 Key Insights

### Why Classical Performs Better (Currently)
1. **Maturity**: 70+ years of algorithmic optimization
2. **Hardware**: Runs natively on classical processors
3. **Simulation Overhead**: Quantum simulator mimics quantum physics classically
4. **Training Data**: Limited samples due to simulation constraints

### When Quantum Might Excel (Future)
1. **Real Quantum Hardware**: 1000+ qubit systems with low noise
2. **Quantum-Structured Problems**: Chemistry, optimization, quantum data
3. **Scale Advantages**: Exponential speedups for certain algorithms
4. **Hybrid Approaches**: Quantum feature extraction + classical classification

### Educational Value
- Understanding quantum computing fundamentals
- Experience with variational quantum algorithms
- Critical evaluation of emerging technologies
- Scientific integrity in performance reporting

## Learning Resources

- [Qiskit Textbook](https://qiskit.org/textbook)
- [IBM Quantum Experience](https://quantum-computing.ibm.com)
- [Variational Quantum Algorithms](https://arxiv.org/abs/2012.09265)
- [Quantum Machine Learning](https://arxiv.org/abs/1611.09347)


##  Acknowledgments

- Dataset: [Fedesoriano - Healthcare Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Quantum Framework: [Qiskit](https://qiskit.org/) by IBM
- Inspiration: Advancing understanding of practical quantum computing applications



---

**Note**: This is an educational project demonstrating quantum machine learning concepts. The quantum classifier's performance reflects the current state of quantum computing technology (2025) and serves as a realistic baseline for future improvements as quantum hardware advances.

