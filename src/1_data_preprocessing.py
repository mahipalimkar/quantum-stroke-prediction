"""
SCRIPT 1: DATA PREPROCESSING
=============================

WHAT THIS SCRIPT DOES:
This script takes the raw stroke dataset and prepares it for machine learning.
We need to clean the data, handle missing values, balance the classes, and 
select the most important features for our quantum classifier.

WHY WE NEED THIS:
- Raw data has missing values that will break our models
- The dataset is imbalanced (very few stroke cases) which makes learning hard
- Quantum simulators can only handle a few qubits, so we need to reduce features
- Features have different scales (age is 0-100, BMI is 10-60) which needs normalization

STEPS:
1. Load the data
2. Explore basic statistics
3. Handle missing values
4. Encode categorical variables (convert text to numbers)
5. Balance the classes using SMOTE
6. Select top features
7. Normalize features
8. Split into train/test sets
9. Save processed data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import os

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("QUANTUM STROKE PREDICTION - DATA PREPROCESSING")
print("="*70)

# ============================================================================
# STEP 1: LOAD THE DATA
# ============================================================================
print("\n[STEP 1] Loading the dataset...")

# Update this path to where you saved the CSV file
DATA_PATH = 'data/healthcare-dataset-stroke-data.csv'

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    print("Please download the dataset from Kaggle and place it in the data/ folder")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 2: INITIAL DATA EXPLORATION
# ============================================================================
print("\n[STEP 2] Exploring the data...")

print("\nFirst few rows:")
print(df.head())

print("\nColumn names and types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution (stroke cases):")
print(df['stroke'].value_counts())
print(f"Percentage with stroke: {df['stroke'].mean()*100:.2f}%")
print("⚠ Notice: The dataset is highly imbalanced! We'll fix this later.")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================================
print("\n[STEP 3] Handling missing values...")

# BMI has some missing values - let's fill them with the median
# Why median? It's robust to outliers (unlike mean)
if df['bmi'].isnull().sum() > 0:
    median_bmi = df['bmi'].median()
    df['bmi'].fillna(median_bmi, inplace=True)
    print(f"✓ Filled {df['bmi'].isnull().sum()} missing BMI values with median: {median_bmi:.2f}")

# Drop the 'id' column - it's just a unique identifier, not useful for prediction
df = df.drop('id', axis=1)
print("✓ Removed 'id' column")

# ============================================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n[STEP 4] Converting categorical variables to numbers...")

"""
WHY: Machine learning models (including quantum) need numbers, not text.
We'll use Label Encoding to convert categories to integers.

Example: gender: 'Male' -> 0, 'Female' -> 1
"""

# Create a copy to avoid warnings
df_encoded = df.copy()

# List of categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"✓ Encoded '{col}': {list(le.classes_)}")

# ============================================================================
# STEP 5: FEATURE SELECTION
# ============================================================================
print("\n[STEP 5] Selecting the most important features...")

"""
WHY: Quantum simulators are limited in the number of qubits they can handle.
We'll select the top 6 most predictive features using statistical tests.

This uses ANOVA F-statistic to measure how much each feature correlates with stroke.
"""

# Separate features (X) and target (y)
X = df_encoded.drop('stroke', axis=1)
y = df_encoded['stroke']

# Select top 6 features
selector = SelectKBest(score_func=f_classif, k=6)
X_selected = selector.fit_transform(X, y)

# Get the names of selected features
selected_feature_mask = selector.get_support()
selected_features = X.columns[selected_feature_mask].tolist()

print(f"✓ Selected top 6 features: {selected_features}")
print("\nFeature importance scores:")
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)
print(feature_scores.head(10))

# Create dataframe with selected features
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# ============================================================================
# STEP 6: BALANCE CLASSES WITH SMOTE
# ============================================================================
print("\n[STEP 6] Balancing classes using SMOTE...")

"""
WHY: We have very few stroke cases (5%) compared to non-stroke (95%).
If we train on this imbalanced data, the model will just predict "no stroke" 
for everyone and get 95% accuracy without learning anything useful!

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples
of the minority class (stroke=1) by interpolating between existing examples.

Think of it like this: If you have two stroke patients with similar characteristics,
SMOTE creates a "virtual" patient somewhere between them.
"""

print(f"Before SMOTE: {y.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected_df, y)

print(f"After SMOTE: {pd.Series(y_balanced).value_counts().to_dict()}")
print("✓ Classes are now balanced!")

# ============================================================================
# STEP 7: NORMALIZE FEATURES
# ============================================================================
print("\n[STEP 7] Normalizing features...")

"""
WHY: Features have different scales:
- age: 0-100
- avg_glucose_level: 50-300
- bmi: 10-60

Normalization scales everything to similar ranges (mean=0, std=1).
This helps both classical and quantum algorithms learn better.
"""

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_balanced)

print("✓ Features normalized (mean=0, std=1)")
print("\nNormalized data statistics:")
print(f"  Mean: {X_normalized.mean(axis=0)}")
print(f"  Std: {X_normalized.std(axis=0)}")

# ============================================================================
# STEP 8: SPLIT INTO TRAIN AND TEST SETS
# ============================================================================
print("\n[STEP 8] Splitting data into train and test sets...")

"""
WHY: We need separate data to train the model and test its performance.
If we test on the same data we trained on, we'd get falsely high accuracy
(the model would have "memorized" the answers).

We use 80% for training and 20% for testing.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_balanced, 
    test_size=0.2, 
    random_state=42,
    stratify=y_balanced  # Keep class balance in both sets
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 9: SAVE PROCESSED DATA
# ============================================================================
print("\n[STEP 9] Saving processed data...")

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save the processed data
np.save('results/X_train.npy', X_train)
np.save('results/X_test.npy', X_test)
np.save('results/y_train.npy', y_train)
np.save('results/y_test.npy', y_test)

# Save feature names and scaler for later use
import joblib
joblib.dump(selected_features, 'results/selected_features.pkl')
joblib.dump(scaler, 'results/scaler.pkl')

print("✓ Saved processed data to results/ folder:")
print(f"  - X_train.npy: {X_train.shape}")
print(f"  - X_test.npy: {X_test.shape}")
print(f"  - y_train.npy: {y_train.shape}")
print(f"  - y_test.npy: {y_test.shape}")
print(f"  - selected_features.pkl")
print(f"  - scaler.pkl")

# ============================================================================
# BONUS: CREATE VISUALIZATION
# ============================================================================
print("\n[BONUS] Creating visualization of processed data...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Processed Data - Feature Distributions', fontsize=16)

for idx, feature in enumerate(selected_features):
    row = idx // 3
    col = idx % 3
    
    # Plot distribution for stroke vs non-stroke
    stroke_data = X_train[y_train == 1, idx]
    no_stroke_data = X_train[y_train == 0, idx]
    
    axes[row, col].hist([no_stroke_data, stroke_data], 
                        bins=30, 
                        label=['No Stroke', 'Stroke'],
                        alpha=0.7)
    axes[row, col].set_title(feature)
    axes[row, col].legend()
    axes[row, col].set_xlabel('Normalized Value')
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved feature distribution plot to results/feature_distributions.png")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)
print("\nNext step: Run '2_classical_models.py' to train classical ML models")