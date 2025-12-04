"""
train_models.py
----------------
Trains classical ML models (SVM, KNN, Random Forest)
on the UCI Daily and Sports Activities dataset using
preprocessed flattened features.

This script:
 - Loads preprocessed dataset (CSV)
 - Scales features
 - Applies PCA (50 components)
 - Performs 10-fold stratified cross-validation
 - Trains 3 classifiers
 - Saves accuracy and F1 scores
 - Saves confusion matrices as PNG files
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
DATA_PATH = "processed_dataset.csv"   # <-- THIS MUST MATCH preprocess.py OUTPUT

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).values
y = df["label"].values


# -----------------------------------------------------------
# Scale features
# -----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------------------------------------
# PCA dimensionality reduction
# -----------------------------------------------------------
pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA transformed shape: {X_pca.shape}")
print(f"Total Variance Explained: {pca.explained_variance_ratio_.sum():.4f}")


# -----------------------------------------------------------
# Classifiers
# -----------------------------------------------------------
models = {
    "SVM": LinearSVC(C=1.0, max_iter=10000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42)
}

# -----------------------------------------------------------
# Evaluation using Stratified 10-fold CV
# -----------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}

for model_name, model in models.items():
    accuracies = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X_pca, y):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, preds))
        f1_scores.append(f1_score(y_test, preds, average='macro'))

    results[model_name] = {
        "Accuracy": np.mean(accuracies),
        "F1 Score": np.mean(f1_scores)
    }

    print(f"{model_name} Accuracy: {results[model_name]['Accuracy']:.4f}")
    print(f"{model_name} F1 Score: {results[model_name]['F1 Score']:.4f}")


# -----------------------------------------------------------
# Train-test confusion matrices (for appendix)
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.25, random_state=42, stratify=y
)

os.makedirs("plots", exist_ok=True)

for name, model in models.items():

    # Train on train split
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Save image
    out_path = f"plots/{name.lower()}_cm.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------------------------------------
# Save numerical results
# -----------------------------------------------------------
results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv", index=True)

print("\nFinal model results saved to model_results.csv")
print("Confusion matrices saved in /plots/")
