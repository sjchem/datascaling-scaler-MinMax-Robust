# Scaler & Model Comparison Template
# This notebook automatically compares StandardScaler, MinMaxScaler, and RobustScaler
# using KNN and Logistic Regression models on any dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Step 1: Load your dataset
# -------------------------------
# Example: df = pd.read_csv('your_dataset.csv')

print("\n👉 Load your dataset here:")
# df = pd.read_csv('data.csv')  # Uncomment and update this line

# For demo purpose, let's create a small sample dataset (Iris)
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# -------------------------------
# Step 2: Define features and target
# -------------------------------

TARGET = 'target'  # Change this to your target column name
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# Step 3: Define scalers & models
# -------------------------------

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

# -------------------------------
# Step 4: Train and evaluate
# -------------------------------

results = []

for scaler_name, scaler in scalers.items():
    # Fit scaler on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        # Cross-validation (optional)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results.append({
            'Scaler': scaler_name,
            'Model': model_name,
            'Accuracy': acc,
            'CV_Mean': np.mean(cv_scores),
            'CV_STD': np.std(cv_scores)
        })

# -------------------------------
# Step 5: Summarize results
# -------------------------------

results_df = pd.DataFrame(results)
print("\n✅ Model Comparison Summary:\n")
print(results_df.sort_values(by='CV_Mean', ascending=False))

# -------------------------------
# Step 6: Plot comparison
# -------------------------------

plt.figure(figsize=(8,5)) # or make your own size
for model_name in models.keys():
    subset = results_df[results_df['Model'] == model_name]
    plt.bar(subset['Scaler'], subset['CV_Mean'], alpha=0.7, label=model_name)

plt.title('Scaler vs Model Comparison (CV Mean Accuracy)')
plt.ylabel('Mean CV Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\n✨ Done! You can now replace the Iris dataset with your own CSV and re-run.")
print("Just update the TARGET variable and CSV path at the top.")
