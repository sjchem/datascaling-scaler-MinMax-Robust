# datascaling-scaler-MinMax-Robust
Data scaling and comparing both KNN (distance-based) and Logistic Regression (gradient-based)
Comparing both KNN (distance-based) and Logistic Regression (gradient-based) will clearly show how different scalers affect models differently.

Here’s what the notebook will do step by step:
Load any CSV dataset you provide.
Ask you (or define) the target column.
Split data into train/test sets.
Apply three scalers → StandardScaler, MinMaxScaler, RobustScaler.
Train and evaluate both KNN and Logistic Regression models for each scaler.
Show a bar chart comparing accuracies (or another metric).
Print a summary table of results.

# How you can run
You can open this in Jupyter or Colab and simply:
Replace the sample Iris dataset with your own CSV (df = pd.read_csv('yourfile.csv')),
Set the correct TARGET column name,
Run all cells to see accuracy and cross-validation performance for each scaler-model pair.
