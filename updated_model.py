# Churn prediction model — Random Forest baseline
# Predicts which video-streaming subscribers will continue their subscription.
# Expects train.csv (with the Churn label) and test.csv in the working directory.
# Neither CSV is included in this repository - see README.md.

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
missing = [name for name in ("train.csv", "test.csv") if not os.path.exists(name)]
if missing:
    raise SystemExit(
        "Missing data file(s): " + ", ".join(missing) + ".\n"
        "The challenge datasets are not part of this repository - obtain train.csv "
        "(with the Churn label) and test.csv and place them in the working directory."
    )
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_ids = test_df["CustomerID"]

# Basic cleaning: fill missing values instead of dropping rows
train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(test_df.median(numeric_only=True))

# One-hot encode categoricals; align test columns exactly to train columns
X = pd.get_dummies(train_df.drop(columns=["CustomerID", "Churn"]))
y = train_df["Churn"]
X_test = pd.get_dummies(test_df.drop(columns=["CustomerID"])).reindex(columns=X.columns, fill_value=0)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale with the same scaler for train, validation, and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Validate
val_predictions = model.predict_proba(X_val_scaled)[:, 1]
print("Validation ROC AUC Score:", roc_auc_score(y_val, val_predictions))

# Predict on the test set and write the submission file
test_predictions = model.predict_proba(X_test_scaled)[:, 1]
prediction_df = pd.DataFrame({"CustomerID": test_ids, "predicted_probability": test_predictions})
prediction_df.to_csv("predictions.csv", index=False)
print("Wrote predictions.csv with", len(prediction_df), "rows")
