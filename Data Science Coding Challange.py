# Data Science Coding Challenge — starter script
# Churn prediction for a video streaming company: predict which subscribers
# will continue their subscription for another month.

# --- Explore, Clean, Validate, and Visualize the Data (optional) ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
print(train_df.head())

# Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(train_df['MonthlyCharges'], bins=30, kde=True)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.show()


# --- Make predictions (required) ---
# Create a dataframe named prediction_df with exactly 104,480 entries plus a
# header row, predicting the likelihood of churn for subscriptions in test_df.
# The file must have exactly 2 columns: CustomerID and predicted_probability
# (numeric predicted probabilities between 0 and 1, e.g. from
# estimator.predict_proba(X)[:, 1]).
#
# Example prediction submission (naive — replace with your own model):
#   from sklearn.dummy import DummyClassifier
#   clf = DummyClassifier(strategy="prior").fit(X_train, y_train)
#   predicted_probability = clf.predict_proba(test_df.drop(['CustomerID'], axis=1))[:, 1]
#   prediction_df = pd.DataFrame({'CustomerID': test_df['CustomerID'],
#                                 'predicted_probability': predicted_probability})
#   prediction_df.to_csv('predictions.csv', index=False)
