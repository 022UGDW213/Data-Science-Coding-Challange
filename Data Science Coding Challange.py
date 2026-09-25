# Data Science Coding Challenge — starter script
# Churn prediction for a video streaming company: predict which subscribers
# will continue their subscription for another month.

# --- Explore, Clean, Validate, and Visualize the Data (optional) ---

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
if not os.path.exists("train.csv"):
    raise SystemExit(
        "train.csv was not found in the working directory.\n"
        "The challenge datasets are not part of this repository - obtain train.csv "
        "and test.csv and place them next to this script."
    )
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
# Write predictions.csv with a header row plus one row per row of test.csv, and
# exactly 2 columns:
#   CustomerID            - copied unchanged from test.csv
#   predicted_probability - numeric probability in [0, 1] that the subscription
#                           is continued, e.g. estimator.predict_proba(X)[:, 1]
#
# The original challenge text quoted a fixed submission row count (104,480).
# That figure is NOT verifiable from this repository because test.csv is not
# included, so read the real row count from test.csv instead of hard-coding it.
#
# updated_model.py in this repository is a complete, working submission
# (100-tree Random Forest) that writes predictions.csv in exactly this format.
