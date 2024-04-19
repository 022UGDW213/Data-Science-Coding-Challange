

Explore, Clean, Validate, and Visualize the Data (optional)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()

# Data Visualization
plt.figure(figsize=(10, 6))
sns.distplot(train_df['MonthlyCharges'], bins=30, kde=True)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.show()


Make predictions (required)
Remember you should create a dataframe named prediction_df with exactly 104,480 entries plus a header row attempting to predict the likelihood of churn for subscriptions in test_df. Your submission will throw an error if you have extra columns (beyond CustomerID and predicted_probaility) or extra rows.

The file should have exactly 2 columns: CustomerID (sorted in any order) predicted_probability (contains your numeric predicted probabilities between 0 and 1, e.g. from estimator.predict_proba(X, y)[:, 1])

The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of prediction_df with column names CustomerID and predicted_probability!

Example prediction submission:
The code below is a very naive prediction method that simply predicts churn using a Dummy Classifier. This is used as just an example showing the submission format required. Please change/alter/delete this code below and create your own improved prediction methods for generating prediction_df.

PLEASE CHANGE CODE BELOW TO IMPLEMENT YOUR OWN PREDICTIONS




# Import pandas library
import pandas as pd

# Replace `predicted_probability` with your actual predicted probabilities
# Use your classifier to make predictions on test_df using `predict_proba` method
predicted_probability = your_classifier.predict_proba(test_df.drop(['CustomerID'], axis=1))[:, 1]

# Combine predictions with the 'CustomerID' column into a DataFrame
prediction_df = pd.DataFrame({'CustomerID': test_df['CustomerID'], 
                              'predicted_probability': predicted_probability})