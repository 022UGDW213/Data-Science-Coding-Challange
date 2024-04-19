# Data-Science-Coding-Challange
Imagine that you are a new data scientist at this video streaming company and you are tasked with building a model that can predict which existing subscribers will continue their subscriptions for another month.

### Understanding the Datasets
## Train vs. Test

In this competition, you’ll gain access to two datasets that are samples of past subscriptions of a video streaming platform that contain information about the customer, the customers streaming preferences, and their activity in the subscription thus far. One dataset is titled train.csv and the other is titled test.csv.

train.csv contains 70% of the overall sample (243,787 subscriptions to be exact) and importantly, will reveal whether or not the subscription was continued into the next month (the “ground truth”).

The test.csv dataset contains the exact same information about the remaining segment of the overall sample (104,480 subscriptions to be exact), but does not disclose the “ground truth” for each subscription. It’s your job to predict this outcome!

Using the patterns you find in the train.csv data, predict whether the subscriptions in test.csv will be continued for another month, or not.

## Dataset descriptions

Both train.csv and test.csv contain one row for each unique subscription. For each subscription, a single observation (CustomerID) is included during which the subscription was active.

In addition to this identifier column, the train.csv dataset also contains the target label for the task, a binary column Churn.

Besides that column, both datasets have an identical set of features that can be used to train your model to make predictions. Below you can see descriptions of each feature. Familiarize yourself with them so that you can harness them most effectively for this machine learning task!

import pandas as pd
data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions

## Import Python Modules
# Install pip Requirements Ubuntu Linux
 
First, import the primary modules that will be used in this project. Remember as this is an open-ended project please feel free to make use of any of your favorite libraries that you feel may be useful for this challenge. For example some of the following popular packages may be useful:

# import pandas
# import numpy
# import Scipy
# import Scikit-learn
# import keras
# import maplotlib
# import seaborn
etc, etc

