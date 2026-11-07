# Regression Trees Taxi Tip
# Convertido de notebook para script Python

# ----------------------------------------------------------------------
# # **Regression Trees**

# ----------------------------------------------------------------------
# Estimated time needed: **30** minutes

# ----------------------------------------------------------------------
# In this exercise session you will use a real dataset to train a regression tree model. The dataset includes information about taxi tip and was collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). You will use the trained model to predict the amount of tip paid.

# ----------------------------------------------------------------------
# ## Objectives

# ----------------------------------------------------------------------
# After completing this lab you will be able to:

# ----------------------------------------------------------------------
# * Perform basic data preprocessing using Scikit-Learn
# * Model a regression task using Scikit-Learn
# * Train a Decision Tree Regressor model
# * Run inference and assess the quality of the trained models

# ----------------------------------------------------------------------
# Introduction
# The dataset used in this exercise session is a subset of the publicly available TLC Dataset (all rights reserved by Taxi & Limousine Commission (TLC), City of New York). The prediction of the tip amount can be modeled as a regression problem. To train the model you can use part of the input dataset and the remaining data can be used to assess the quality of the trained model.

# ----------------------------------------------------------------------
# Import Libraries

# ----------------------------------------------------------------------
# Make sure the libraries required are available by executing the cell below.

# !pip install numpy  # (comando de notebook)
# !pip install pandas  # (comando de notebook)
# !pip install matplotlib  # (comando de notebook)
# !pip install scikit-learn  # (comando de notebook)

# ----------------------------------------------------------------------
# Import the libraries we need to use in this lab

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline  # (comando de notebook)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Dataset Analysis

# ----------------------------------------------------------------------
# In this section you will read the dataset in a Pandas dataframe and visualize its content. You will also look at some data statistics.
#
# Note: A Pandas dataframe is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure. For more information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html.

# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data

# ----------------------------------------------------------------------
# Each row in the dataset represents a taxi trip. As shown above, each row has 13 variables. One of the variables is `tip_amount` which will be the target variable. Your objective will be to train a model that uses the other variables to predict the value of the `tip_amount` variable.

# ----------------------------------------------------------------------
# To understand the dataset a little better, let us plot the correlation of the target variable against the input variables.

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# ----------------------------------------------------------------------
# This shows us that the input features `payment_type`, `VendorID`, `store_and_fwd_flag` and `improvement_surcharge` have little to no correlation with the target variable.

# ----------------------------------------------------------------------
# Dataset Preprocessing

# ----------------------------------------------------------------------
# You will now prepare the data for training by applying normalization to the input features.

# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

# ----------------------------------------------------------------------
# Dataset Train/Test Split

# ----------------------------------------------------------------------
# Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------------------------------------------------
# Build a Decision Tree Regressor model with Scikit-Learn

# ----------------------------------------------------------------------
# Regression Trees are implemented using `DecisionTreeRegressor`.
#
# The important parameters of the model are:
#
# `criterion`: The function used to measure error, we use 'squared_error'.
#
# `max_depth` - The maximum depth the tree is allowed to take; we use 8.

# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8,
                               random_state=35)

# ----------------------------------------------------------------------
# Now lets train our model using the `fit` method on the `DecisionTreeRegressor` object providing our training data

dt_reg.fit(X_train, y_train)

# ----------------------------------------------------------------------
# Evaluate the Scikit-Learn and Snap ML Decision Tree Regressor Models

# ----------------------------------------------------------------------
# To evaluate our dataset we will use the `score` method of the `DecisionTreeRegressor` object providing our testing data, this number is the $R^2$ value which indicates the coefficient of determination. We will also evaluate the Mean Squared Error $(MSE)$ of the regression output with respect to the test set target values. High $R^2$ and low $MSE$ values are expected from a good regression model.

# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

# ----------------------------------------------------------------------
# ## Practice

# ----------------------------------------------------------------------
# Q1. What if we change the max_depth to 12? How would the $MSE$ and $R^2$ be affected?

# ----------------------------------------------------------------------
# Q2. Identify the top 3 features with the most effect on the `tip_amount`.

# ----------------------------------------------------------------------
# Q3. Since we identified 4 features which are not correlated with the target variable, try removing these variables from the input set and see the effect on the $MSE$ and $R^2$ value.

# ----------------------------------------------------------------------
# Q4. Check the effect of **decreasing** the `max_depth` parameter to 4 on the $MSE$ and $R^2$ values.
