# Logistic Regression
# Convertido de notebook para script Python

# ----------------------------------------------------------------------
# # Logistic Regression with Python
#
# Estimated time needed: **30** minutes
#
#
# ## Objectives
#
# After completing this lab you will be able to:
#
# * Use Logistic Regression for classification
# * Preprocess data for modeling
# * Implement Logistic regression on real world data

# ----------------------------------------------------------------------
# ## Install and import the required libraries
# Make sure the required libraries are available by executing the cell below.

# !pip install numpy==2.2.0  # (comando de notebook)
# !pip install pandas==2.2.3  # (comando de notebook)
# !pip install scikit-learn==1.6.0  # (comando de notebook)
# !pip install matplotlib==3.9.3  # (comando de notebook)

# ----------------------------------------------------------------------
# Let's first import required libraries:

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# %matplotlib inline   # (comando de notebook)

import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# ## Classification with Logistic Regression
#
# ### Scenario
# Assume that you are working for a telecommunications company which is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is more likely to leave the company.

# ----------------------------------------------------------------------
# ###  Load the Telco Churn data 
# Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer and it records various demographic and service usage information. Before you can work with the data, you must use the URL to get the ChurnData.csv.

# ----------------------------------------------------------------------
# ### About the dataset
# We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
#
# This data set provides you information about customer preferences, services opted, personal details, etc. which helps you predict customer churn.

# ----------------------------------------------------------------------
# ### Load Data from URL

# churn_df = pd.read_csv("ChurnData.csv")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

churn_df

# ----------------------------------------------------------------------
# Let's select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the scikit-learn algorithm:

# ----------------------------------------------------------------------
# ## Data Preprocessing

# ----------------------------------------------------------------------
# For this lab, we can use a subset of the fields available to develop out model. Let us assume that the fields we use are 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' and of course 'churn'.

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df

# ----------------------------------------------------------------------
# For modeling the input fields X and the target field y need to be fixed. Since that the target to be predicted is 'churn', the data under this field will be stored under the variable 'y'. We may use any combination or all of the remaining fields as the input. Store these values in the variable 'X'.

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  #print the first 5 values

y = np.asarray(churn_df['churn'])
y[0:5] #print the first 5 values

# ----------------------------------------------------------------------
# It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale. This helps the model learn faster and improves the model performance. We may make use of StandardScalar function in the Scikit-Learn library.

X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]

# ----------------------------------------------------------------------
# ### Splitting the dataset

# ----------------------------------------------------------------------
# The trained model has to be tested and evaluated on data which has not been used during training. Therefore, it is required to separate a part of the data for testing and the remaining for training. For this, we may make use of the train_test_split function in the scikit-learn library.

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

# ----------------------------------------------------------------------
# ## Logistic Regression Classifier modeling

# ----------------------------------------------------------------------
# Let's build the model using __LogisticRegression__ from the Scikit-learn package and fit our model with train data set.

LR = LogisticRegression().fit(X_train,y_train)

# ----------------------------------------------------------------------
# Fitting, or in simple terms training, gives us a model that has now learnt from the traning data and can be used to predict the output variable. Let us predict the churn parameter for the test data set.

yhat = LR.predict(X_test)
yhat[:10]

# ----------------------------------------------------------------------
# To understand this prediction, we can also have a look at the prediction probability of data point of the test data set. Use the function __predict_proba__ , we can get the probability of each class. The first column is the probability of the record belonging to class 0, and second column that of class 1. Note that the class prediction system uses the threshold for class prediction as 0.5. This means that the class predicted is the one which is most likely.

yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]

# ----------------------------------------------------------------------
# Since the purpose here is to predict the 1 class more acccurately, you can also examine what role each input feature has to play in the prediction of the 1 class. Consider the code below.

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# ----------------------------------------------------------------------
# Large positive value of LR Coefficient for a given field indicates that increase in this parameter will lead to better chance of a positive, i.e. 1 class. A large negative value indicates the opposite, which means that an increase in this parameter will lead to poorer chance of a positive class. A lower absolute value indicates weaker affect of the change in that field on the predicted class. Let us examine this with the following exercises.

# ----------------------------------------------------------------------
# ## Performance Evaluation

# ----------------------------------------------------------------------
# Once the predictions have been generated, it becomes prudent to evaluate the performance of the model in predicting the target variable. Let us evaluate the log-loss value.
#
# ### log loss
#
# Log loss (Logarithmic loss), also known as Binary Cross entropy loss, is a function that generates a loss value based on the class wise prediction probabilities and the actual class labels. The lower the log loss value, the better the model is considered to be.

log_loss(y_test, yhat_prob)

# ----------------------------------------------------------------------
# ## Practice Exercises
# Try to attempt the following questions yourself based on what you learnt in this lab.
#
# a. Let us assume we add the feature 'callcard' to the original set of input features. What will the value of log loss be in this case?
#
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.6039104035600186.
#
# b. Let us assume we add the feature 'wireless' to the original set of input features. What will the value of log loss be in this case?
#
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.7227054293985518.
#
# c. What happens to the log loss value if we add both "callcard" and "wireless" to the input features?
#
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.7760557225417114
#
# d. What happens to the log loss if we remove the feature 'equip' from the original set of input features?
#
# Hint
# Reuse all the code statements above after modifying the value of churn_df Make sure to edit the list of features feeding the variable X. The expected answer is 0.5302427350245369
#
# e. What happens to the log loss if we remove the features 'income' and 'employ' from the original set of input features?
#
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.6529317169884828.

# ----------------------------------------------------------------------
# ### Congratulations! You're ready to move on to your next lesson!
#
#
# ## Author
#
# Abishek Gagneja
#
#
#  ### Other Contributors
#
# Jeff Grossman
#
# <!--
# ## Change Log
#
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# |2024-11-05 | 3.0 | Abhishek | Updated the descriptions, codes and lab flow | 
# |2021-01-21  | 2.2  | Lakshmi  |  Updated sklearn library|
# | 2020-11-03  | 2.1  | Lakshmi  |  Updated URL of csv |
# | 2020-08-27  | 2.0  | Lavanya  |  Moved lab to course repo in GitLab |
# |   |   |   |   |
# |   |   |   |   |

# ----------------------------------------------------------------------
# © IBM Corporation. All rights reserved.
