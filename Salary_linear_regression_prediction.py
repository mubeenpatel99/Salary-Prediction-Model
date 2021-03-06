# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:47:06 2020

@author: Mubeen Patel
"""
# Data Preprocessing template

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Datasets
dataset = pd.read_csv('Salary_Data.csv')
experience = dataset.iloc[:, 0].values
salary = dataset.iloc[:, 1].values

# Splitting the data into training and test set
from sklearn.model_selection import train_test_split
experience_train, experience_test, salary_train, salary_test = train_test_split(experience, salary, test_size = 1/3, random_state = 0)

# Feature Scaling will be taken care by Linear regression

# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(experience_train.reshape(-1, 1), salary_train)

# Predicting the Test set results
salary_pred = regressor.predict(experience_test.reshape(-1, 1))

# Visualsing the Training set results
plt.scatter(experience_train, salary_train, color='red')
plt.plot(experience_train, regressor.predict(experience_train.reshape(-1,1)), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visulaising the Testing set results
plt.scatter(experience_test, salary_test, color='red')
plt.plot(experience_train, regressor.predict(experience_train.reshape(-1,1)), color='blue')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()