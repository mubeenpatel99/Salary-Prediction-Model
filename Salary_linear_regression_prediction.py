# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:47:06 2020

@author: Mubeen Patel
"""
# Data Preprocessing template

# importing libraries
import numpy as np
import matplotlib as plt
import pandas as pd

#importing Datasets
dataset = pd.read_csv('Salary_Data.csv')
experience = dataset.iloc[:, 0].values
salary = dataset.iloc[:, 1].values

# Splitting the data into training and test set
from sklearn.model_selection import train_test_split
experience_train, experience_test, salary_train, salary_test = train_test_split(experience, salary, test_size = 1/3, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
