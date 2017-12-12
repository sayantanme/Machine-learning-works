# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#Fit simple linear regressor to trainining set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predictingon the test result set
y_pred = regressor.predict(X_test)

#plotting and visualising on training set
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Experience vs Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#plotting and visualising on test set
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Experience vs Experience(Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
