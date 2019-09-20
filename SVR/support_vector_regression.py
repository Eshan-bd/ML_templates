"""
Created on Mon Aug 19 02:03:30 2019

@author: Eshan
"""
#%%

# ## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% [markdown]
# ## Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(len(y), 1))
y = y.ravel()
#%%
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]) ))
y_pred = sc_y.inverse_transform(y_pred)

#%%
# Visualizing the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#%%
# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, -1)))

