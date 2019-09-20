"""
Created on Mon Aug 30 08:09:30 2019

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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(len(y), 1))
y = y.ravel()"""
#%%
# Fitting SVR to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#%%
# predicting a new result
y_pred = regressor.predict(np.array([[6.5]]))

#%%
# Visualizing the SVR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decesion Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

