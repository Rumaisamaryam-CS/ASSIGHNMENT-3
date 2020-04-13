#MACHINE LEARNING ASSIGNMENT3
#RUMAISA MARYAM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_startups.csv')
dataset['Sum']=dataset[['R&D Spend', 'Administration', 'Marketing Spend']].sum(axis=1)

Xc = dataset.loc[(dataset.State=='California'),['Sum']]
Yc = dataset.loc[(dataset.State=='California'),['Profit']]


# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
Xc_train, Xc_test, Yc_train, Yc_test = train_test_split(Xc, Yc, test_size = 0.3, random_state = 0)"""
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Xc, Yc)

# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(Xc, Yc, color = 'red')
plt.plot(Xc, regressor.predict(Xc), color = 'blue')
plt.title('California Spending vs Profit (Decision Tree Regression)')
plt.xlabel('Spending')
plt.ylabel('Profit')
plt.show()

# Predicting a new result with Decision Tree
Z = regressor.predict([[9000000]])
print ("California =" ,Z)

Xf = dataset.loc[(dataset.State=='Florida'),['Sum']]
Yf = dataset.loc[(dataset.State=='Florida'),['Profit']]

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
Xf_train, Xf_test, Yf_train, Yf_test = train_test_split(Xf, Yf, test_size = 0.3, random_state = 0)"""
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Xf, Yf)

# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(Xf, Yf, color = 'red')
plt.plot(Xf, regressor.predict(Xf), color = 'blue')
plt.title('Florida Spending vs Profit (Decision Tree Regression)')
plt.xlabel('Spending')
plt.ylabel('Profit')
plt.show()

# Predicting a new result with Decision Tree
Z1= regressor.predict([[9000000]])
print ("Florida = " ,Z1)

print("Hence, the profit of California would be more than Florida.")