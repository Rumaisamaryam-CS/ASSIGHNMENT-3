#MACHINE LEARNING ASSIGNMENT3
#RUMAISA MARYAM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Annual Temperature dataset
dataset = pd.read_csv('housing_price.csv')
A = dataset.iloc[:,0:1].values
B = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.2, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
A_poly = poly_reg.fit_transform(A)
poly_reg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the Linear Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_regressor.predict(A), color = 'Black')
plt.title('Id VS House Price (Linear Regression)')
plt.xlabel('Id')
plt.ylabel('House Price')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_reg_2.predict(poly_reg.fit_transform(A)), color = 'Black')
plt.title('Id VS House Price (Polynomial Regression)')
plt.xlabel('Id')
plt.ylabel('House Price')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
A_grid = np.arange(min(A), max(A), 0.1)
A_grid = A_grid.reshape((len(A_grid), 1))
plt.scatter(A, B, color = 'red')
plt.plot(A_grid, lin_reg_2.predict(poly_reg.fit_transform(A_grid)), color = 'Black')
plt.title('Id  VS House Price(Polynomial Regression)')
plt.xlabel('Id')
plt.ylabel('House Price')
plt.show()

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(A, B)

# Visualising the Decision Tree Regression results (higher resolution)
A_grid = np.arange(min(A), max(A), 0.01)
A_grid = A_grid.reshape((len(A_grid), 1))
plt.scatter(A, B, color = 'red')
plt.plot(A_grid, regressor.predict(A_grid), color = 'BLACK')
plt.title('Id  VS House Price (Decision Tree Regression)')
plt.xlabel('Id')
plt.ylabel('House Price')
plt.show()

# Predicting a new result with Linear Regression
X=lin_regressor.predict([[2920]])
print("The result with linear regression for house id 2920 is" , X)


# Predicting a new result with Polynomial Regression
Y=lin_reg_2.predict(poly_reg.fit_transform([[2920]]))
print("The result with polynomial regression for house id 2920  is" ,  Y)


# Predicting a new result with Decision Tree
Z = regressor.predict([[2920]])
Z1= regressor.predict([[2929]])
print("The result with decision tree for house id 2920  is" , Z)
print("The result with decision tree for house id 2929  is" , Z1)

