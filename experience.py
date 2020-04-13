#MACHINE LEARNING ASSIGNMENT3
#RUMAISA MARYAM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Annual Temperature dataset
dataset = pd.read_csv('expense.csv')
A = dataset.iloc[:,0:1].values
B = dataset.iloc[:, 1:2].values


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
plt.title('Monthly Expense VS Income (Linear Regression)')
plt.xlabel('Monthly Expense')
plt.ylabel('Income')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_reg_2.predict(poly_reg.fit_transform(A)), color = 'Black')
plt.title('Months Experience VS Income (Polynomial Regression)')
plt.xlabel('Months experience')
plt.ylabel('Income')
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
plt.title('Months Experience VS Income (Decision Tree Regression)')
plt.xlabel('Months experience')
plt.ylabel('Income')
plt.show()

# Predicting a new result with Linear Regression
X=lin_regressor.predict([[19]])
print("The result with linear regression for 19 months of experince  is" , X)


# Predicting a new result with Polynomial Regression
Y=lin_reg_2.predict(poly_reg.fit_transform([[19]]))
Y1=lin_reg_2.predict(poly_reg.fit_transform([[50]]))
print("The result with polynomial regression for 19 months of experince  is" ,  Y)
print("The result with polynomial regression for 50 months of experince  is" ,  Y1)


# Predicting a new result with Decision Tree
Z = regressor.predict([[19]])
Z1 = regressor.predict([[50]])
print("The result with decision tree for 19 months of experience is" , Z)
print("The result with decision tree for 50 months of experience is" , Z1)


