#MACHINE LEARNING ASSIGNMENT3
#RUMAISA MARYAM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
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
plt.title('Year VS co2 produced (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('CO2 Produced')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_reg_2.predict(poly_reg.fit_transform(A)), color = 'Black')
plt.title('Year VS co2 produced (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('CO2 Produced')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
A_grid = np.arange(min(A), max(A), 0.1)
A_grid = A_grid.reshape((len(A_grid), 1))
plt.scatter(A, B, color = 'red')
plt.plot(A_grid, lin_reg_2.predict(poly_reg.fit_transform(A_grid)), color = 'Black')
plt.title('Year VS CO2 Produced(Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('CO2 Produced')
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
plt.title('Year VS CO2 Produced(Decision Tree Regression)')
plt.xlabel('Year')
plt.ylabel('CO2 Produced')
plt.show()

# Predicting a new result with Linear Regression
X=lin_regressor.predict([[2011]])
X1=lin_regressor.predict([[2012]])
X2=lin_regressor.predict([[2013]])
print("The result with linear regression for co2 produced in  2011 is" , X)
print("The result with linear regression for co2 produced in 2012 is" , X1)
print("The result with linear regression for co2 produced in  2013 is" , X2)

# Predicting a new result with Polynomial Regression
Y=lin_reg_2.predict(poly_reg.fit_transform([[2011]]))
Y1=lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
Y2=lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print("The result with polynomial regression for co2 produced in 2011  is" ,  Y)
print("The result with polynomial regression for co2 produced in 2012 is" ,  Y1)
print("The result with polynomial regression for co2 produced in 2013 is" ,  Y2)

# Predicting a new result with Decision Tree
Z = regressor.predict([[2011]])
Z1 = regressor.predict([[2012]])
Z2= regressor.predict([[2013]])
print("The result with decision tree for co2 produced in 2011  is" , Z)
print("The result with decision tree for co2 produced in 2012 is" , Z1)
print("The result with decision tree for co2 produced in 2013  is" , Z2)
