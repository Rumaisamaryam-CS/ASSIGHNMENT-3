#MACHINE LEARNING ASSIGNMENT3
#RUMAISA MARYAM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('annual_temp.csv')

#FOR THE FIRST INDUSTRY GCAG
A = dataset.loc[(dataset.Source == 'GCAG'), ['Year']]
B = dataset.loc[(dataset.Source == 'GCAG'), ['Mean']]

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.2, random_state = 0)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4)
A_poly = polyreg.fit_transform(A)
polyreg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the Linear Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, reg.predict(A), color = 'Black')
plt.title('Years VS Annual Temperature For GCAG (Linear Regression)')
plt.xlabel('Years')
plt.ylabel('Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_reg_2.predict(polyreg.fit_transform(A)), color = 'Black')
plt.title('Year VS Annual TemperatureFor GCAG(Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()

# Predicting a new result with Linear Regression
X=reg.predict([[2016]])
X1=reg.predict([[2017]])
print("The result with linear regression for GCAG in 2016 is" , X)
print("The result with linear regression for GCAG in 2017 is" , X1)

# Predicting a new result with Polynomial Regression
Y=lin_reg_2.predict(polyreg.fit_transform([[2016]]))
Y1=lin_reg_2.predict(polyreg.fit_transform([[2017]]))
print("The result with polynomial regression for GCAG in 2016  is" ,  Y)
print("The result with polynomial regression for GCAG in 2017 is" ,  Y1)


#FOR THE SECOND INDUSTRY GISTEMP
A1 = dataset.loc[(dataset.Source == 'GISTEMP'), ['Year']]
B1 = dataset.loc[(dataset.Source == 'GISTEMP'), ['Mean']]

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
A1_train, A1_test, B1_train, B1_test = train_test_split(A1, B1, test_size = 0.2, random_state = 0)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A1, B1)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4)
A1_poly = polyreg.fit_transform(A1)
polyreg.fit(A1_poly, B1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A1_poly, B1)

# Visualising the Linear Regression results
plt.scatter(A1, B1, color = 'red')
plt.plot(A1, reg.predict(A1), color = 'Black')
plt.title('Years VS Annual Temperature For GISTEMP (Linear Regression)')
plt.xlabel('Years')
plt.ylabel('Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(A1, B1, color = 'red')
plt.plot(A1, lin_reg_2.predict(polyreg.fit_transform(A1)), color = 'Black')
plt.title('Years VS Annual Temperature For GISTEMP(Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('Temperature')
plt.show()

# Predicting a new result with Linear Regression
X2=reg.predict([[2016]])
X3=reg.predict([[2017]])
print("The result with linear regression for GCAG in 2016 is" , X2)
print("The result with linear regression for GCAG in 2017 is" , X3)

# Predicting a new result with Polynomial Regression
Y2=lin_reg_2.predict(polyreg.fit_transform([[2016]]))
Y3=lin_reg_2.predict(polyreg.fit_transform([[2017]]))
print("The result with polynomial regression for GCAG in 2016  is" ,  Y2)
print("The result with polynomial regression for GCAG in 2017 is" ,  Y3)