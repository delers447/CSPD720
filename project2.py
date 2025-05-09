#! /usr/bin/python3

#Project 3 for CSDP 720 by Dan Rice

import pandas as pd
import math

# for data splitting, transforming and model training
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.feature_selection import f_regression
from sklearn import preprocessing 

def simple_Linear_regression():
	datafile = "IRIS.csv"
	df = pd.read_csv(datafile)
	print(df.to_string())
	X = df[["sepal_length"]]
	Y  = df[["petal_length"]]

	reg_model = LinearRegression().fit(X, Y)

	#(b - bias)
	bo = reg_model.intercept_[0]
	b1 = reg_model.coef_[0][0]

	def simple_regression(x):
		return bo + b1*x

	print(f"The lineare gression model is: y = {bo} + {b1}x")
	print(f"The correlation coeffieicient is {math.sqrt(reg_model.score(X, Y))} and r^2 = {reg_model.score(X, Y)}.")
	print(f"Adjusted R^2 is {1 - (1-reg_model.score(X, Y))*(len(Y)-1)/(len(Y)-X.shape[1]-1)}")
	print(f"The estimated petal length when sepal length is {7.5} is {simple_regression(7.5)}")


def multiple_linear_regression():
	datafile = "IRIS.csv"
	df = pd.read_csv(datafile)
	print(df.to_string())
	X = df[["sepal_length", "sepal_width", "petal_width"]]
	Y  = df[["petal_length"]]

	reg_model = LinearRegression().fit(X, Y)

	#(b - bias)
	bo = reg_model.intercept_[0]
	b1 = reg_model.coef_[0][0]
	b2 = reg_model.coef_[0][1]
	b3 = reg_model.coef_[0][2]

	

	def regression(x1, x2, x3):
		return bo + b1*x1 + b2*x2 + b3*x3

	print(f"The lineare gression model is: y = {bo} + {b1}x1 + {b2}x2 + {b3}x3")
	print(f"The correlation coeffieicient is {math.sqrt(reg_model.score(X, Y))} and r^2 = {reg_model.score(X, Y)}.")
	print(f"Adjusted R^2 is {1 - (1-reg_model.score(X, Y))*(len(Y)-1)/(len(Y)-X.shape[1]-1)}")
	print(f"For a specimen with (7.5, 2.5, 3.5) for (sepal_length, sepal_width, petal_width) respectively,")
	print(f"The estimated petal length is: {regression(7.5, 2.5, 3.5)}")
	print(f"Ie, F( 7.5, 2.5, 3.5) = {regression(7.5, 2.5, 3.5)}")

	f_statistic, p_values = f_regression(X, Y)
	print(f"The F-score for sepal_length is {f_statistic[0]} with a p-value of {p_values[0]}.")
	print(f"The F-score for sepal_width is {f_statistic[1]} with a p-value of {p_values[1]}.")
	print(f"The F-score for petal_width is {f_statistic[2]} with a p-value of {p_values[2]}.")
	print(f"All three independent variables are statisticall significant.")

simple_linear_regression()
multiple_linear_regression()
