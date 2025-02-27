#! /usr/bin/python3

from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
import scipy.stats as st 
import random
import statistics

def sample(data, n):
	indexes = []
	for i in range(n):
		indexes.append(random.randint(0,len(data)-1))

	results = []
	for index in indexes:
		results.append(data[index])

	return results

def problem_1():
	males = np.random.normal(180, 20, 10)
	females = np.random.normal(150, 15, 10)
	weights = list(males) + list(females)

	jackknife_means = []
	for i in range(1000):
		random.shuffle(weights)
		jackknife_means.append(statistics.mean(weights[:-5]))

	jackknife_means.sort()
	print("\nProblem 1, Part A")
	print(f"The jackknife confidence interval for the weights is: ({jackknife_means[24]}, {jackknife_means[975]})")

	bootstrap_means = []
	for i in range(1000):
		sample_data = sample(weights, 15)
		bootstrap_means.append(statistics.mean(sample_data))

	bootstrap_means.sort()
	print("\nProblem 1, Part B")
	print(f"The boot strapped confidence interval for the weights is: ({bootstrap_means[24]}, {bootstrap_means[975]})")

	print("\nProblem 1, Part C")
	print("""The confidence intervals for the boot strapped method are wider than that of the jackknife.""")

	print("\nProblem 1, Part D")
	print("""The boot strap would be good to estimate the distirbution of a statistics, 
		while the jackknife wold be better to estimate the bias and variance of a statistics.

		For example, I would estimate the average loss of a training session with the boot strap 
		while I would train the variation of the less with the jack knife.  
		Both being good since I don't have the sampling distribution but can be estimated 
		with either sampling methods.""")

def problem_2_bivariate():
	#Bivariate Solution
	x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	y = [45.23, 58.45, 32.67, 41.39, 50.12, 
			61.12, 48.90, 39.54, 54.21, 62.34]

	plt.plot(x, y)
	plt.axis([0, 11, 0, 75])

	x = sm.add_constant(x)

	quantiles = [0.025, .50, 0.975]
	models = []
	for tau in quantiles:
		model = sm.QuantReg(y, x).fit(q=tau)
		models.append(model)

	#Plot the regression lines for different quantiles
	plt.scatter(x[:, 1], y, alpha=0.6, label='Data')
	for i, tau in enumerate(quantiles):
	    plt.plot(x[:, 1], models[i].fittedvalues, label=f'Quantile {tau}')

	plt.title("Quantile Regression")
	plt.xlabel("Transaction ID")
	plt.ylabel("Amount Spent (USD)")
	plt.legend()

	print(f"\nProblem 2, Part c")
	print("""The data presentation is clearly bi variate, 
		so constructued the distribution of the 0.25 and 0.975 quantiles
		to depict the confidence range.
		Thee is an overall postive relationship between the two variables
		but Simpson Paradox could used for the subsections
		which exhibit a negative correlation.""")

	plt.show()

def problem_2_univariate():
	# Data (amount spent by customers)
	y = np.array([45.23, 58.45, 32.67, 41.39, 50.12, 61.12, 48.90, 39.54, 54.21, 62.34])


	X = np.ones(len(y)) 
	X = sm.add_constant(X) 

	model = sm.QuantReg(y, X)
	result = model.fit(q=0.5)  

	# Compute confidence intervals for the coefficients
	conf_int = result.conf_int(alpha=0.05)  # 95% confidence interval (alpha = 0.05)

	# Output the confidence intervals
	print("\n Problem 2, Part a")
	print(f"95% Confidence Intervals for the regression coefficients:")
	print(conf_int)

	#Traditional Method, 1-sample t interval.
	interval = st.t.interval(0.95, df=len(y)-1, loc=np.mean(y), scale=st.sem(y))
	print(f"\n Problem 2, Part b")
	print(f"95% Confidence Intervals for the mean is:")
	print(interval)

	print("""The use of quantile regression to create a confidence interval 
		for a uni-variate situation does not really compare to the one sample t-interval.""")


problem_1()
problem_2_univariate()
problem_2_bivariate()