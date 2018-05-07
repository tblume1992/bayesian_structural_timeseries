# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:03:23 2018

@author: t-blu
"""


import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter 
import seaborn as sns
import statsmodels.api as sm
sns.set(color_codes=True)

df = pd.read_csv('C:\\Users\\t-blu\\Downloads\\advertising-and-sales-data-36-co.csv')

df = df.drop(df.index[len(df)-1])
train = df.head(24)
test = df.tail(12)



year_1_sales = pd.DataFrame(train.iloc[:,2].head(12))
year_2_sales = pd.DataFrame(train.iloc[:,2].tail(12))
year_1_sampled_means = []
for i in range(1000):
    year_1_sampled_means.append(np.mean(np.random.choice(year_1_sales.iloc[:,0],100, replace=True)))
year_2_sampled_means = []
for i in range(1000):
    year_2_sampled_means.append(np.mean(np.random.choice(year_2_sales.iloc[:,0],100, replace=True))) 
    
    

year_1_avg = float(np.mean(year_1_sales))
year_2_avg = float(np.mean(year_2_sales))
year_1_sales['Time'] = year_1_sales.index.values+1
year_2_sales['Time'] = year_2_sales.index.values+1
trend = year_2_avg - year_1_avg
trend_distribution = np.std(year_1_sampled_means) + np.std(year_2_sampled_means)
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=2)
filtered_sales_1 = pd.DataFrame(kf.em(year_1_sales).smooth(year_1_sales)[0])
filtered_sales_2 = pd.DataFrame(kf.em(year_2_sales-trend).smooth(year_2_sales-trend)[0])
filtered_sales = filtered_sales_1.append(filtered_sales_2).reset_index(drop = True)






sales = pd.DataFrame(train.iloc[:,2])

sales['Time'] = train.index.values+1

plt.plot(sales.iloc[:,0],)
plt.plot(filtered_sales)
plt.axhline(year_1_avg, color='k', linestyle='dashed', linewidth=1, label = 'Mean')
plt.axhline(year_2_avg, color='red', linestyle='dashed', linewidth=1, label = 'Mean')
plt.fill_between(sales.index.values, np.squeeze(np.min(year_1_sampled_means)),np.squeeze(np.max(year_1_sampled_means)), alpha = .2, color = 'black')
plt.fill_between(sales.index.values, np.squeeze(np.min(year_2_sampled_means)),np.squeeze(np.max(year_2_sampled_means)), alpha = .2, color = 'red')
plt.show()


sales = sales.drop(['Time'], axis = 1)
stationary_sales = np.asarray(sales) - np.asarray(filtered_sales)
y = (stationary_sales)

x = (train.iloc[:,1])
data = dict(x=x, y=y)
with pm.Model() as model:
    pm.glm.GLM.from_formula('y ~ x', data)
    trace = pm.sample(5500, cores=2)
coefficients = pm.trace_to_dataframe(trace).iloc[:,1].tail(500)
look = pm.trace_to_dataframe(trace).iloc[:,0].tail(500)
intercept = np.mean(look)
expected_coefficient = np.mean(coefficients)
upper_bound = np.max(coefficients)
lower_bound = np.min(coefficients)
sns.distplot(coefficients, rug=True)
plt.axvline(coefficients.mean(), color='k', linestyle='dashed', linewidth=1, label = 'Mean')
plt.axvline(coefficients.median(), color='r', linestyle='dashed', linewidth=1, label = 'Median')
plt.legend();
plt.show()
test_marketing = test.iloc[:,1]
seasonality = np.squeeze(np.asarray((.2*filtered_sales_1+.8*filtered_sales_2)))
prediction = np.asarray(test_marketing*(expected_coefficient)) + (seasonality) + trend + intercept
upper_prediction = np.asarray(upper_bound*test_marketing) + seasonality + trend + trend_distribution + intercept
lower_prediction = np.asarray(lower_bound*test_marketing) + seasonality + trend -trend_distribution + intercept
upper_prediction = train.iloc[:,2].append(pd.DataFrame(upper_prediction), ignore_index = True)
lower_prediction = train.iloc[:,2].append(pd.DataFrame(lower_prediction), ignore_index = True)

prediction = train.iloc[:,2].append(pd.DataFrame(prediction), ignore_index = True)

plt.plot(prediction, color = 'black', linestyle = 'dashdot', alpha = 1, label = 'Predicted')
plt.fill_between(prediction.index.values, np.squeeze(np.asarray(lower_prediction)),np.squeeze(np.asarray(upper_prediction)), alpha = .2, color = 'grey')
plt.axvline(23, color = 'grey', linestyle = 'dashed')
plt.axvline(11, color = 'grey', linestyle = 'dashed')
append_me = pd.DataFrame(np.zeros([24,12]))
predictions = pd.DataFrame(np.expand_dims(test_marketing, axis = 1) * np.transpose(np.expand_dims(coefficients, axis = 1))+ trend + intercept)
seasonality = pd.DataFrame(np.repeat(np.expand_dims(seasonality, axis = 1),500, axis = 1)) 
predictions = predictions + seasonality


predictions = append_me.append(predictions, ignore_index = True)
plt.plot(predictions, color = 'dimgray', alpha = .01)
plt.plot(df.iloc[:,2], color = 'black', label = 'Actual')
plt.ylim((0, 100)) 
plt.legend()
plt.show()
