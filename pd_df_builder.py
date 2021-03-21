# build initial df using pandas_datareader
# date range



#qunadl, tiingo, pandas_datareader

import yahoo_fin as yf 

import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
import numpy as np
date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')


timestamp_date_rng = pd.to_datetime(date_rng, infer_datetime_format=True)

# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf
import matplotlib.pyplot as plt



def multi_stock():
	tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']
	data = yf.download(tickers_list,'2015-1-1')['Adj Close']
	print(data.head())

	def plot():
		((data.pct_change()+1).cumprod()).plot(figsize=(10, 7))
		plt.legend()
		plt.title("Returns", fontsize=16)
		plt.ylabel('Cumulative Returns', fontsize=14)
		plt.xlabel('Year', fontsize=14)
		plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
		plt.show()
	#plot()
	return data

df = multi_stock()


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()
'''
mu will then be a pandas series of estimated expected returns for each asset,
and S will be the estimated covariance matrix (part of it is shown below):
'''
print(mu)
print(S)


