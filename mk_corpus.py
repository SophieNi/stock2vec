import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def mk_corpse(inputFilePath, outputFilePath):
	df = pd.read_csv(inputFilePath)
	df = df[~df.isnull().any(axis=1)]

	df_open = df.pivot(index='Time', columns='TICKER', values='OPEN.PRICE')
	df_close = df.pivot(index='Time', columns='TICKER', values='CLOSE.PRICE')
	# get difference in open and close price for each ticker
	df_diff = np.subtract(df_open, df_close)
	# substitute na values with the value in previous dates
	df_diff = df_diff.fillna(method = 'ffill')

	# set rolling window size to 20
	window_size=20
	window_end = df_diff.shape[0]-window_size
	for tckr_name in df_diff.columns:
	    tckr_vals = df_diff[tckr_name]
	    for i in range(window_end):
	        ticker_window_vals = tckr_vals.iloc[i:(i+window_size)]
	        universe_window_vals = df_diff.iloc[i:(i+window_size):]
	        # for each day, find the correlation between ticker and other tickers
	        top_corrs = universe_window_vals.apply(lambda x : pearsonr(x, ticker_window_vals)[0]).nlargest(21).tail(20)
	        # append ['Ticker', 'Date', 'Top 20 tickers', 'Top 20 cor vals'] to results
	        out.append([tckr_name, df_diff.index[(i+window_size)], str(list(top_corrs.index)), str(list(top_corrs.values))])

	# output        
	out = pd.DataFrame(out, columns=['Ticker', 'Date', 'Top 20 tickers', 'Top 20 cor vals'])
	out.to_csv(outputFilePath)

mk_corpse(inputFilePath, outputFilePath)