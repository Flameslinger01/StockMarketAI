# import
import pandas as pd
import yfinance as yf

# main loop
tickerStrings = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB']
df_list = list()
for ticker in tickerStrings:
  data = yf.download(ticker, group_by="Ticker", period='2mo')
  data['ticker'] = ticker
  df_list.append(data)

# combine all dataframes into a single dataframe
df = pd.concat(df_list)

# save to csv
df.to_csv('testticker.csv')