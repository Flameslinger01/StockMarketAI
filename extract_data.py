import numpy as np

class stock:
    def __init__(self, open, high, low, close, volume, ticker):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.ticker = ticker

data = np.genfromtxt('/workspaces/StockMarketAI/data/testticker2.csv', delimiter=',', dtype=None, encoding=None)
stockList = []
open = []
high = []
low = []
close = []
volume = []
ticker = 'hola'
temp = stock(open,high,low,close,volume,ticker)

for i in range(1, len(data)):
    currentTicker = data[i,7]
    if(currentTicker != ticker):
        if(ticker != 'hola'):
            temp = stock(open,high,low,close,volume,ticker)
            stockList.append(temp)
        open = []
        high = []
        low = []
        close = []
        volume = []
        ticker = currentTicker
    else:
        open.append(data[i,1])
        high.append(data[i,2])
        low.append(data[i,3])
        close.append(data[i,4])
        volume.append(data[i,6])
    
temp = stock(open,high,low,close,volume,ticker)
stockList.append(temp)

for i in range(4):
    print(stockList[i].ticker)
    print(stockList[i].open[-1])