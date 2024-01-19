import numpy as np

class stock:
    def __init__(self, open, high, low, close, volume, ticker):
        self.open = open     # list of floats, opening price for each minute
        self.high = high     # list of floats, highest price
        self.low = low       # list of floats, lowest price
        self.close = close   # list of floats, closing price
        self.volume = volume # list of ints, volume
        self.ticker = ticker # string, ticker name

def generateStockList(data): # returns a list of a list containing stock class, shares, then buys
    data = np.genfromtxt(data, delimiter=',', dtype=None, encoding=None)
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
                temp = [stock(open,high,low,close,volume,ticker), 0, 0] # stock, shares, buy price
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
        
    temp = [stock(open,high,low,close,volume,ticker), 0, 0]
    stockList.append(temp)
    return stockList

test = generateStockList('data/minute.csv')
print(test[0][0].ticker) # first company, first minute