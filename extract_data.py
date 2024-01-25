import numpy as np

class stock:
    def __init__(self, open, high, low, close, volume, ticker):
        self.open = open     # list of floats, opening price for each minute
        self.high = high     # list of floats, highest price
        self.low = low       # list of floats, lowest price
        self.close = close   # list of floats, closing price
        self.volume = volume # list of ints, volume
        self.ticker = ticker # string, ticker name

def generateStockList(data, day, startingDay): # returns a list of a list containing stock class, shares, then buys
    data = np.genfromtxt(data, delimiter=',', dtype=None, encoding=None)
    stockList = []
    open = []
    high = []
    low = []
    close = []
    volume = []
    ticker = data[1,7]
    print(ticker)
    temp = stock(open,high,low,close,volume,ticker)

    for i in range(1, len(data)):
        currentDate = data[i,0]
        if(currentDate[8:10] != str(startingDay + day)):
            continue
        else:
            currentTicker = data[i,7]
            if(currentTicker != ticker):
                temp = stock(open,high,low,close,volume,ticker)
                stockList.append(temp)
                open = [data[i,1]]
                high = [data[i,2]]
                low = [data[i,3]]
                close = [data[i,4]]
                volume = [data[i,6]]
                ticker = currentTicker
            else:
                open.append(data[i,1])
                high.append(data[i,2])
                low.append(data[i,3])
                close.append(data[i,4])
                volume.append(data[i,6])
        
    temp = stock(open,high,low,close,volume,ticker)
    stockList.append(temp)
    return stockList

test = generateStockList('data/minute.csv',0,17)
print(test[0].open[0]) # first company, first minute
print([[stock.open[i] for i in range(5)] for stock in test])