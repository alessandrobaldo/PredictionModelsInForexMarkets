import pandas as pd
import numpy as np

def EMA(data, windowSize):
    return data.ewm(span=windowSize,min_periods=windowSize,adjust=False).mean()

def SMA(data, windowSize):
    return data.rolling(window=windowSize,min_periods=windowSize).mean()

def WMA(data, windowSize):
    weights = list(range(1,windowSize+1))
    denom = np.sum(weights)
    return (data.rolling(window=windowSize, min_periods=windowSize).apply(lambda x: np.sum(weights*x) / denom, raw=False))

def HMA(data, windowSize):
    return WMA((2*WMA(data, int(windowSize/2)) - WMA(data,windowSize)), int(np.sqrt(windowSize)))

def MACD(close):
    return EMA(close,12) - EMA(close,26)

def TypicalPrice(high,low,close):
    return (high+low+close)/3

def CCI(typicalPrice):
    smatp = EMA(typicalPrice,20)
    avgDev = typicalPrice.rolling(window=20,min_periods=20).std()
    return (typicalPrice - smatp)/(0.015*avgDev)

def StochasticOscillator(close, high, low):
    return 100*(close - low.rolling(window = 14, min_periods=14).min())/(high.rolling(window = 14,min_periods=14).max() -low.rolling(window = 14, min_periods=14).min())

def RSI(close):
    delta = close.diff()
    up = delta.copy()
    up[delta<=0] = 0.0
    down = abs(delta.copy())
    down[delta>0] = 0.0
    roll_up = EMA(up,14)
    roll_down = EMA(down,14)
    RS = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + RS))

def ROC(close, window):
    return 100*close.diff(window-1)/close.shift(window-1)

def PPO(close):
    return 100*MACD(close)/EMA(close,26)

def KST(close):
    RCMA1 = SMA(ROC(close,10),10)
    RCMA2 = SMA(ROC(close,15),10)
    RCMA3 = SMA(ROC(close,20),10)
    RCMA4 = SMA(ROC(close,30),15)
    return SMA((RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4),9)

def BollingerBandUp(typicalPrice):
    return SMA(typicalPrice,20) + 2*np.std(typicalPrice.iloc[-20:])

def BollingerBandMiddle(typicalPrice):
    return SMA(typicalPrice,20)

def BollingerBandDown(typicalPrice):
    return SMA(typicalPrice,20) - 2*np.std(typicalPrice.iloc[-20:])
    
def normalizeData(matr, indicators, scaler):
    for j in range(len(indicators)):
        if isinstance(scaler[indicators[j]][0],int):
            minVal = scaler[indicators[j]][0]
        else:
            minVal = np.min(matr[indicators.index(scaler[indicators[j]][0]),:])
        
        if isinstance(scaler[indicators[j]][1],int):
            maxVal = scaler[indicators[j]][1]
        else:
            maxVal = np.max(matr[indicators.index(scaler[indicators[j]][1]),:])
        
        matr[j,:] = (matr[j,:] - minVal) / (maxVal - minVal)
    #print(matr)
    return matr
        