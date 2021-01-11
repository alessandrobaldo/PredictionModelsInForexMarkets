import pandas as pd
import numpy as np
from .IndicatorsFunction import *
import time

def generateData(filename, columns, index, indicators):
	data = pd.read_csv(filename, names=columns)
	data.set_index(index)
	
	if indicators:
		windows = [6,20,50,100]

		close = data['Close']
		st = time.time()
		data['Yield'] = (data['Close'] - data['Open'])/(data['Open'])
		end = time.time()
		print("Yield: {}".format(end-st))
		
		st = time.time()
		data['PercentageVolume'] = (data['High'] - data['Low'])*(10**4) / data['Volume']
		end = time.time()
		print("PercVol: {}".format(end-st))
		
		#Moving Averages
		st = time.time()
		for windowSize in windows:
			st = time.time()
			data['SMA{}'.format(windowSize)] = SMA(close,windowSize)
			end = time.time()
			print("SMA{}: {}".format(windowSize,end-st))
			st = time.time()
			data['EMA{}'.format(windowSize)] = EMA(close,windowSize)
			end = time.time()
			print("EMA{}: {}".format(windowSize,end-st))
			st = time.time()
			data['WMA{}'.format(windowSize)] = WMA(close,windowSize)
			end = time.time()
			print("WMA{}: {}".format(windowSize,end-st))
			st = time.time()
			data['HMA{}'.format(windowSize)] = HMA(close,windowSize)
			end = time.time()
			print("HMA{}: {}".format(windowSize,end-st))
		end = time.time()
		print("MAs: {}".format(end-st))
		
		#Oscillators
		#Moving Average Convergence/Divergence
		st = time.time()
		data['MACD'] = MACD(close)
		end = time.time()
		print("MACD: {}".format(end-st))
		
		#Commodity Channel Index
		st = time.time()
		typicalPrice = TypicalPrice(data['High'], data['Low'], close)
		data['CCI'] = CCI(typicalPrice)
		end = time.time()
		print("CCI: {}".format(end-st))
		
		#Stochastic Oscillator
		st = time.time()
		data['Stochastic Oscillator'] = StochasticOscillator(close, data['High'], data['Low'])
		end = time.time()
		print("StochOsc: {}".format(end-st))
		
		#Relative Strength Index
		st = time.time()
		data['RSI'] = RSI(close)
		end = time.time()
		print("RSI: {}".format(end-st))
		
		#Rate of Change
		st = time.time()
		data['ROC'] = ROC(close,12)
		end = time.time()
		print("ROC: {}".format(end-st))
		
		#Percentage Price Oscillator
		st = time.time()
		data['PPO'] = PPO(close)
		end = time.time()
		print("PPO: {}".format(end-st))
		
		#Know Sure Thing
		st = time.time()
		data['KST'] = KST(close)
		end = time.time()
		print("KST: {}".format(end-st))
		

		#Bollinger Bands: Up, Down, Middle
		st = time.time()
		data['BOLU'] = BollingerBandUp(typicalPrice)
		data['BOLD'] = BollingerBandDown(typicalPrice)
		data['BOLM'] = BollingerBandMiddle(typicalPrice)
		end = time.time()
		print("BOLS: {}".format(end-st))
	
	data = data.dropna(axis=0,how='any').round(decimals=6)
	
	return data

def selectData(data,columnsToRemove):
	indicators = data.columns
	indicators = [elem for elem in indicators if elem not in columnsToRemove]
	df = data[indicators]
	return df

def normalizeData(data):
	return (data - data.min())/(data.max() - data.min())

def generateImages(data):
	images = []
	for i in range(data.shape[0]-len(data.columns)):
		img = np.zeros((len(data.columns),len(data.columns)), dtype=float)
		for j in range(len(data.columns)):
			img[:,j] = data.iloc[i:i+len(data.columns), data.columns.get_loc(data.columns[j])]
		img = img.reshape(len(data.columns),len(data.columns),1)
		images.append(img)
	return images

def generateTrainingTest(images, label):
	sets = [
		(images[i:i+365*4],#trainX
		 images[i+365*4:i+365*5],#testX
		 np.array(label)[i:i+365*4],#trainY
		 np.array(label)[i+365*4:i+365*5]) #testY
		for i in range(0,len(images)-365*5,365)
	]
	
	return sets

