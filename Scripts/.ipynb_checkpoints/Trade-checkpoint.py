import random
from queue import Queue, PriorityQueue
import numpy as np
import matplotlib.pyplot as plt

def trade(close,high,low,preds, budget = 10000, smooth_factor = 0.5, risk_factor = 5, leverage = 100, margin_call = 50, verbose = True):
	RED = '\033[91m'
	GREEN = '\033[92m'
	BLUE = '\033[94m'
	END = '\033[0m'
	history = {
		"time":[],
		"budget":[],
		"lots":[],
		"time_lots":[]
	}
	
	print(BLUE+"SETTINGS:\n-Initial Budget: {}\n-Smoothing Factor (confidence):{}\n-Risk-Revenue Ratio: 1:{}\n-Leverage: {}x\n-Safety Margin on Budget: {}%".\
		  format(budget, smooth_factor, risk_factor, leverage, margin_call)+END)
	
	entryPoints = []
	for i, elem in enumerate(preds):
		'''
		The prediction tells that the current price could be a local minima, so the price after this timeframe should grow
		A BUY ORDER can be placed
		'''
		if i>0 and i<len(preds)-1 and (preds[i]<preds[i-1] and preds[i]<preds[i+1]):
			entryPoints.append({
				"time":i,
				"O": "BUY",
				"TP": smooth_factor*(preds[i+1]-preds[i]),
				"SL": smooth_factor*(preds[i+1]-preds[i])/risk_factor
			})

		elif i>0 and i<len(preds)-1 and (preds[i]>preds[i-1] and preds[i]>preds[i+1]):
			entryPoints.append({
				"time":i,
				"O": "SELL",
				"TP": smooth_factor*(preds[i]-preds[i+1]),
				"SL": smooth_factor*(preds[i]-preds[i+1])/risk_factor
			})
		'''
		The prediction tells that the current price could be a local maxima, so the price after this timeframe should decrease
		A SELL ORDER can be placed
		'''
		
	
	t = 0
	events = PriorityQueue()
	initialBudget = budget
	lots = max(initialBudget*leverage/1e6, 0.01)
	step_lots = max(0.01, lots/10)
	busyMargin = 0
	
	'''
	Initialize the queue with all the detected entry points 
	'''
	for event in entryPoints:
		events.put(("Start",event["time"],event))
	
	'''
	Simulate until:
	- the budget is not finished
	- the events are not finished
	- the simulation has not reached the present time
	'''
	while budget > 0 and events.qsize()>0 and t<len(close)-1:
		event = events.get()
		t = event[1]
		
		history["time"].append(t)
		history["budget"].append(budget)
		
		if event[0] == "Start":
			if event[2]["O"] == "BUY":
				openOrder = {
					"time": t,
					"O": "BUY",
					"start": close[t],
					"current": close[t],
					"TP": close[t] + event[2]["TP"],
					"SL": close[t] - event[2]["SL"]
				}
			else:
				openOrder = {
					"time": t,
					"O": "SELL",
					"start": close[t],
					"current": close[t],
					"TP": close[t] - event[2]["TP"],
					"SL": close[t] + event[2]["SL"]
				}
			
			'''
			Maximum Lot is determined by the available Margin (the amount of budget above the margin call - the margin occupied by all the open transactions) x leverage_factor
			'''
			maximumLot = max(0,leverage*(budget*(1-margin_call/100)-busyMargin)/1e5)
			if maximumLot != 0:
				lots = min(max(0.01,lots+step_lots), maximumLot)
				history["lots"].append(lots)
				history["time_lots"].append(t)
				busyMargin += lots*1e5*openOrder["start"]/leverage

				if verbose:
					print("\tOpening {} Order at time {}, Lots: {:.2f}, Entry: {:.5f}, TP: {:.5f} ({} pips), SL: {:.5f} ({} pips)".\
						  format(event[2]["O"],t,lots,openOrder["start"],openOrder["TP"],int(abs(openOrder["TP"]-openOrder["start"])*1e4), openOrder["SL"], int(abs(openOrder["SL"]-openOrder["start"])*1e4)))
				events.put(("OpenOrder",t+1,openOrder,lots))
		
		else:
			order = event[2]
			if verbose:
				print("\tMonitoring order open at time {}, Lots: {:.2f}, Entry: {:.5f}, TP: {:.5f}, SL: {:.5f}, Current Price: {:.5f}".format(order["time"],lots,order["start"],order["TP"],order["SL"], close[t]))
			
			if order["O"] == "BUY":
				'''
				BUY ORDER
				If in this new time frame the highest price overwhlemed the TP, it means that the order has been closed and a profit is taken with respect to the last check.
				Conversely, if the lowest price resulted smaller than the SL, it means that the order has been closed and a loss is taken with respect to the last check
				'''
				if low[t] <= order["SL"]:
					lots -= 2*step_lots
					budget -= event[3]*abs(order["SL"]-order["current"])*1e5
					busyMargin -= event[3]*1e5*order["start"]/leverage
					if verbose:
						print("\t"+RED+"LOST:"+END+" {:,.2f}".format(event[3]*abs(order["SL"]-order["start"])*1e5))
				
				elif high[t] >= order["TP"]:
					budget += event[3]*abs(order["TP"]-order["current"])*1e5
					busyMargin -= event[3]*1e5*order["start"]/leverage
					if verbose:
						print("\t"+GREEN+"GAINED:"+END+" {:,.2f}".format(event[3]*abs(order["TP"]-order["start"])*1e5))

				else:
					'''
					The order is still open so:
					- Update the budget considering the real time variation with respect to the last check
					- Update then the last check to the current closing price of the time frame
					'''
					budget += event[3]*(close[t]-order["current"])*1e5
					order["current"] = close[t]
					events.put(("OpenOrder",t+1,order,event[3]))
			else:
				'''
				SELL ORDER
				If in this new time frame the highest price overwhlemed the SL, it means that the order has been closed and a loss is taken with respect to the last check.
				Conversely, if the lowest price resulted smaller than the TP, it means that the order has been closed and a profit is taken with respect to the last check
				'''
				if high[t] >= order["SL"]:
					lots -= 2*step_lots
					budget -= event[3]*abs(order["SL"]-order["current"])*1e5
					busyMargin -= event[3]*1e5*order["start"]/leverage
					if verbose:
						print("\t"+RED+"LOST:"+END+" {:,.2f}".format(event[3]*abs(order["SL"]-order["start"])*1e5))
						
				elif low[t] <= order["TP"]:
					budget += event[3]*abs(order["TP"]-order["current"])*1e5
					busyMargin -= event[3]*1e5*order["start"]/leverage
					if verbose:
						print("\t"+GREEN+"GAINED:"+END+" {:,.2f}".format(event[3]*abs(order["TP"]-order["start"])*1e5))

				else:
					'''
					The order is still open so:
					- Update the budget considering the real time variation with respect to the last check
					- Update then the last check to the current closing price of the time frame
					'''
					budget += event[3]*(order["current"]-close[t])*1e5
					order["current"] = close[t]
					events.put(("OpenOrder",t+1,order,event[3]))
		if verbose:
			print("CURRENT BUDGET: {:,.2f}, BUSY MARGIN: {:,.2f}, MAXIMUM LOT: {:.2f}".format(budget, busyMargin, leverage*(budget-busyMargin)/1e5))
	
	print("STARTED WITH BUDGET: {:,.2f}, ENDED WITH BUDGET: {:,.2f}".format(initialBudget, budget))
	
	return history