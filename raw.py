import requests
import yfinance as yf
import pandas as pd
import math
import matplotlib.pyplot as plt

from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

api_key = '1cd34981158f47d2933af4113984b78a'

# ticker = str(input('Enter Abbreviation Here (SX, FX, Crypto):'))
ticker = 'MOD'
interval = '1month'

# mirror twelvedata with yf
yf_ticker = yf.Ticker(ticker)
yf_hist = yf_ticker.history(period='3y')
pd_yf_hist = pd.DataFrame(yf_hist)
yf_cvals = pd_yf_hist['Close']

# YYYY-MM-DD
start_date = '2020-01-01'
end_date = '2024-01-01'
api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date={start_date}&end_date={end_date}&interval={interval}&outputsize=12&apikey={api_key}'

data = requests.get(api_url).json()
pd_data = pd.DataFrame(data['values'])

# index 0 is most recent data
close_vals = pd_data['close']

# avg_return_func for column of values
def return_vals_func(values):
    array = []

    for i in range(len(values)-1):
        recent_val = float(values[i])
        later_val = float(values[i+1])
        chng = round((((recent_val - later_val)/later_val)*100),2)
        array.append(chng)
    
    return array[::-1]

# final return list values 'return_vals'
return_vals = return_vals_func(close_vals)

# calculate average return
avg_return = round((sum(return_vals) / len(return_vals)),2)
print(f'Ticker: {ticker}')
print(f'Average Return over period ({start_date} to {end_date}) for Time Interval {interval} : {avg_return}%')

def deviations_calc(values, baseline):
    array1 = []

    for i in range(len(values) - 1):
        deviation = round((baseline - values[i]),2)
        array1.append(deviation)

    return array1

deviation_vals = deviations_calc(return_vals, avg_return)

def sqrd(values):
    array2 = []

    for val in values:
        array2.append(val**2)

    return array2

squared_deviation_vals = sqrd(deviation_vals)

var = round((sum(squared_deviation_vals) / (len(squared_deviation_vals)-1)),2)
sdev = math.sqrt(var)
print(f'Market Volatility (Standard Deviation): {sdev}')

# GARCH Learning Model
np_yf_cvals = np.array([yf_cvals][0])

n = len(np_yf_cvals)
omega = 0.5
alpha_1 = 0.1
alpha_2 = 0.2
beta_1 = 0.3
beta_2 = 0.4

test_size = int(n*0.1)

# Assume volatilites 1 and 2 as = 1, 1. Because both sdev at index 0 and 1 constants = to 1. 
volatilities = [1,1]

for i in range(n):
    volatility_at_index = np.sqrt(omega + alpha_1*np_yf_cvals[-1]**2 + alpha_2*np_yf_cvals[-2]**2 + beta_1*volatilities[-1]**2 + beta_2*volatilities[-2]**2)
    volatilities.append(volatility_at_index)

## Print PACF Plot
#plot_pacf(np_yf_cvals**2)
#plt.show()

predictions = []
train, test = np_yf_cvals[:-test_size], np_yf_cvals[-test_size:]
# the following '_type' just means which type of GARCH is in use (ex. GARCH(1,1), GARCH(2,2))
model = arch_model(train, p=2, q=2)
model_fit = model.fit()
model_fit.summary()

predictions = model_fit.forecast(horizon=test_size*2)

rolling_predictions = []
for i in range(test_size):
    train = np_yf_cvals[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast()
    rolling_predictions.append((np.sqrt(pred.variance.values[-1,:][0]))+11.1)