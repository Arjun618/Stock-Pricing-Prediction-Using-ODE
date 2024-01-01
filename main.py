import vectorbt as vbt
import numpy as np
import math
from lppls import lppls
from lppls import lppls_cmaes
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# For Exponential model
start = '2013-01-01'
end = '2023-01-01'
price_data = vbt.YFData.download('AAPL', start=start, end=end).get('Close')
price_data.plot()
n = len(price_data)
mu = math.log(price_data.values[n-1] / price_data.values[0])
t = np.linspace(0, 1, n)
u = []
for i in range(n):
    u.append(price_data.values[0] * math.exp(mu * t[i]))

plt.plot(t, price_data.values, color="blue")
plt.plot(t, u, color="green")
plt.show()

# For LPPLS
date = price_data.index
price_log = np.log(price_data.values)
time = [pd.Timestamp.toordinal(datetime.datetime.strptime(str(t1), '%Y-%m-%d %H:%M:%S')) for t1 in date.astype('datetime64[ns]')]
observations = np.array([time, price_log])
MAX_SEARCHES = 25
lppls_model = lppls.LPPLS(observations=observations)
tc, m, w, a, b, c, c1, c2, o, d = lppls_model.fit(MAX_SEARCHES, minimizer='SLSQP')

def lppls_(t, tc, m, w, a, b, c1, c2):
    return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * np.log(tc - t))) + (c2 * np.sin(w * np.log(tc - t)))))

fit = lppls_(time, tc, m, w, a, b, c1, c2)
plt.plot(date, price_log, color='tab:blue')
plt.plot(date, fit, color='tab:orange')
plt.show()

# For GBM
mu = 0.1
n = 100
T = 1
M = 100
S0 = 100
sigma = 0.3
dt = T / n

St = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T)
St = np.vstack([np.ones(M), St])
St = S0 * St.cumprod(axis=0)

time = np.linspace(0, T, n+1)
tt = np.full(shape=(M, n+1), fill_value=time).T
plt.plot(tt, St)
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title("Realizations of Geometric Brownian Motion\n$dS_t = \mu S_t dt + \sigma S_t dW_t$\n$S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, mu, sigma))
plt.show()

# For final model
import pandas as pd
import numpy as np
import yfinance as yf

stock_name = 'AAPL'
start_date = '2013-01-01'
end_date = '2023-01-01'
pred_end_date = '2023-3-01'
scen_size = 1000

prices = yf.download(stock_name, start=start_date, end=pred_end_date)['Adj Close']
train_set = prices.loc[:end_date]
test_set = prices.loc[end_date:pred_end_date]

daily_returns = ((train_set / train_set.shift(1)) - 1)[1:]
So = train_set[-1]
dt = 1  # day # User input
n_of_wkdays = pd.date_range(start=pd.to_datetime(end_date, format="%Y-%m-%d") + pd.Timedelta('1 days'),
                            end=pd.to_datetime(pred_end_date, format="%Y-%m-%d")).to_series().map(lambda x: 1 if
                                                                                                    x.isoweekday() in range(1, 6) else 0).sum()
T = n_of_wkdays
N = T / dt
t = np.arange(1, int(N) + 1)
mu = np.mean(daily_returns)
sigma = np.std(daily_returns)

b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}
drift = (mu - 0.5 * sigma ** 2) * t
diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}

S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)])
S = np.hstack((np.array([[So] for scen in range(scen_size)]), S))  # add So to the beginning series

S_max = [S[:, i].max() for i in range(0, int(N))]
S_min = [S[:, i].min() for i in range(0, int(N))]
S_pred = 0.5 * np.array(S_max) + 0.5 * np.array(S_min)

final_df = pd.DataFrame(data=[test_set.reset_index()['Adj Close'], S_pred],
                        index=['real', 'pred']).T
final_df.index = test_set.index

mse = 1 / len(final_df) * np.sum((final_df['pred'] - final_df['real']) ** 2)

plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.suptitle('Monte-Carlo Simulation: ' + str(scen_size) + ' simulations', fontsize=20)
plt.title('Asset considered: {}'.format(stock_name))
plt.ylabel('USD Price')
plt.xlabel('Prediction Days')

for i in range(scen_size):
    plt.plot(pd.date_range(start=train_set.index[-1], end=pred_end_date, freq='D').map(lambda x: x if
                                                                                        x.isoweekday() in range(1, 6) else np.nan).dropna(),
             S[i, :])
plt.show()

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.suptitle('Predicted Price vs Real Price', fontsize=20)
plt.ylabel('USD Price')
plt.plot(final_df)
plt.legend(['Real Price', 'Predicted Price'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.show()
