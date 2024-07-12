from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import backtrader as bt
from alpaca_trade_api.rest import REST, TimeFrame
import time
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

# Data Gathering
def fetch_market_data(symbol, start_date, end_date):
    data = bt.feeds.YahooFinanceData(dataname=symbol, fromdate=start_date, todate=end_date)
    return data.getdata()

# Technical Indicators
def calculate_sma(data, period):
    return data.close.rolling(period).mean()

def calculate_rsi(data, period=14):
    delta = data.close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up = up.rolling(period).mean()
    roll_down = down.abs().rolling(period).mean()
    rsi = 100.0 - (100.0 / (1.0 + (roll_up / roll_down)))
    return rsi

# Machine Learning Models
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Backtesting
class TradingStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
    )

    def __init__(self):
        self.sma = calculate_sma(self.data, self.params.sma_period)
        self.rsi = calculate_rsi(self.data, self.params.rsi_period)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0] and self.rsi[0] < 30:
                self.buy()
        else:
            if self.data.close[0] < self.sma[0] or self.rsi[0] > 70:
                self.sell()

def backtest_strategy(symbol, start_date, end_date):
    cerebro = bt.Cerebro()
    data = fetch_market_data(symbol, start_date, end_date)
    cerebro.adddata(data)
    cerebro.addstrategy(TradingStrategy)
    cerebro.run()
    return cerebro.broker.getvalue()

# Live Trading
def execute_trade(symbol, side, quantity):
    api = REST(
        key_id='your_alpaca_api_key',
        secret_key='your_alpaca_secret_key',
        base_url='https://paper-api.alpaca.markets'
    )

    if side == 'buy':
        api.submit_order(symbol=symbol, qty=quantity, side='buy', type='market', time_in_force='day')
    elif side == 'sell':
        api.submit_order(symbol=symbol, qty=quantity, side='sell', type='market', time_in_force='day')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    result = backtest_strategy(symbol, start_date, end_date)
    return jsonify({'result': result})

@app.route('/live_trade', methods=['POST'])
def live_trade():
    symbol = request.form['symbol']
    side = request.form['side']
    quantity = int(request.form['quantity'])
    execute_trade(symbol, side, quantity)
    return jsonify({'message': 'Trade executed successfully'})

if __name__ == '__main__':
    app.run(debug=True)