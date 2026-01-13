import time
import threading
import datetime
import pandas as pd
import numpy as np
import os
import sys
import joblib
from collections import deque

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

# Project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tensor-reinforcement'))
from feature_engineering import add_technical_indicators
from rainbow_agent import RainbowAgent

# Configuration
SYMBOL = "NIFTY50"
EXCHANGE = "NSE"
CURRENCY = "INR"
SEC_TYPE = "STK" # Check if NIFTY50 is index or stock on your feed
HOST = "127.0.0.1"
PORT = 7497 # 7497 paper, 7496 live
CLIENT_ID = 99

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'tensor-reinforcement', 'saved_networks_rainbow')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
# We need to pick a specific model checkpoint. 
# For auto-loading, we can pick the latest or a specific one.
# Let's assume we want to load 'rainbow_dqn_episode_5.h5' for now (or find latest).

class TradingApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        
        # State
        self.nextOrderId = None
        self.history = deque(maxlen=200) # Keep last 200 bars
        self.current_position = 0
        self.reqId_mapping = {} # reqId -> symbol
        
        # Load Model & Scaler
        self.agent = None
        self.scaler = None
        self._load_resources()

    def _load_resources(self):
        print("Loading AI Resources...")
        
        # Load Scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run training first.")
        self.scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded.")
        
        # Initialize Agent
        # We need to know state_dim and action_dim match training
        # Features: 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR'
        state_dim = 11 
        action_dim = 3
        self.agent = RainbowAgent(state_dim, action_dim)
        self.agent.epsilon = 0.0 # No exploration in live trading!
        
        # Find latest model
        # Simple logic: check for episode 5, 10, 15...
        # Ideally user points to a specific file.
        model_path = os.path.join(MODELS_DIR, 'rainbow_dqn_episode_20.h5')
        if not os.path.exists(model_path):
             # Fallback to episode 5 if 20 doesn't exist yet
             model_path = os.path.join(MODELS_DIR, 'rainbow_dqn_episode_5.h5')
             
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: No trained model found at {model_path}. Agent is untrained.")

    def nextValidId(self, orderId: int):
        self.nextOrderId = orderId
        print(f"Next Valid Order ID: {self.nextOrderId}")
        
    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} {errorCode} {errorString}")

    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        """
        Received a new 5-second bar (or whatever interval requested).
        We aggregate these or use them directly. 
        Note: realtimeBar is 5 seconds. Our model trained on 1-minute data??
        WARNING: If model trained on 1-min, we should aggregate 12 5-sec bars or request 1-min historical data updates.
        IB 'reqRealTimeBars' is strictly 5 seconds.
        Architecture decision: We will append 5-sec bars, but only TRADE every 12 bars (1 minute).
        OR better: use 5-second bars as the new "candle" if we want HFT.
        
        BUT our data pipeline features (RSI 14) implies 14 *time units*.
        If we trained on 1-min candles, RSI-14 covers 14 minutes.
        If we feed 5-sec candles, RSI-14 covers 70 seconds.
        This IS A MISMATCH.
        
        Fix: We must resample or use 'reqHistoricalData' with keepUpToDate=True to get 1-min bars.
        Let's use reqHistoricalData with keepUpToDate=True.
        """
        # This callback is for reqRealTimeBars. 
        # We will use historicalDataUpdate instead.
        pass

    def historicalData(self, reqId, bar):
        # Initial history download
        self.history.append({
            'Open': bar.open, 
            'High': bar.high, 
            'Low': bar.low, 
            'Close': bar.close, 
            'Volume': bar.volume
        })
        
    def historicalDataUpdate(self, reqId, bar):
        """
        Called when a new bar is completed (if keepUpToDate=True).
        """
        print(f"New Bar: {bar.date} C:{bar.close}")
        
        # Append to history
        self.history.append({
            'Open': bar.open, 
            'High': bar.high, 
            'Low': bar.low, 
            'Close': bar.close, 
            'Volume': bar.volume
        })
        
        self.on_new_candle()

    def on_new_candle(self):
        if len(self.history) < 60:
            print("Waiting for more history...")
            return

        # 1. Prepare Data
        df = pd.DataFrame(self.history)
        
        # 2. Add Indicators
        df = add_technical_indicators(df)
        
        # 3. Extract Features (Last Row)
        # Features must match training: ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']
        
        last_row = df.iloc[-1][features].values.reshape(1, -1)
        
        # 4. Scale
        state = self.scaler.transform(last_row)
        
        # 5. Predict
        # Flatten state for agent (1, 11) -> (11,)
        action = self.agent.act(state[0])
        
        self.execute_trade(action)

    def execute_trade(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        
        action_str = ["HOLD", "BUY", "SELL"][action]
        print(f"AI Decision: {action_str}")
        
        if action == 0:
            return
            
        if action == 1 and self.current_position <= 0:
            print("Placing BUY Order...")
            self.place_order("BUY")
            self.current_position = 1
            
        elif action == 2 and self.current_position >= 0:
            print("Placing SELL Order...")
            self.place_order("SELL")
            self.current_position = -1

    def place_order(self, direction):
        if self.nextOrderId is None:
            print("Error: NextOrderId not ready")
            return
            
        contract = Contract()
        contract.symbol = SYMBOL
        contract.secType = SEC_TYPE
        contract.exchange = EXCHANGE
        contract.currency = CURRENCY
        
        order = Order()
        order.action = direction
        order.orderType = "MKT"
        order.totalQuantity = 1 # Simple 1 lot
        
        print(f"Sending {direction} order for {SYMBOL}...")
        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1

def run_loop(app):
    app.run()

def main():
    app = TradingApp()
    app.connect(HOST, PORT, CLIENT_ID)
    
    # Start thread
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()
    
    time.sleep(1)
    
    if not app.isConnected():
        print("Failed to connect to IBKR. Check Gateway/TWS.")
        return

    # Request Data
    contract = Contract()
    contract.symbol = SYMBOL
    contract.secType = SEC_TYPE
    contract.exchange = EXCHANGE
    contract.currency = CURRENCY
    
    print("Requesting Live 1-min Bars...")
    
    # Req ID 1
    # keepUpToDate=True allows streaming of pending bars
    app.reqHistoricalData(1, contract, "", "2 D", "1 min", "TRADES", 1, 1, True, [])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        app.disconnect()

if __name__ == "__main__":
    main()
