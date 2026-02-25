
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys
import joblib
from datetime import datetime

# Add project root to path for imports
# Current file is in <root>/mt/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'tensor-reinforcement'))

from feature_engineering import add_technical_indicators
from rainbow_agent import RainbowAgent
from mt.modern_mt_client import connect, disconnect, ACCOUNT, PASSWORD, SERVER

# Configuration
SYMBOL = "EURUSD" # Default logic uses EURUSD for demo availability
TIMEFRAME = mt5.TIMEFRAME_M1
VOLUME = 0.01 # Minimum lot size
MODELS_DIR = os.path.join(PROJECT_ROOT, 'tensor-reinforcement', 'saved_networks_rainbow')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

class MT5Trader:
    def __init__(self):
        self.scaler = None
        self.agent = None
        self.last_bar_time = None
        self.current_position = 0 # Track roughly
        self._load_resources()
        
    def _load_resources(self):
        print("Loading AI Resources...")
        if not os.path.exists(SCALER_PATH):
             # Try absolute path fallback if running from weird cwd
             pass
             
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded.")
        except Exception as e:
            print(f"Error loading scaler from {SCALER_PATH}: {e}")
            raise
        
        state_dim = 11
        action_dim = 3
        self.agent = RainbowAgent(state_dim, action_dim)
        self.agent.epsilon = 0.0 # No exploration in live
        
        # Load model (same logic as IB)
        model_path = os.path.join(MODELS_DIR, 'rainbow_dqn_episode_20.h5')
        if not os.path.exists(model_path):
             model_path = os.path.join(MODELS_DIR, 'rainbow_dqn_episode_5.h5')
        
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: No model found at {model_path}! Agent is running untrained.")

    def get_latest_data(self):
        # We need historical data for indicators (RSI requires e.g. 14 bars)
        # Fetching 200 bars to be safe
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 200)
        if rates is None or len(rates) == 0:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'time': 'DateTime', 'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        return df

    def execute_trade(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        action_str = ["HOLD", "BUY", "SELL"][action]
        print(f"AI Decision: {action_str}")
        
        if action == 0:
            return
            
        # Check current positions to avoid stacking if not desired
        positions = mt5.positions_get(symbol=SYMBOL)
        net_position = 0
        if positions:
            for pos in positions:
                if pos.type == mt5.ORDER_TYPE_BUY:
                    net_position += pos.volume
                elif pos.type == mt5.ORDER_TYPE_SELL:
                    net_position -= pos.volume
        
        print(f"Current Net Position: {net_position:.2f}")
        
        # Strategy:
        # If BUY signal (1):
        #   - We want to be Long.
        #   - If Short, Close Short (Buy).
        #   - If Flat, Open Long (Buy).
        #   - If Long, Hold? Or Buy more? Let's just Buy 0.01 more for now as per simple logic.
        
        order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).ask if action == 1 else mt5.symbol_info_tick(SYMBOL).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": VOLUME,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "DeepTrader AI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"Sending {action_str} order...")
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order Failed: {result.retcode} - {result.comment}")
        else:
            print(f"Order Executed: {result.comment}")

    def run(self):
        if not connect():
            return
            
        print(f"Starting Live Trader on {SYMBOL}...")
        
        # Initial check
        df = self.get_latest_data()
        if df is not None:
            self.last_bar_time = df.iloc[-2]['DateTime'] # Mark last closed bar
            print(f"Last closed bar: {self.last_bar_time}")
            
        try:
            while True:
                # Polling loop
                time.sleep(1) # Check every second
                
                df = self.get_latest_data()
                if df is None:
                    continue
                    
                # Check if new bar (candle closed)
                # DataFrame last row (-1) is currently developing bar
                # Row (-2) is the last completed bar
                completed_bar = df.iloc[-2]
                completed_time = completed_bar['DateTime']
                
                if self.last_bar_time != completed_time:
                    print(f"New Bar Completed: {completed_time}")
                    self.last_bar_time = completed_time
                    
                    # Calculate Indicators
                    df_tech = add_technical_indicators(df)
                    
                    # Features from the completed bar (iloc[-2])
                    # Note: We use the already computed technical indicators which might use the 'current' bar
                    # but usually for decision making on CLOSE, we look at the closed bar properties.
                    
                    row = df_tech.iloc[-2]
                    
                    # Features list from feature_engineering/training must match exactly
                    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']
                    
                    # Ensure values are float
                    vector = row[features].values.astype(float).reshape(1, -1)
                    
                    # Scale
                    state = self.scaler.transform(vector)
                    
                    # Action
                    action = self.agent.act(state[0])
                    self.execute_trade(action)
                        
        except KeyboardInterrupt:
            print("Stopping...")
            disconnect()
        except Exception as e:
            print(f"Critical Error: {e}")
            disconnect()

if __name__ == "__main__":
    bot = MT5Trader()
    bot.run()
