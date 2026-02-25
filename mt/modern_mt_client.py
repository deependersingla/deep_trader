
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

# Credentials
ACCOUNT = 101170533
PASSWORD = "Ax*gMrH8"
SERVER = "MetaQuotes-Demo"

def connect():
    """
    Initializes connection to MetaTrader 5.
    Returns True if successful, False otherwise.
    """
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    # Check if already connected to the correct account
    # Note: mt5.login() is needed if not already logged in or to switch accounts
    authorized = mt5.login(ACCOUNT, password=PASSWORD, server=SERVER)
    if authorized:
        print(f"Connected to account #{ACCOUNT}")
    else:
        print("failed to connect at account #{}, error code: {}".format(ACCOUNT, mt5.last_error()))
        
    return authorized

def disconnect():
    mt5.shutdown()

def download_historical_data(symbol, duration="2 D", bar_size="1 min", what_to_show="TRADES"):
    """
    Downloads historical data from MT5 and returns a pandas DataFrame 
    compatible with the format: ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'WAP', 'Count']
    
    :param symbol: Symbol name (e.g., "EURUSD")
    :param duration: Duration string (e.g., "2 D"). Parsed to determine number of bars or time range.
    :param bar_size: Bar size (e.g., "1 min"). Mapped to MT5 timeframes.
    :param what_to_show: ignored in MT5 context, kept for compatibility.
    :return: DataFrame or empty DataFrame if failed.
    """
    
    if not connect():
        return pd.DataFrame()
    
    # Map bar_size to MT5 timeframes
    timeframe_map = {
        "1 min": mt5.TIMEFRAME_M1,
        "5 mins": mt5.TIMEFRAME_M5,
        "15 mins": mt5.TIMEFRAME_M15,
        "30 mins": mt5.TIMEFRAME_M30,
        "1 hour": mt5.TIMEFRAME_H1,
        "4 hours": mt5.TIMEFRAME_H4,
        "1 day": mt5.TIMEFRAME_D1,
    }
    
    tf = timeframe_map.get(bar_size, mt5.TIMEFRAME_M1)
    
    # Parse duration to limit rows roughly (MT5 counts bars usually)
    # Simple heuristic: "2 D" -> 2 days. 
    # For minute data: 2 * 24 * 60 = 2880 bars approx.
    num_bars = 1000 # Default
    if "D" in duration:
        try:
            days = int(duration.split()[0])
            if "min" in bar_size:
                minutes = 1 if bar_size == "1 min" else int(bar_size.split()[0])
                num_bars = (days * 24 * 60) // minutes
            elif "hour" in bar_size:
                 num_bars = days * 24
            else:
                 num_bars = days
        except:
             pass
    
    # Safety margin
    num_bars = int(num_bars * 1.5)
    
    # Download data
    # copy_rates_from_pos(symbol, timeframe, start_pos, count)
    print(f"Requesting data for {symbol} (approx {num_bars} bars)...")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_bars)
    
    disconnect()
    
    if rates is None:
        print(f"No data for {symbol} (Check if symbol exists in Market Watch)")
        return pd.DataFrame()
    
    if len(rates) == 0:
        print(f"Received empty data for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time in seconds to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Rename columns to match IB format
    # MT5 cols: time, open, high, low, close, tick_volume, spread, real_volume
    # Target: DateTime, Open, High, Low, Close, Volume, WAP, Count
    
    df.rename(columns={
        'time': 'DateTime', 
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'tick_volume': 'Volume' # Using tick_volume as proxy for Volume in FX/CFD
    }, inplace=True)
    
    # Add missing columns
    df['WAP'] = df['Close'] # Approximation or 0
    df['Count'] = df['Volume'] # Approximation
    
    # Select and reorder columns
    final_cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'WAP', 'Count']
    df = df[final_cols]
    
    return df

if __name__ == "__main__":
    # Example
    sym = "EURUSD"
    print(f"Testing download for {sym}...")
    df = download_historical_data(sym, duration="1 D", bar_size="1 min")
    if not df.empty:
        print(df.head())
        print(df.tail())
    else:
        print("DataFrame is empty.")
