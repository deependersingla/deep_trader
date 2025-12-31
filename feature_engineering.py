import pandas as pd
import numpy as np

def compute_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, slow=26, fast=12, signal=9):
    """Moving Average Convergence Divergence"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_atr(high, low, close, window=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def add_technical_indicators(df):
    """Adds RSI, MACD, BB, ATR to the dataframe"""
    df = df.copy()
    
    # Ensure Close is float
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    
    # RSI
    df['RSI'] = compute_rsi(close)
    
    # MACD
    df['MACD'], df['MACD_Signal'] = compute_macd(close)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(close)
    
    # ATR
    df['ATR'] = compute_atr(high, low, close)
    
    # Fill NaNs resulting from rolling windows
    df.bfill(inplace=True)
    df.fillna(0, inplace=True) # Fallback
    
    return df
