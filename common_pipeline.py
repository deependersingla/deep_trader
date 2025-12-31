import pandas as pd
import numpy as np
import os
from feature_engineering import add_technical_indicators

def load_and_process_data(file_path, target_threshold=0.0005):
    """
    Loads data, adds indicators, and generates balanced targets.
    
    Args:
        file_path: Path to CSV.
        target_threshold: Minimum return to classify as Buy/Sell (~0.05% default).
        
    Returns:
        df: Processed DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
        
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Basic Cleaning
    cols = ['Open', 'High', 'Low', 'Close', 'Volume Traded']
    if 'Volume Traded' in df.columns:
        df['Volume'] = df['Volume Traded']
    else:
        # Generate dummy volume if missing (rare case)
        if 'Volume' not in df.columns:
            df['Volume'] = 1.0 
            
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 2. Feature Engineering
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. Target Generation (Logic Fix)
    # We want to predict the NEXT candle's direction relative to CURRENT close
    # Or maybe trend over next N candles? Let's keep it simple: Next Close vs Current Close
    
    # Return = (Next_Close - Current_Close) / Current_Close
    # Return = (Next_Close - Current_Close) / Current_Close
    # We want (Close[t+1] - Close[t]) / Close[t]
    
    df['Next_Close'] = df['Close'].shift(-1)
    df['Future_Return'] = (df['Next_Close'] - df['Close']) / df['Close']
    
    # Classify
    # 0: Hold (Return within -threshold and +threshold)
    # 1: Buy  (Return > threshold)
    # 2: Sell (Return < -threshold)
    
    conditions = [
        (df['Future_Return'] > target_threshold),  # Buy
        (df['Future_Return'] < -target_threshold) # Sell
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    # Drop the last row which has NaN target
    df.dropna(inplace=True)
    
    print("Data processing complete.")
    return df

def get_feature_columns():
    return ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR']

if __name__ == "__main__":
    # Test Block
    path = os.path.join(os.path.dirname(__file__), 'data/sorted_data.csv')
    try:
        df = load_and_process_data(path)
        print(f"Shape: {df.shape}")
        
        unique, counts = np.unique(df['Target'], return_counts=True)
        dist = dict(zip(unique, counts))
        total = sum(counts)
        print("Class Distribution:")
        for k, v in dist.items():
            print(f"Class {k}: {v} ({v/total*100:.2f}%)")
            
    except Exception as e:
        print(e)
