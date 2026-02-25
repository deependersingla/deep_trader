
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import time
from mt.mt_api_contract import ACCOUNT, PASSWORD, SERVER

# List of symbols to download
# Replacing NIFTY stocks with Forex/Indices available on MetaQuotes-Demo
SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
    'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
]

OUTPUT_DIR = 'mt/csv_data'

def connect():
    if not mt5.initialize():
        print("initialize() failed")
        return False
        
    if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
        print("login() failed")
        mt5.shutdown()
        return False
        
    return True

def download_data(symbol):
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return

    # Download 2600 days of data approx (replicating original logic)
    # Original used loop for large history. MT5 copy_rates_from is easier.
    # Let's get 50000 M1 bars
    count = 50000 
    timeframe = mt5.TIMEFRAME_M1
    
    print(f"Downloading {count} bars for {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Format to match original CSV structure:
    # Script, DateTime, Open, High, Low, Close, Volume, WAP, Count
    df.rename(columns={
        'time': 'DateTime', 'open': 'Open', 'high': 'High', 
        'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)
    
    df['Script'] = symbol
    df['WAP'] = df['Close']
    df['Count'] = df['Volume']
    
    # Reorder
    cols = ['Script', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'WAP', 'Count']
    df = df[cols]
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    filename = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

def main():
    if not connect():
        return

    for sym in SYMBOLS:
        download_data(sym)
        time.sleep(1) # Gentle on the API

    mt5.shutdown()
    print("All done")

if __name__ == "__main__":
    main()
