
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt.mt_api_contract import ACCOUNT, PASSWORD, SERVER

# Symbol to download
SYMBOL = 'EURUSD'

# Output directory
OUTPUT_DIR = 'mt/csv_data'

# Date range: January 1, 2020 to January 27, 2026
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 1, 27, 23, 59, 59)

def connect():
    """Connect to MetaTrader 5"""
    if not mt5.initialize():
        print("initialize() failed")
        return False
        
    if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
        print("login() failed")
        mt5.shutdown()
        return False
        
    print(f"Connected to MT5 - Account: {ACCOUNT}, Server: {SERVER}")
    return True

def download_historical_data(symbol, start_date, end_date):
    """
    Download M1 historical data for a symbol from start_date to end_date
    Downloads data in yearly chunks with delays to allow server data loading
    """
    import time
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None
    
    print(f"\n📥 Downloading M1 data for {symbol}")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"   Timeframe: M1 (1 minute)")
    print(f"   ⏱️  Using delays to allow data loading from server...\n")
    
    # Force symbol synchronization - important for historical data
    print(f"   🔄 Synchronizing {symbol} data with server...")
    mt5.symbol_select(symbol, True)
    time.sleep(3)  # Give time for initial sync
    
    # Use M1 (1-minute) timeframe
    timeframe = mt5.TIMEFRAME_M1
    
    all_rates = []
    current_year = start_date.year
    end_year = end_date.year
    
    # Download data year by year with retry logic
    while current_year <= end_year:
        # Define chunk boundaries
        chunk_start = datetime(current_year, 1, 1) if current_year > start_date.year else start_date
        chunk_end = datetime(current_year, 12, 31, 23, 59, 59) if current_year < end_year else end_date
        
        print(f"   📅 {current_year}... ", end='', flush=True)
        
        # Try multiple times with increasing delays
        max_retries = 3
        rates = None
        
        for attempt in range(max_retries):
            # Get rates for this chunk
            rates = mt5.copy_rates_range(symbol, timeframe, chunk_start, chunk_end)
            
            if rates is not None and len(rates) > 0:
                # Success!
                break
            else:
                # Failed, wait before retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    print(f"⏳ (retry {attempt + 1}/{max_retries}, wait {wait_time}s)... ", end='', flush=True)
                    time.sleep(wait_time)
        
        if rates is not None and len(rates) > 0:
            all_rates.append(rates)
            print(f"✅ {len(rates):,} bars")
        else:
            error = mt5.last_error()
            print(f"❌ No data (Error: {error})")
        
        # Small delay between years to avoid overwhelming the server
        time.sleep(1)
        current_year += 1
    
    if not all_rates:
        print(f"\n❌ No M1 data retrieved for {symbol}")
        print(f"\n💡 Suggestions:")
        print(f"   - Vérifiez que le symbole {symbol} est bien disponible")
        print(f"   - Essayez de charger les données manuellement dans MT5 d'abord")
        print(f"   - Le serveur démo peut avoir des limitations sur les données M1")
        return None
    
    # Combine all chunks
    print(f"\n🔄 Combining {len(all_rates)} chunk(s)...")
    import numpy as np
    combined_rates = np.concatenate(all_rates)
    print(f"✅ Total bars retrieved: {len(combined_rates):,}")
    
    # Convert to DataFrame
    df = pd.DataFrame(combined_rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Format to match the structure
    df.rename(columns={
        'time': 'DateTime', 
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'tick_volume': 'Volume'
    }, inplace=True)
    
    df['Script'] = symbol
    df['WAP'] = df['Close']  # Weighted Average Price (approximated as Close)
    df['Count'] = df['Volume']
    
    # Reorder columns
    cols = ['Script', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'WAP', 'Count']
    df = df[cols]
    
    # Sort by date (oldest first)
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    return df

def save_data(df, symbol):
    """Save DataFrame to CSV file"""
    if df is None or df.empty:
        print("No data to save")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    # Save to CSV with date range in filename
    filename = os.path.join(OUTPUT_DIR, f"{symbol}_M1_2020_2026.csv")
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Data saved successfully!")
    print(f"   📁 File: {filename}")
    print(f"   📊 Rows: {len(df):,}")
    print(f"   📅 Date range: {df['DateTime'].min()} → {df['DateTime'].max()}")
    print(f"   💾 File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")

def main():
    """Main function"""
    print("="*70)
    print("EURUSD M1 Historical Data Downloader")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print("="*70)
    
    # Connect to MT5
    if not connect():
        return
    
    # Download M1 data for specified date range
    df = download_historical_data(SYMBOL, START_DATE, END_DATE)
    
    # Save data
    if df is not None:
        save_data(df, SYMBOL)
    
    # Disconnect
    mt5.shutdown()
    print("\n" + "="*70)
    print("Process completed!")
    print("="*70)

if __name__ == "__main__":
    main()
