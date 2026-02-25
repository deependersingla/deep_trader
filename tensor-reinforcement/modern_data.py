import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import scale

class StockDataLoader:
    def __init__(self, file_path, sequence_length=10):
        """
        Args:
            file_path: Path to the CSV file.
            sequence_length: Length of history to include in the state.
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.data = None
        self.state_data = None
        self.price_data = None
        self.dates = None
    
    def load_csv(self):
        """
        Loads the CSV and parses dates.
        Expected columns: timestamp, open, high, low, close, volume (lowercase)
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        print(f"Loading data from {self.file_path}...")
        self.data = pd.read_csv(self.file_path)
        
        # Standardize column names to lowercase just in case
        self.data.columns = [c.lower() for c in self.data.columns]
        
        # Parse timestamp
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.sort_values('timestamp', inplace=True)
        elif 'datetime' in self.data.columns:
            # Fallback for old format if needed, but primary target is new file
            self.data['timestamp'] = pd.to_datetime(self.data['datetime'])
            self.data.sort_values('timestamp', inplace=True)
            
        # Ensure we have numerical columns
        cols_to_use = ['open', 'close', 'low', 'high']
        if 'volume' in self.data.columns:
            cols_to_use.append('volume')
            
        for col in cols_to_use:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        self.data.dropna(subset=cols_to_use, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(self.data)} rows. Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) # Neutral fill

    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def preprocess(self):
        """
        Generates features:
        - Rolling averages relative to price
        - Returns
        - RSI
        - MACD
        """
        feature_cols = ['open', 'high', 'low', 'close']
        if 'volume' in self.data.columns:
            feature_cols.append('volume')
            
        df_features = self.data.copy()

        # Compute Moving Averages
        df_features['MA_60'] = df_features['close'].rolling(window=60).mean()
        df_features['MA_300'] = df_features['close'].rolling(window=300).mean()
        df_features['MA_900'] = df_features['close'].rolling(window=900).mean()
        
        # Ratios
        df_features['Ratio_60'] = df_features['close'] / df_features['MA_60']
        df_features['Ratio_300'] = df_features['close'] / df_features['MA_300']
        df_features['Ratio_900'] = df_features['close'] / df_features['MA_900']
        
        # Returns
        df_features['Returns'] = df_features['close'].pct_change()

        # Indicators
        df_features['RSI'] = self.calculate_rsi(df_features['close'])
        macd, signal = self.calculate_macd(df_features['close'])
        df_features['MACD'] = macd
        df_features['Signal'] = signal
        
        df_features.dropna(inplace=True)
        
        # Verify alignment
        self.dates = df_features['timestamp'].values
        self.price_data = df_features['close'].values
        
        # Final State Vector Columns
        # Adding RSI, MACD, Signal to features
        X = df_features[['Ratio_60', 'Ratio_300', 'Ratio_900', 'Returns', 'RSI', 'MACD', 'Signal']].values
        
        # Check for NaNs or Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaNs/Infs found in features. Replacing with 0.")
            X = np.nan_to_num(X)

        self.state_data = scale(X)
        if np.any(np.isnan(self.state_data)):
             print("Warning: NaNs generated during scaling. Fixing.")
             self.state_data = np.nan_to_num(self.state_data)
        
        print(f"Preprocessing complete. State shape: {self.state_data.shape}")
        
    def get_data(self):
        """
        Returns processed data and associated dates/prices.
        """
        return self.state_data, self.price_data, self.dates

if __name__ == "__main__":
    # Quick test
    loader = StockDataLoader("../data/EURUSD_M1_2012_2025.csv")
    try:
        loader.load_csv()
        loader.preprocess()
    except Exception as e:
        print(f"Error during test: {e}")
