import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import scale

class StockDataLoader:
    def __init__(self, file_path, sequence_length=10):
        """
        Args:
            file_path: Path to the CSV file.
            sequence_length: Length of history to include in the state (if we were doing sequence input, 
                             but here we follow the original logic of specific moving averages).
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.data = None
        self.state_columns = []
    
    def load_csv(self):
        """
        Loads the CSV and parses dates.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        # Reading sorted_data.csv format: Index, DateTime, Open, Close, Low, High, Volume Traded
        # Note: The file seems to have "Nifty" as a column in some rows based on the old script?
        # Let's inspect the first few lines of real data if possible, but assuming standard CSV for now.
        
        # We'll use the column names from the header.
        self.data = pd.read_csv(self.file_path)
        
        # Ensure we have numerical columns
        cols_to_use = ['Open', 'Close', 'Low', 'High']
        if 'Volume Traded' in self.data.columns:
            cols_to_use.append('Volume Traded')
            
        for col in cols_to_use:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        self.data.dropna(subset=cols_to_use, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(self.data)} rows from {self.file_path}")

    def preprocess(self):
        """
        Generates features similar to the original:
        - Normalized values
        - Rolling averages
        """
        # We focus on 'Close' price for the "price" action, but the original used multiple columns.
        # Original: temp = [data[2], data[3], data[4], data[5],data[8]] -> Open, High, Low, Close, Volume?
        # Let's create a feature vector from these.
        
        feature_cols = ['Open', 'High', 'Low', 'Close']
        if 'Volume Traded' in self.data.columns:
            feature_cols.append('Volume Traded')
            
        df_features = self.data[feature_cols].copy()

        # The original code did complex normalization: accumulate data, find mean, divide by mean.
        # We will use a more standard approach: Rolling Window Normalization or Percentage Change.
        # However, to keep spirit of "trends", we'll calculate rolling averages relative to current price.
        
        # Compute Moving Averages
        # Assuming 1 minute intervals (based on "last_one_hour_average = -60")
        df_features['MA_60'] = df_features['Close'].rolling(window=60).mean()
        df_features['MA_300'] = df_features['Close'].rolling(window=300).mean() # 5 hours / 1 day roughly?
        df_features['MA_900'] = df_features['Close'].rolling(window=900).mean() # 3 days?
        
        # Ratios (Normalization invariant to absolute price level)
        df_features['Ratio_60'] = df_features['Close'] / df_features['MA_60']
        df_features['Ratio_300'] = df_features['Close'] / df_features['MA_300']
        df_features['Ratio_900'] = df_features['Close'] / df_features['MA_900']
        
        # Add simpler returns
        df_features['Returns'] = df_features['Close'].pct_change()
        
        df_features.dropna(inplace=True)
        
        # Final State Vector Columns
        self.state_data = df_features[['Ratio_60', 'Ratio_300', 'Ratio_900', 'Returns']].values
        
        # We also need the raw prices for the simulation (calculating portfolio value)
        self.price_data = df_features['Close'].values
        
        # Scale the state data for better NN performance
        # Handle constant columns to avoid NaNs
        X = df_features[['Ratio_60', 'Ratio_300', 'Ratio_900', 'Returns']].values
        
        # Check for NaNs or Inf introduced by logic
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaNs/Infs found in features. Replacing with 0.")
            X = np.nan_to_num(X)

        self.state_data = scale(X)
        if np.any(np.isnan(self.state_data)):
             print("Warning: NaNs generated during scaling (constant columns?). Fixing.")
             self.state_data = np.nan_to_num(self.state_data)
        
        print(f"Preprocessing complete. State shape: {self.state_data.shape}")
        
    def get_data(self):
        return self.state_data, self.price_data

if __name__ == "__main__":
    # Quick test
    loader = StockDataLoader("../data/sorted_data.csv")
    try:
        loader.load_csv()
        loader.preprocess()
    except Exception as e:
        print(f"Error during test: {e}")
