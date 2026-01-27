import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modern_lstm import create_lstm_model
import os
import sys

# Add project root to sys.path to find common_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_pipeline import load_and_process_data, get_feature_columns

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sorted_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def process_data_lstm_pipeline(window_size=60):
    # Use the common pipeline
    # We can tune threshold here if needed, but default 0.0005 gave 9% buy/sell
    df = load_and_process_data(DATA_PATH, target_threshold=0.0005)
    
    # Feature Selection
    features = get_feature_columns() # Open, High, Low, Close, Volume, RSI, MACD...
    print(f"Features: {features}")
    
    data_values = df[features].values
    targets = df['Target'].values
    
    X = []
    y = []
    
    # Optimization
    start_idx = max(window_size, len(df) - 50000)
    print(f"Processing last {len(df) - start_idx} steps for LSTM training...")
    
    for i in range(start_idx, len(df)-1):
        window = data_values[i-window_size : i]
        
        # Local Normalization
        means = np.mean(window, axis=0)
        stds = np.std(window, axis=0) + 1e-8
        std_window = (window - means) / stds
        
        # Scaler
        mm_min = np.min(std_window, axis=0)
        mm_max = np.max(std_window, axis=0) + 1e-8
        scaled_window = (std_window - mm_min) / (mm_max - mm_min)
        
        X.append(scaled_window)
        y.append(targets[i])
        
        if i % 10000 == 0:
            print(f"Processed {i}...")

    return np.array(X), np.array(y), len(features)

def main():
    WINDOW_SIZE = 60
    
    try:
        X, y, n_features = process_data_lstm_pipeline(window_size=WINDOW_SIZE)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
        
    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    
    if len(X) == 0:
        print("No data generated.")
        return

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    # Input shape: (timesteps, features) = (60, n_features)
    model = create_lstm_model(input_shape=(WINDOW_SIZE, n_features), n_out=3)
    
    # Train
    print("Starting Training...")
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=5, 
              batch_size=64)
    
    # Save
    save_path = os.path.join(MODELS_DIR, 'modern_lstm_v2.h5')
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
