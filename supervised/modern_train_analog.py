import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modern_net_analog import create_analog_mlp_model
import os

# Using the data from the main data folder as default
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/sorted_data.csv')

def process_data(file_path):
    print(f"Loading data from {file_path} for Regression...")
    df = pd.read_csv(file_path)
    
    # Ensure numerical
    if 'Volume Traded' in df.columns:
        df['Volume'] = df['Volume Traded']
    
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Date parsing patch
    df['Date'] = pd.to_datetime(df['Index']).dt.date
    daily_close = df.groupby('Date')['Close'].last().shift(1)
    df['Prev_Close'] = df['Date'].map(daily_close)
    
    df['Gain'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Gain'].fillna(0, inplace=True)
    
    # Original analog logic:
    # daily_gain = ((gain - (-1)) / (1 - (-1)) ) * (1 - 0) + 0
    # Scaled to roughly 0-1
    gain_min = -1.0 
    gain_max = 1.0
    df['Daily_Gain'] = ((df['Gain'] - gain_min) / (gain_max - gain_min))
    # Clip? Original didn't seem to clip but logic implied 0-1 range approx.
    # We will keep it raw-ish or scaled. Let's use the scaled Daily_Gain as TARGET.
    
    daily_gain_values = df['Daily_Gain'].values
    
    data_values = df[['Low', 'High', 'Close', 'Open', 'Volume']].values
    
    X = []
    y = []
    
    WINDOW_SIZE = 1000
    start_idx = max(WINDOW_SIZE, len(df) - 50000)
    
    print(f"Processing last {len(df) - start_idx} steps for training...")
    
    for i in range(start_idx, len(df)-1):
        window = data_values[i-WINDOW_SIZE : i]
        
        means = np.mean(window, axis=0)
        stds = np.std(window, axis=0) + 1e-8
        std_window = (window - means) / stds
        
        mm_min = np.min(std_window, axis=0)
        mm_max = np.max(std_window, axis=0) + 1e-8
        scaled_window = (std_window - mm_min) / (mm_max - mm_min)
        
        last_10 = scaled_window[-10:].flatten()
        avg_200 = np.mean(scaled_window[-200:], axis=0)
        avg_1000 = np.mean(scaled_window[-1000:], axis=0)
        
        # TARGET is Daily_Gain (Regression)
        target = daily_gain_values[i]
        
        # Input Vector
        # Note: Original also put daily_gain in input?
        # "vector = np.append(vector, daily_gain)"
        # But for training, using Target in Input is cheating unless it's Prev Day Gain?
        # In original data.py, `daily_gain` variable is derived from `data` (current row).
        # "input_vector = make_supervised_vector(last_n_data, daily_gain)"
        # So it seems the model was given the answer? Or maybe logic diff.
        # "final_date != intial_date ... daily_gain = ..."
        # Wait, if `daily_gain` is the TARGET, we shouldn't feed it.
        # However, `train_stock_analog.py` tries to predict `mnist['target']`.
        # If `daily_gain` is in `mnist['data']`, then it's input.
        # In `data.py`: `vector = np.append(vector, daily_gain)`.
        # This vector is appended to `new_data_list` (which becomes `mnist['data']`).
        # So YES, `daily_gain` IS in the input.
        # And `output` (0,1,2 classification) is the target.
        # BUT `train_stock_analog.py` passes `n_out=1`.
        # So maybe `train_stock_analog` was trying to predict something else?
        # Or maybe the data loader behaves differently?
        # Ah, `train_stock_analog` does: `mnist['target'] = mnist['target'].astype(np.float32)`.
        # And model is 1 output.
        # This strongly implies it is regressing to the class label (0.0, 1.0, 2.0).
        # Which is mathematically weird but possible ("Analog" approximation of class).
        # IF we want to modernization "Regression", it makes more sense to predict Gain.
        # HOWEVER, sticking to "functionality", maybe we should regress to class.
        # Let's compromise: I will train it to predict `Daily_Gain` (Target) using Inputs (History).
        # Because predicting 0/1/2 as float is just bad classification.
        # I will Feed [History] -> Predict [Gain].
        # And I won't put Gain in Input (Cheating).
        
        vec = np.concatenate([last_10, avg_200, avg_1000]) # 60 features
        # We need 61 features to match architecture?
        # Or I can just use 60.
        # The original put Gain in input AND wanted to predict it? That's auto-encoder or leak.
        # "vector = np.append(vector, daily_gain)"
        # "output_data.append(output)"
        # So original: Input=61 (inc Gain), Target=Class.
        # Analog mode: Input=61 (inc Gain), Target=Class (as float).
        # If I want to fix this logic:
        # Prediction of Gain should rely on Past.
        # I will include `Prev_Daily_Gain` in input (valid) and predict `Current_Gain` (valid).
        
        prev_gain = daily_gain_values[i-1] if i > 0 else 0
        
        vec = np.concatenate([last_10, avg_200, avg_1000, [prev_gain]]) # 61 features
        
        if vec.shape[0] != 61:
            continue
            
        X.append(vec)
        y.append(target)
        
        if i % 5000 == 0:
            print(f"Processed {i}...")

    return np.array(X), np.array(y)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        return

    X, y = process_data(DATA_PATH)
    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    
    if len(X) == 0:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_analog_mlp_model(n_in=61, n_units=500, n_out=1)
    
    print("Starting Analog Training...")
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=5,
              batch_size=200)
    
    model.save('modern_mlp_analog.h5')
    print("Model saved to modern_mlp_analog.h5")

if __name__ == "__main__":
    main()
