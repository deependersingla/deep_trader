import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam

def create_lstm_model(input_shape, n_out=3):
    """
    Creates an LSTM model for price prediction.
    
    Args:
        input_shape: (timesteps, features) tuple.
        n_out: Number of output classes (Buy, Sell, Hold).
    """
    model = Sequential()
    
    # Layer 1: LSTM with return sequences
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Layer 2: LSTM without return sequences
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Layer 3: Dense
    model.add(Dense(32, activation='relu'))
    
    # Output Layer
    model.add(Dense(n_out, activation='softmax'))
    
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=Adam(learning_rate=0.001), 
        metrics=['accuracy']
    )
    
    return model
