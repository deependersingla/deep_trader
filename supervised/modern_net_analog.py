import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_analog_mlp_model(n_in=61, n_units=500, n_out=1):
    """
    Keras MLP for Regression (Analog) mode.
    Output is linear (predicting Gain).
    """
    model = Sequential()
    model.add(Input(shape=(n_in,)))
    model.add(Dense(n_units, activation='relu'))
    model.add(Dense(n_units, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Linear output for regression
    model.add(Dense(n_out, activation='linear'))
    
    model.compile(optimizer='adam', 
                  loss='mse', 
                  metrics=['mae'])
    return model
