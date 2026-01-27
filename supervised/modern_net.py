import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_mlp_model(n_in=61, n_units=500, n_out=3):
    """
    Recreates the original Chainer MLP architecture in Keras.
    Structure:
    Input(61) -> Dense(500, ReLU) -> Dense(500, ReLU) -> Dense(200, ReLU)
              -> Dense(100, ReLU) -> Dense(50, ReLU) -> Dense(10, ReLU) -> Dense(3, Softmax)
    """
    model = Sequential()
    model.add(Input(shape=(n_in,)))
    model.add(Dense(n_units, activation='relu'))
    model.add(Dense(n_units, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_out, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
