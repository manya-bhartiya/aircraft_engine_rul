from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm(input_shape):
    model = Sequential([
        LSTM(100, activation='tanh', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model