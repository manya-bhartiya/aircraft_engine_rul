from src.preprocess import load_and_preprocess, create_sequences
from src.model import build_lstm
import numpy as np

# Load dataset
df, features, scaler = load_and_preprocess('data/engine_data.csv')

# Create sequences
X, y = create_sequences(df, features, window=30)

# Split train-test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build and train model
model = build_lstm((X.shape[1], X.shape[2]))
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)

# Save model
model.save('models/lstm_rul_model.h5')
print("✅ Model trained and saved successfully!")