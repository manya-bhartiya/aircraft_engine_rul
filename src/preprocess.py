import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Calculate Remaining Useful Life (RUL)
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']

    # Normalize sensor and health features
    feature_cols = [col for col in df.columns if col not in ['engine_id', 'cycle', 'max_cycle', 'RUL']]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols, scaler


def create_sequences(df, feature_cols, window=30):
    X, y = [], []
    for eid in df['engine_id'].unique():
        temp = df[df['engine_id'] == eid]
        for i in range(len(temp) - window):
            seq_x = temp[feature_cols].iloc[i:i+window].values
            seq_y = temp['RUL'].iloc[i+window]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)