import pandas as pd

def load_cmapss_to_csv(input_path, output_path):
    # Define column names
    col_names = [
        'id', 'cycle', 
        'setting1', 'setting2', 'setting3',
        'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6',
        'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11',
        'sensor12', 'sensor13', 'sensor14', 'sensor15', 'sensor16',
        'sensor17', 'sensor18', 'sensor19', 'sensor20', 'sensor21'
    ]

    # Load raw text data
    df = pd.read_csv(input_path, sep=r'\s+', header=None, names=col_names)

    # Compute Remaining Useful Life (RUL)
    df['max_cycle'] = df.groupby('id')['cycle'].transform('max')
    df['RUL'] = df['max_cycle'] - df['cycle']

    # Drop helper column
    df.drop('max_cycle', axis=1, inplace=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Saved processed dataset to {output_path}")

# Example usage
if __name__ == "__main__":
    load_cmapss_to_csv("data/train_FD001.txt", "data/engine_data.csv")