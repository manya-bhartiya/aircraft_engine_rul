import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Hide TensorFlow warnings

import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from src.preprocess import load_and_preprocess

# Ensure output flushes properly
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def assess_component_health(rul_value):
    """Classify health status for RUL values (range ~1–7)."""
    if rul_value >= 5.5:
        return "Good", "🟢"
    elif 3.5 <= rul_value < 5.5:
        return "Fair", "🟡"
    else:
        return "Poor", "🔴"


def predict_rul_and_health(data_path, model_path, window=30, save_csv=True):
    """
    Load dataset and trained model, predict RUL for each engine,
    and classify overall health.
    """
    print(" Loading and preprocessing data...")
    df, features, _ = load_and_preprocess(data_path)
    print("✅ Data loaded successfully.")

    # 🔽 Detect engine ID column automatically
    engine_col = None
    for col in df.columns:
        if 'engine' in col.lower() or 'unit' in col.lower() or 'id' in col.lower():
            engine_col = col
            break

    if engine_col is None:
        raise KeyError("❌ Could not find an engine ID column in the dataset!")

    print(f"ℹ️ Using engine ID column: '{engine_col}'")

    # Load trained LSTM model
    print("🧠 Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully.")

    results = []

    for eid in df[engine_col].unique():
        engine_df = df[df[engine_col] == eid].reset_index(drop=True)

        # Prepare the last 'window' cycles of data
        seq_df = engine_df[features].iloc[-window:]
        seq = seq_df.values
        seq = np.expand_dims(seq, axis=0)  # Shape: (1, timesteps, features)

        # Predict RUL
        pred = float(model.predict(seq, verbose=0)[0][0])
        status, indicator = assess_component_health(pred)

        results.append({
            "Engine_ID": int(eid) if str(eid).isdigit() else eid,
            "Predicted_RUL": round(pred, 3),
            "Health_Status": status,
            "Indicator": indicator
        })

    result_df = pd.DataFrame(results)

    if save_csv:
        out_path = os.path.join(os.path.dirname(data_path), "predicted_rul_health.csv")
        result_df.to_csv(out_path, index=False)
        print(f"💾 Predictions saved to: {out_path}")

    return result_df


def prompt_and_show(results_df):
    """Prompt user for engine ID(s) and display results."""
    raw = input("\nEnter Engine ID (single or comma-separated), or press Enter to view all: ").strip()

    if raw == "":
        print("\n📋 All predictions:")
        print(results_df.to_string(index=False))
        return

    # Parse engine IDs (support comma-separated input)
    tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
    parsed = []
    for t in tokens:
        try:
            parsed.append(int(t))
        except ValueError:
            parsed.append(t)

    filtered = results_df[results_df["Engine_ID"].isin(parsed)]

    if filtered.empty:
        print(f"⚠️ No data found for Engine ID(s): {tokens}")
    else:
        print("\n📊 Engine Health Report:")
        print(filtered.to_string(index=False))


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Build absolute paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "engine_data.csv")
    model_path = os.path.join(base_dir, "models", "lstm_rul_model.h5")

    try:
        print("🚀 Running predictions...")
        results = predict_rul_and_health(data_path, model_path, window=30, save_csv=True)
        prompt_and_show(results)

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except KeyError as e:
        print(f"⚠️ Data error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {type(e)._name_}: {e}")