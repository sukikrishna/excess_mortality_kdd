import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# File paths
files = {
    "LSTM": "horizon_data_lstm_vs_sarima_2020-2023.csv",
    "Seq2Seq+Attn": "horizon_data_seq2seq_attn_vs_sarima_2020-2023.csv",
    "Seq2Seq": "horizon_data_seq2seq_vs_sarima_2020-2023.csv",
    "TCN": "horizon_data_tcn_vs_sarima_2020-2023.csv",
    "Transformer": "horizon_data_transformer_vs_sarima_2020-2023.csv"
}

# Column prefixes
prefixes = {
    "LSTM": "LSTM",
    "Seq2Seq+Attn": "Seq2Seq+Attention",
    "Seq2Seq": "Seq2Seq",
    "TCN": "TCN",
    "Transformer": "Transformer"
}

def compute_metrics(df, model_col, true_col="Actual", pi_lower=None, pi_upper=None):
    error = df[true_col] - df[model_col]
    abs_error = np.abs(error)
    pct_error = np.abs(error / df[true_col]) * 100
    squared_error = error ** 2

    rmse = np.sqrt(np.mean(squared_error))
    rmse_std = np.std(np.sqrt(squared_error))
    mae = np.mean(abs_error)
    mae_std = np.std(abs_error)
    mape = np.mean(pct_error)
    mape_std = np.std(pct_error)
    mse = np.mean(squared_error)
    mse_std = np.std(squared_error)

    coverage = ((df[true_col] >= df[pi_lower]) & (df[true_col] <= df[pi_upper])).mean() * 100 if pi_lower and pi_upper else np.nan

    return {
        "RMSE": rmse, "RMSE Std": rmse_std,
        "MAE": mae, "MAE Std": mae_std,
        "MAPE": mape, "MAPE Std": mape_std,
        "MSE": mse, "MSE Std": mse_std,
        "PI Coverage": coverage
    }

def evaluate_model(df, prefix, dataset="Train"):
    df_split = df[df["Dataset"] == dataset]
    true_col = "Actual"
    pred_col = f"{prefix}_Mean"
    pi_lower = f"{prefix}_PI_Lower"
    pi_upper = f"{prefix}_PI_Upper"
    return compute_metrics(df_split, pred_col, true_col, pi_lower, pi_upper)

def evaluate_sarima(df, dataset="Train"):
    df_split = df[df["Dataset"] == dataset]
    true_col = "Actual"
    pred_col = "SARIMA_Mean"
    pi_lower = "SARIMA_PI_Lower"
    pi_upper = "SARIMA_PI_Upper"
    return compute_metrics(df_split, pred_col, true_col, pi_lower, pi_upper)

# Collect results
rows = []

for model, path in files.items():
    df = pd.read_csv(path)
    prefix = prefixes[model]

    for split in ["Train", "Test"]:
        # Evaluate model and SARIMA
        model_metrics = evaluate_model(df, prefix, split)
        sarima_metrics = evaluate_sarima(df, split)

        # Append both
        rows.append({
            "Model": model,
            "Type": "Target",
            "Split": split,
            **{f"{k}": v for k, v in model_metrics.items()}
        })
        rows.append({
            "Model": model,
            "Type": "SARIMA",
            "Split": split,
            **{f"{k}": v for k, v in sarima_metrics.items()}
        })

# Create and save summary
summary_df = pd.DataFrame(rows)
summary_df.to_csv("model_and_sarima_metrics_with_std.csv", index=False)
print("Saved: model_and_sarima_metrics_with_std.csv")
