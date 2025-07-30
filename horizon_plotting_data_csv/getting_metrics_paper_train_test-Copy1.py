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

    # Mean and Std Dev
    rmse = np.sqrt(np.mean(squared_error))
    rmse_std = np.std(np.sqrt(squared_error))
    mae = np.mean(abs_error)
    mae_std = np.std(abs_error)
    mape = np.mean(pct_error)
    mape_std = np.std(pct_error)
    mse = np.mean(squared_error)
    mse_std = np.std(squared_error)
    
    # PI coverage
    if pi_lower and pi_upper:
        coverage = ((df[true_col] >= df[pi_lower]) & (df[true_col] <= df[pi_upper])).mean() * 100
    else:
        coverage = np.nan

    return {
        "RMSE": rmse, "RMSE Std": rmse_std,
        "MAE": mae, "MAE Std": mae_std,
        "MAPE": mape, "MAPE Std": mape_std,
        "MSE": mse, "MSE Std": mse_std,
        "PI Coverage": coverage
    }

def evaluate_model(df, prefix):
    mean_col = f"{prefix}_Mean"
    pi_lower_col = f"{prefix}_PI_Lower"
    pi_upper_col = f"{prefix}_PI_Upper"
    train_df = df[df["Dataset"] == "Train"]
    test_df = df[df["Dataset"] == "Test"]
    train_metrics = compute_metrics(train_df, mean_col, pi_lower=pi_lower_col, pi_upper=pi_upper_col)
    test_metrics = compute_metrics(test_df, mean_col, pi_lower=pi_lower_col, pi_upper=pi_upper_col)
    return train_metrics, test_metrics

# Build metrics table
rows = []
for model, path in files.items():
    df = pd.read_csv(path)
    prefix = prefixes[model]
    train, test = evaluate_model(df, prefix)

    rows.append({
        "Model": model,
        "Train RMSE": round(train["RMSE"], 3),
        "Train RMSE Std": round(train["RMSE Std"], 3),
        "Train MAE": round(train["MAE"], 3),
        "Train MAE Std": round(train["MAE Std"], 3),
        "Train MAPE (%)": f"{train['MAPE']:.2f}%",
        "Train MAPE Std": f"{train['MAPE Std']:.2f}%",
        "Train MSE": round(train["MSE"], 1),
        "Train MSE Std": round(train["MSE Std"], 1),
        "Train PI Coverage (%)": f"{train['PI Coverage']:.1f}%",
        "Test RMSE": round(test["RMSE"], 3),
        "Test RMSE Std": round(test["RMSE Std"], 3),
        "Test MAE": round(test["MAE"], 3),
        "Test MAE Std": round(test["MAE Std"], 3),
        "Test MAPE (%)": f"{test['MAPE']:.2f}%",
        "Test MAPE Std": f"{test['MAPE Std']:.2f}%",
        "Test MSE": round(test["MSE"], 1),
        "Test MSE Std": round(test["MSE Std"], 1),
        "Test PI Coverage (%)": f"{test['PI Coverage']:.1f}%"
    })

# Save to CSV
summary_df = pd.DataFrame(rows)
summary_df.to_csv("model_metrics_summary_with_std.csv", index=False)
print("Saved metrics with standard deviations to model_metrics_summary_with_std.csv")
