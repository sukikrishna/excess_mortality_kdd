import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
RESULTS_DIR = 'final_evaluation_results'
PLOTS_DIR = 'model_comparison_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model names and colors
MODEL_COLORS = {
    'sarima': '#2E86AB',      # Blue
    'lstm': '#A23B72',        # Purple
    'tcn': '#F18F01',         # Orange
    'seq2seq_attn': '#C73E1D', # Red
    'transformer': '#2D5016'   # Dark Green
}

MODEL_NAMES = {
    'sarima': 'SARIMA',
    'lstm': 'LSTM',
    'tcn': 'TCN',
    'seq2seq_attn': 'Seq2Seq+Attention',
    'transformer': 'Transformer'
}

def load_results():
    """Load all results and data"""
    print("Loading results and data...")
    
    # Load predictions
    with open(os.path.join(RESULTS_DIR, 'all_predictions.pkl'), 'rb') as f:
        all_predictions = pickle.load(f)
    
    # Load data splits
    with open(os.path.join(RESULTS_DIR, 'data_splits.pkl'), 'rb') as f:
        data_info = pickle.load(f)
    
    # Load final comparison metrics
    final_comparison = pd.read_csv(os.path.join(RESULTS_DIR, 'final_model_comparison.csv'), index_col=0)
    
    return all_predictions, data_info, final_comparison

def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals"""
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = 1.96  # 95% confidence interval
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

def calculate_pi_overlap(lower1, upper1, lower2, upper2):
    """Calculate prediction interval overlap percentage"""
    overlap_lower = np.maximum(lower1, lower2)
    overlap_upper = np.minimum(upper1, upper2)
    overlap_width = np.maximum(0, overlap_upper - overlap_lower)
    width1 = upper1 - lower1
    width2 = upper2 - lower2
    avg_width = (width1 + width2) / 2
    overlap_percentage = np.mean(overlap_width / avg_width) * 100
    return overlap_percentage

def aggregate_predictions_across_trials(model_predictions, model_name):
    """Aggregate predictions across all trials and seeds for a model"""
    all_train_true = []
    all_train_pred = []
    all_test_true = []
    all_test_pred = []
    
    for seed in model_predictions:
        for trial_data in model_predictions[seed]:
            all_train_true.append(trial_data['train_true'])
            all_train_pred.append(trial_data['train_pred'])
            all_test_true.append(trial_data['test_true'])
            all_test_pred.append(trial_data['test_pred'])
    
    # Calculate mean predictions across all trials
    if model_name == 'sarima':
        # SARIMA has full length predictions
        mean_train_pred = np.mean(all_train_pred, axis=0)
        mean_test_pred = np.mean(all_test_pred, axis=0)
        train_true = all_train_true[0]  # Same across trials
        test_true = all_test_true[0]    # Same across trials
    else:
        # Other models may have shorter training predictions due to lookback
        min_train_len = min(len(pred) for pred in all_train_pred)
        min_test_len = min(len(pred) for pred in all_test_pred)
        
        # Truncate to minimum length
        train_preds_truncated = [pred[:min_train_len] for pred in all_train_pred]
        test_preds_truncated = [pred[:min_test_len] for pred in all_test_pred]
        train_true_truncated = [true[:min_train_len] for true in all_train_true]
        test_true_truncated = [true[:min_test_len] for true in all_test_true]
        
        mean_train_pred = np.mean(train_preds_truncated, axis=0)
        mean_test_pred = np.mean(test_preds_truncated, axis=0)
        train_true = train_true_truncated[0]
        test_true = test_true_truncated[0]
    
    return train_true, mean_train_pred, test_true, mean_test_pred

def create_model_comparison_plot(sarima_data, model_data, model_name, data_info, save_path):
    """Create comparison plot between SARIMA and another model"""
    
    # Unpack data
    sarima_train_true, sarima_train_pred, sarima_test_true, sarima_test_pred = sarima_data
    model_train_true, model_train_pred, model_test_true, model_test_pred = model_data
    
    # Get date information
    full_data = data_info['full_data']
    train_val_data = data_info['train_val_data']
    test_data = data_info['test_data']
    
    # Create date ranges
    if model_name == 'sarima':
        # SARIMA has predictions for entire training period
        train_dates = train_val_data['Month'].values
        train_actual = train_val_data['Deaths'].values
        train_pred_sarima = sarima_train_pred
        train_pred_model = model_train_pred
    else:
        # Other models start predictions after lookback period
        lookback = len(train_val_data) - len(model_train_true)
        train_dates = train_val_data['Month'].iloc[lookback:].values
        train_actual = train_val_data['Deaths'].iloc[lookback:].values
        train_pred_sarima = sarima_train_pred[lookback:]  # Align SARIMA with model
        train_pred_model = model_train_pred
    
    test_dates = test_data['Month'].values
    test_actual = test_data['Deaths'].values
    
    # Combine train and test data for plotting
    all_dates = np.concatenate([train_dates, test_dates])
    all_actual = np.concatenate([train_actual, test_actual])
    all_pred_sarima = np.concatenate([train_pred_sarima, sarima_test_pred])
    all_pred_model = np.concatenate([train_pred_model, model_test_pred])
    
    # Calculate prediction intervals
    sarima_lower, sarima_upper = calculate_prediction_intervals(all_actual, all_pred_sarima)
    model_lower, model_upper = calculate_prediction_intervals(all_actual, all_pred_model)
    
    # Calculate PI overlap
    pi_overlap = calculate_pi_overlap(sarima_lower, sarima_upper, model_lower, model_upper)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual data
    ax.plot(all_dates, all_actual, label='Actual Data', color='black', linewidth=2, zorder=5)
    
    # Plot SARIMA predictions
    ax.plot(all_dates, all_pred_sarima, label='SARIMA Predictions', 
            color=MODEL_COLORS['sarima'], linewidth=1.5, alpha=0.8)
    
    # Plot model predictions
    ax.plot(all_dates, all_pred_model, label=f'{MODEL_NAMES[model_name]} Predictions', 
            color=MODEL_COLORS[model_name], linewidth=1.5, alpha=0.8)
    
    # Add prediction intervals
    ax.fill_between(all_dates, sarima_lower, sarima_upper, 
                    color=MODEL_COLORS['sarima'], alpha=0.2, 
                    label='SARIMA 95% PI')
    ax.fill_between(all_dates, model_lower, model_upper, 
                    color=MODEL_COLORS[model_name], alpha=0.2, 
                    label=f'{MODEL_NAMES[model_name]} 95% PI')
    
    # Add vertical line at forecast start
    forecast_start = test_data['Month'].iloc[0]
    ax.axvline(forecast_start, color='red', linestyle='--', alpha=0.7, 
               label='Forecast Start (Jan 2020)')
    
    # Formatting
    ax.set_title(f'Mortality Forecasting: SARIMA vs {MODEL_NAMES[model_name]}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Deaths', fontsize=12)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'PI Overlap: {pi_overlap:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Add caption
    caption = (f'Figure: Comparison of actual substance overdose mortality data with SARIMA and '
              f'{MODEL_NAMES[model_name]} model predictions from 2015 to 2020, including 95% prediction intervals. '
              f'The black line represents actual data, colored lines show model predictions with shaded '
              f'prediction intervals. The red dashed line marks the start of out-of-sample forecasting in January 2020.')
    
    plt.figtext(0.5, -0.02, caption, wrap=True, ha='center', fontsize=9, style='italic')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return pi_overlap

def create_metrics_comparison_table(final_comparison, all_predictions, data_info):
    """Create comprehensive metrics comparison table"""
    
    # Calculate additional metrics for each model
    metrics_data = []
    
    for model_name in ['sarima', 'lstm', 'tcn', 'seq2seq_attn', 'transformer']:
        if model_name in all_predictions:
            # Get aggregated predictions
            train_true, train_pred, test_true, test_pred = aggregate_predictions_across_trials(
                all_predictions[model_name], model_name)
            
            # Calculate metrics for training set
            train_rmse = np.sqrt(np.mean((train_true - train_pred) ** 2))
            train_mae = np.mean(np.abs(train_true - train_pred))
            train_mape = np.mean(np.abs((train_true - train_pred) / train_true)) * 100
            train_mse = np.mean((train_true - train_pred) ** 2)
            
            # Calculate metrics for test set
            test_rmse = np.sqrt(np.mean((test_true - test_pred) ** 2))
            test_mae = np.mean(np.abs(test_true - test_pred))
            test_mape = np.mean(np.abs((test_true - test_pred) / test_true)) * 100
            test_mse = np.mean((test_true - test_pred) ** 2)
            
            # Calculate prediction intervals and coverage
            train_lower, train_upper = calculate_prediction_intervals(train_true, train_pred)
            test_lower, test_upper = calculate_prediction_intervals(test_true, test_pred)
            
            train_coverage = np.mean((train_true >= train_lower) & (train_true <= train_upper)) * 100
            test_coverage = np.mean((test_true >= test_lower) & (test_true <= test_upper)) * 100
            
            metrics_data.append({
                'Model': MODEL_NAMES[model_name],
                'Train_RMSE': f"{train_rmse:.3f}",
                'Train_MAE': f"{train_mae:.3f}",
                'Train_MAPE': f"{train_mape:.2f}%",
                'Train_MSE': f"{train_mse:.1f}",
                'Train_PI_Coverage': f"{train_coverage:.1f}%",
                'Test_RMSE': f"{test_rmse:.3f}",
                'Test_MAE': f"{test_mae:.3f}",
                'Test_MAPE': f"{test_mape:.2f}%",
                'Test_MSE': f"{test_mse:.1f}",
                'Test_PI_Coverage': f"{test_coverage:.1f}%"
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    metrics_df.to_csv(os.path.join(PLOTS_DIR, 'comprehensive_metrics_table.csv'), index=False)
    
    # Create a nice table plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns,
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color the header
    for i in range(len(metrics_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(metrics_df) + 1):
        for j in range(len(metrics_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Comprehensive Model Performance Comparison\n(Training and Test Set Metrics)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(PLOTS_DIR, 'metrics_comparison_table.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return metrics_df

def create_pi_overlap_heatmap(all_predictions):
    """Create heatmap showing prediction interval overlaps between all models"""
    
    models = ['sarima', 'lstm', 'tcn', 'seq2seq_attn', 'transformer']
    model_preds = {}
    
    # Get aggregated predictions for all models
    for model in models:
        if model in all_predictions:
            train_true, train_pred, test_true, test_pred = aggregate_predictions_across_trials(
                all_predictions[model], model)
            
            # Combine train and test
            all_true = np.concatenate([train_true, test_true])
            all_pred = np.concatenate([train_pred, test_pred])
            
            # Calculate prediction intervals
            lower, upper = calculate_prediction_intervals(all_true, all_pred)
            model_preds[model] = (lower, upper)
    
    # Calculate overlap matrix
    overlap_matrix = np.zeros((len(models), len(models)))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if model1 in model_preds and model2 in model_preds:
                lower1, upper1 = model_preds[model1]
                lower2, upper2 = model_preds[model2]
                overlap = calculate_pi_overlap(lower1, upper1, lower2, upper2)
                overlap_matrix[i, j] = overlap
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    model_labels = [MODEL_NAMES[model] for model in models]
    
    im = ax.imshow(overlap_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.set_yticklabels(model_labels)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{overlap_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black" if overlap_matrix[i, j] < 50 else "white",
                          fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Interval Overlap (%)', rotation=270, labelpad=20)
    
    ax.set_title('Prediction Interval Overlap Between Models', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pi_overlap_heatmap.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return overlap_matrix

def main():
    """Main plotting function"""
    print("Starting comprehensive model plotting...")
    
    # Load all results
    all_predictions, data_info, final_comparison = load_results()
    
    # Get SARIMA data for comparisons
    sarima_data = aggregate_predictions_across_trials(all_predictions['sarima'], 'sarima')
    
    # Create individual comparison plots
    pi_overlaps = {}
    
    for model_name in ['lstm', 'tcn', 'seq2seq_attn', 'transformer']:
        if model_name in all_predictions:
            print(f"Creating comparison plot: SARIMA vs {model_name.upper()}")
            
            model_data = aggregate_predictions_across_trials(all_predictions[model_name], model_name)
            
            save_path = os.path.join(PLOTS_DIR, f'sarima_vs_{model_name}_comparison.png')
            pi_overlap = create_model_comparison_plot(sarima_data, model_data, model_name, 
                                                    data_info, save_path)
            pi_overlaps[model_name] = pi_overlap
            
            print(f"  Saved: {save_path}")
            print(f"  PI Overlap with SARIMA: {pi_overlap:.2f}%")
    
    # Create comprehensive metrics table
    print("\nCreating comprehensive metrics comparison table...")
    metrics_df = create_metrics_comparison_table(final_comparison, all_predictions, data_info)
    print("  Saved: comprehensive_metrics_table.csv and metrics_comparison_table.png")
    
    # Create PI overlap heatmap
    print("\nCreating prediction interval overlap heatmap...")
    overlap_matrix = create_pi_overlap_heatmap(all_predictions)
    print("  Saved: pi_overlap_heatmap.png")
    
    # Save PI overlap summary
    pi_overlap_summary = pd.DataFrame({
        'Model': [MODEL_NAMES[model] for model in pi_overlaps.keys()],
        'PI_Overlap_with_SARIMA': [f"{overlap:.2f}%" for overlap in pi_overlaps.values()]
    })
    pi_overlap_summary.to_csv(os.path.join(PLOTS_DIR, 'pi_overlap_summary.csv'), index=False)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")
    print("Generated files:")
    print("  - sarima_vs_[model]_comparison.png (4 plots)")
    print("  - comprehensive_metrics_table.csv")
    print("  - metrics_comparison_table.png")
    print("  - pi_overlap_heatmap.png")
    print("  - pi_overlap_summary.csv")
    
    print("\nPI Overlap Summary:")
    print(pi_overlap_summary.to_string(index=False))

if __name__ == "__main__":
    main()