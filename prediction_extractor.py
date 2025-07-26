import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Configuration
RESULTS_DIR = 'final_evaluation_results'
OUTPUT_DIR = 'extracted_predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAMES = {
    'sarima': 'SARIMA',
    'lstm': 'LSTM',
    'tcn': 'TCN',
    'seq2seq_attn': 'Seq2Seq_Attention',
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
    
    return all_predictions, data_info

def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """Calculate prediction intervals"""
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = 1.96  # 95% confidence interval
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

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
    
    # Calculate mean and std predictions across all trials
    if model_name == 'sarima':
        # SARIMA has full length predictions
        mean_train_pred = np.mean(all_train_pred, axis=0)
        std_train_pred = np.std(all_train_pred, axis=0)
        mean_test_pred = np.mean(all_test_pred, axis=0)
        std_test_pred = np.std(all_test_pred, axis=0)
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
        std_train_pred = np.std(train_preds_truncated, axis=0)
        mean_test_pred = np.mean(test_preds_truncated, axis=0)
        std_test_pred = np.std(test_preds_truncated, axis=0)
        train_true = train_true_truncated[0]
        test_true = test_true_truncated[0]
    
    return train_true, mean_train_pred, std_train_pred, test_true, mean_test_pred, std_test_pred

def create_long_format_dataset(all_predictions, data_info):
    """Create a long-format dataset with all predictions aligned by date"""
    
    # Get date information
    train_val_data = data_info['train_val_data']
    test_data = data_info['test_data']
    
    # Initialize the main dataframe with dates and actual values
    full_dates = pd.concat([train_val_data['Month'], test_data['Month']], ignore_index=True)
    full_actual = pd.concat([train_val_data['Deaths'], test_data['Deaths']], ignore_index=True)
    
    result_df = pd.DataFrame({
        'Date': full_dates,
        'Actual': full_actual,
        'Period': ['Training'] * len(train_val_data) + ['Test'] * len(test_data)
    })
    
    # Add predictions for each model
    for model_name in all_predictions.keys():
        print(f"Processing {model_name}...")
        
        train_true, mean_train_pred, std_train_pred, test_true, mean_test_pred, std_test_pred = \
            aggregate_predictions_across_trials(all_predictions[model_name], model_name)
        
        # Calculate prediction intervals
        train_lower, train_upper = calculate_prediction_intervals(train_true, mean_train_pred)
        test_lower, test_upper = calculate_prediction_intervals(test_true, mean_test_pred)
        
        # Initialize prediction columns with NaN
        pred_col = f'{MODEL_NAMES[model_name]}_Prediction'
        lower_col = f'{MODEL_NAMES[model_name]}_Lower_PI'
        upper_col = f'{MODEL_NAMES[model_name]}_Upper_PI'
        std_col = f'{MODEL_NAMES[model_name]}_Std'
        
        result_df[pred_col] = np.nan
        result_df[lower_col] = np.nan
        result_df[upper_col] = np.nan
        result_df[std_col] = np.nan
        
        if model_name == 'sarima':
            # SARIMA has predictions for entire training period
            result_df.loc[:len(train_val_data)-1, pred_col] = mean_train_pred
            result_df.loc[:len(train_val_data)-1, lower_col] = train_lower
            result_df.loc[:len(train_val_data)-1, upper_col] = train_upper
            result_df.loc[:len(train_val_data)-1, std_col] = std_train_pred
        else:
            # Other models start predictions after lookback period
            lookback = len(train_val_data) - len(train_true)
            result_df.loc[lookback:len(train_val_data)-1, pred_col] = mean_train_pred
            result_df.loc[lookback:len(train_val_data)-1, lower_col] = train_lower
            result_df.loc[lookback:len(train_val_data)-1, upper_col] = train_upper
            result_df.loc[lookback:len(train_val_data)-1, std_col] = std_train_pred
        
        # Add test predictions
        test_start_idx = len(train_val_data)
        result_df.loc[test_start_idx:test_start_idx+len(test_true)-1, pred_col] = mean_test_pred
        result_df.loc[test_start_idx:test_start_idx+len(test_true)-1, lower_col] = test_lower
        result_df.loc[test_start_idx:test_start_idx+len(test_true)-1, upper_col] = test_upper
        result_df.loc[test_start_idx:test_start_idx+len(test_true)-1, std_col] = std_test_pred
    
    return result_df

def create_individual_model_files(all_predictions, data_info):
    """Create individual CSV files for each model's predictions"""
    
    train_val_data = data_info['train_val_data']
    test_data = data_info['test_data']
    
    for model_name in all_predictions.keys():
        print(f"Creating individual files for {model_name}...")
        
        train_true, mean_train_pred, std_train_pred, test_true, mean_test_pred, std_test_pred = \
            aggregate_predictions_across_trials(all_predictions[model_name], model_name)
        
        # Calculate prediction intervals
        train_lower, train_upper = calculate_prediction_intervals(train_true, mean_train_pred)
        test_lower, test_upper = calculate_prediction_intervals(test_true, mean_test_pred)
        
        # Create training predictions file
        if model_name == 'sarima':
            # SARIMA has predictions for entire training period
            train_dates = train_val_data['Month'].values
            train_actual = train_val_data['Deaths'].values
        else:
            # Other models start after lookback period
            lookback = len(train_val_data) - len(train_true)
            train_dates = train_val_data['Month'].iloc[lookback:].values
            train_actual = train_val_data['Deaths'].iloc[lookback:].values
        
        train_df = pd.DataFrame({
            'Date': train_dates,
            'Actual': train_actual,
            'Prediction': mean_train_pred,
            'Prediction_Std': std_train_pred,
            'Lower_PI': train_lower,
            'Upper_PI': train_upper,
            'Period': 'Training'
        })
        
        # Create test predictions file
        test_df = pd.DataFrame({
            'Date': test_data['Month'].values,
            'Actual': test_true,
            'Prediction': mean_test_pred,
            'Prediction_Std': std_test_pred,
            'Lower_PI': test_lower,
            'Upper_PI': test_upper,
            'Period': 'Test'
        })
        
        # Save individual files
        model_dir = os.path.join(OUTPUT_DIR, MODEL_NAMES[model_name])
        os.makedirs(model_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(model_dir, 'training_predictions.csv'), index=False)
        test_df.to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False)
        
        # Create combined file for this model
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df.to_csv(os.path.join(model_dir, 'all_predictions.csv'), index=False)
        
        print(f"  Saved files for {MODEL_NAMES[model_name]} in {model_dir}")

def create_summary_statistics(all_predictions, data_info):
    """Create summary statistics across all models"""
    
    summary_stats = []
    
    for model_name in all_predictions.keys():
        train_true, mean_train_pred, std_train_pred, test_true, mean_test_pred, std_test_pred = \
            aggregate_predictions_across_trials(all_predictions[model_name], model_name)
        
        # Calculate metrics
        train_rmse = np.sqrt(np.mean((train_true - mean_train_pred) ** 2))
        train_mae = np.mean(np.abs(train_true - mean_train_pred))
        train_mape = np.mean(np.abs((train_true - mean_train_pred) / train_true)) * 100
        
        test_rmse = np.sqrt(np.mean((test_true - mean_test_pred) ** 2))
        test_mae = np.mean(np.abs(test_true - mean_test_pred))
        test_mape = np.mean(np.abs((test_true - mean_test_pred) / test_true)) * 100
        
        # Calculate prediction intervals coverage
        train_lower, train_upper = calculate_prediction_intervals(train_true, mean_train_pred)
        test_lower, test_upper = calculate_prediction_intervals(test_true, mean_test_pred)
        
        train_coverage = np.mean((train_true >= train_lower) & (train_true <= train_upper)) * 100
        test_coverage = np.mean((test_true >= test_lower) & (test_true <= test_upper)) * 100
        
        # Calculate prediction interval width
        train_pi_width = np.mean(train_upper - train_lower)
        test_pi_width = np.mean(test_upper - test_lower)
        
        summary_stats.append({
            'Model': MODEL_NAMES[model_name],
            'Train_RMSE': train_rmse,
            'Train_MAE': train_mae,
            'Train_MAPE': train_mape,
            'Train_PI_Coverage': train_coverage,
            'Train_PI_Width': train_pi_width,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_MAPE': test_mape,
            'Test_PI_Coverage': test_coverage,
            'Test_PI_Width': test_pi_width,
            'Prediction_Uncertainty_Train': np.mean(std_train_pred),
            'Prediction_Uncertainty_Test': np.mean(std_test_pred)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.round(4)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)
    
    return summary_df

def create_pairwise_comparisons(all_predictions):
    """Create pairwise model comparison metrics"""
    
    models = list(all_predictions.keys())
    model_data = {}
    
    # Get aggregated predictions for all models
    for model_name in models:
        train_true, mean_train_pred, std_train_pred, test_true, mean_test_pred, std_test_pred = \
            aggregate_predictions_across_trials(all_predictions[model_name], model_name)
        
        # Combine train and test (align by removing lookback differences)
        if model_name == 'sarima':
            all_true = np.concatenate([train_true, test_true])
            all_pred = np.concatenate([mean_train_pred, mean_test_pred])
        else:
            all_true = np.concatenate([train_true, test_true])
            all_pred = np.concatenate([mean_train_pred, mean_test_pred])
        
        model_data[model_name] = (all_true, all_pred)
    
    # Create pairwise comparison matrix
    comparison_results = []
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:
                true1, pred1 = model_data[model1]
                true2, pred2 = model_data[model2]
                
                # Align lengths if necessary
                min_len = min(len(pred1), len(pred2))
                true_aligned = true1[:min_len]
                pred1_aligned = pred1[:min_len]
                pred2_aligned = pred2[:min_len]
                
                # Calculate relative performance metrics
                rmse1 = np.sqrt(np.mean((true_aligned - pred1_aligned) ** 2))
                rmse2 = np.sqrt(np.mean((true_aligned - pred2_aligned) ** 2))
                rmse_ratio = rmse1 / rmse2
                
                mae1 = np.mean(np.abs(true_aligned - pred1_aligned))
                mae2 = np.mean(np.abs(true_aligned - pred2_aligned))
                mae_ratio = mae1 / mae2
                
                # Calculate correlation between predictions
                pred_correlation = np.corrcoef(pred1_aligned, pred2_aligned)[0, 1]
                
                # Calculate prediction interval overlap
                lower1, upper1 = calculate_prediction_intervals(true_aligned, pred1_aligned)
                lower2, upper2 = calculate_prediction_intervals(true_aligned, pred2_aligned)
                
                overlap_lower = np.maximum(lower1, lower2)
                overlap_upper = np.minimum(upper1, upper2)
                overlap_width = np.maximum(0, overlap_upper - overlap_lower)
                width1 = upper1 - lower1
                width2 = upper2 - lower2
                avg_width = (width1 + width2) / 2
                pi_overlap = np.mean(overlap_width / avg_width) * 100
                
                comparison_results.append({
                    'Model_1': MODEL_NAMES[model1],
                    'Model_2': MODEL_NAMES[model2],
                    'RMSE_Ratio': rmse_ratio,
                    'MAE_Ratio': mae_ratio,
                    'Prediction_Correlation': pred_correlation,
                    'PI_Overlap_Percent': pi_overlap
                })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.round(4)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_model_comparisons.csv'), index=False)
    
    return comparison_df

def main():
    """Main extraction function"""
    print("Starting prediction data extraction...")
    
    # Load all results
    all_predictions, data_info = load_results()
    
    print(f"Found {len(all_predictions)} models with predictions")
    
    # Create long-format dataset
    print("\nCreating comprehensive long-format dataset...")
    long_df = create_long_format_dataset(all_predictions, data_info)
    long_df.to_csv(os.path.join(OUTPUT_DIR, 'all_models_predictions_long_format.csv'), index=False)
    print(f"  Saved: all_models_predictions_long_format.csv")
    print(f"  Dataset shape: {long_df.shape}")
    
    # Create individual model files
    print("\nCreating individual model prediction files...")
    create_individual_model_files(all_predictions, data_info)
    
    # Create summary statistics
    print("\nCalculating summary statistics...")
    summary_df = create_summary_statistics(all_predictions, data_info)
    print(f"  Saved: model_performance_summary.csv")
    print("\nModel Performance Summary:")
    print(summary_df[['Model', 'Test_RMSE', 'Test_MAE', 'Test_MAPE']].to_string(index=False))
    
    # Create pairwise comparisons
    print("\nCreating pairwise model comparisons...")
    comparison_df = create_pairwise_comparisons(all_predictions)
    print(f"  Saved: pairwise_model_comparisons.csv")
    
    # Create data overview
    overview = {
        'Total_Models': len(all_predictions),
        'Training_Period': f"{data_info['train_val_data']['Month'].min()} to {data_info['train_val_data']['Month'].max()}",
        'Test_Period': f"{data_info['test_data']['Month'].min()} to {data_info['test_data']['Month'].max()}",
        'Training_Samples': len(data_info['train_val_data']),
        'Test_Samples': len(data_info['test_data']),
        'Total_Samples': len(data_info['full_data']),
        'Models_Evaluated': list(MODEL_NAMES.values())
    }
    
    overview_df = pd.DataFrame([overview])
    overview_df.to_csv(os.path.join(OUTPUT_DIR, 'data_overview.csv'), index=False)
    
    print(f"\nExtraction complete! All files saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - all_models_predictions_long_format.csv (comprehensive dataset)")
    print("  - model_performance_summary.csv (performance metrics)")
    print("  - pairwise_model_comparisons.csv (model comparisons)")
    print("  - data_overview.csv (dataset information)")
    print("  - [ModelName]/training_predictions.csv (individual model files)")
    print("  - [ModelName]/test_predictions.csv (individual model files)")
    print("  - [ModelName]/all_predictions.csv (individual model files)")
    
    print(f"\nDirectory structure:")
    print(f"{OUTPUT_DIR}/")
    for model in MODEL_NAMES.values():
        print(f"├── {model}/")
        print(f"│   ├── training_predictions.csv")
        print(f"│   ├── test_predictions.csv")
        print(f"│   └── all_predictions.csv")
    print("├── all_models_predictions_long_format.csv")
    print("├── model_performance_summary.csv")
    print("├── pairwise_model_comparisons.csv")
    print("└── data_overview.csv")

if __name__ == "__main__":
    main()