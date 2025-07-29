import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_hyperparams_from_path(path, model_name):
    """Extract hyperparameters from the folder path based on model type"""
    
    if model_name in ['seq2seq', 'seq2seq_attn']:
        # Pattern for seq2seq: lookback_3_bs_8_epochs_50_enc_64_dec_64_att_False
        pattern = r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)_enc_(\d+)_dec_(\d+)_att_(True|False)'
        match = re.search(pattern, path)
        if match:
            return {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3)),
                'encoder_units': int(match.group(4)),
                'decoder_units': int(match.group(5)),
                'attention': match.group(6) == 'True'
            }
    
    elif model_name == 'transformer':
        # Pattern for transformer: lookback_12_bs_32_epochs_50_dmodel_64_heads_2
        pattern = r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)_dmodel_(\d+)_heads_(\d+)'
        match = re.search(pattern, path)
        if match:
            return {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3)),
                'd_model': int(match.group(4)),
                'n_heads': int(match.group(5))
            }
    
    elif model_name == 'sarima':
        # Pattern for SARIMA: order_(1, 1, 1)_seasonal_(1, 1, 1, 12)
        pattern = r'order_\((\d+),\s*(\d+),\s*(\d+)\)_seasonal_\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        match = re.search(pattern, path)
        if match:
            return {
                'p': int(match.group(1)),
                'd': int(match.group(2)),
                'q': int(match.group(3)),
                'seasonal_p': int(match.group(4)),
                'seasonal_d': int(match.group(5)),
                'seasonal_q': int(match.group(6)),
                'seasonal_period': int(match.group(7))
            }
    
    else:  # lstm, tcn
        # Pattern for LSTM/TCN: lookback_12_bs_32_epochs_50
        pattern = r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)'
        match = re.search(pattern, path)
        if match:
            return {
                'lookback': int(match.group(1)),
                'batch_size': int(match.group(2)),
                'epochs': int(match.group(3))
            }
    
    return None

def load_summary_metrics(file_path):
    """Load summary metrics from CSV file"""
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Extract mean, median, and std values
        metrics = {}
        for col in df.columns:
            if col in ['RMSE', 'MAE', 'MAPE']:  # Only keep the main metrics
                metrics[f'{col}_mean'] = df.loc['mean', col]
                metrics[f'{col}_median'] = df.loc['median', col] if 'median' in df.index else df.loc['mean', col]
                metrics[f'{col}_std'] = df.loc['std', col]
        return metrics
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def consolidate_single_model(model_name, results_dir='results_old_val'):
    """Consolidate hyperparameter results for a single model and save in model folder"""
    model_path = Path(results_dir) / model_name / 'fixed_seed_variability'
    
    if not model_path.exists():
        print(f"Path {model_path} does not exist")
        return
    
    train_results = []
    validation_results = []
    
    print(f"Processing {model_name}...")
    
    # Iterate through all hyperparameter combination folders
    config_count = 0
    for config_folder in model_path.iterdir():
        if config_folder.is_dir():
            config_count += 1
            hyperparams = extract_hyperparams_from_path(config_folder.name, model_name)
            if hyperparams is None:
                print(f"  Warning: Could not parse hyperparameters from {config_folder.name}")
                continue
            
            # Load train metrics
            train_file = config_folder / 'summary_metrics_train.csv'
            if train_file.exists():
                train_metrics = load_summary_metrics(train_file)
                if train_metrics:
                    result_row = {**hyperparams, **train_metrics, 'config_name': config_folder.name}
                    train_results.append(result_row)
            
            # Load validation metrics
            validation_file = config_folder / 'summary_metrics_validation.csv'
            if validation_file.exists():
                validation_metrics = load_summary_metrics(validation_file)
                if validation_metrics:
                    result_row = {**hyperparams, **validation_metrics, 'config_name': config_folder.name}
                    validation_results.append(result_row)
    
    print(f"  Found {config_count} configurations")
    
    # Convert to DataFrames and sort by RMSE_mean (best first)
    if train_results:
        train_df = pd.DataFrame(train_results)
        train_df = train_df.sort_values('RMSE_mean')
        
        # Save in model folder
        train_output_path = Path(results_dir) / model_name / 'best_hyperparameters_train.csv'
        train_df.to_csv(train_output_path, index=False)
        print(f"  ✓ Saved train results: {train_output_path}")
        print(f"    Best train RMSE: {train_df.iloc[0]['RMSE_mean']:.4f} ± {train_df.iloc[0]['RMSE_std']:.4f}")
        
        # Print best config based on model type
        print_best_config(train_df.iloc[0], model_name, "train")
    
    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        validation_df = validation_df.sort_values('RMSE_mean')
        
        # Save in model folder
        validation_output_path = Path(results_dir) / model_name / 'best_hyperparameters_validation.csv'
        validation_df.to_csv(validation_output_path, index=False)
        print(f"  ✓ Saved validation results: {validation_output_path}")
        print(f"    Best validation RMSE: {validation_df.iloc[0]['RMSE_mean']:.4f} ± {validation_df.iloc[0]['RMSE_std']:.4f}")
        
        # Print best config based on model type
        print_best_config(validation_df.iloc[0], model_name, "validation")
    
    print()
    return train_results, validation_results

def print_best_config(best_row, model_name, split_name):
    """Print the best configuration in a readable format"""
    if model_name in ['seq2seq', 'seq2seq_attn']:
        print(f"    Best {split_name} config: lookback={best_row['lookback']}, "
              f"bs={best_row['batch_size']}, epochs={best_row['epochs']}, "
              f"enc={best_row['encoder_units']}, dec={best_row['decoder_units']}, "
              f"att={best_row['attention']}")
    elif model_name == 'transformer':
        print(f"    Best {split_name} config: lookback={best_row['lookback']}, "
              f"bs={best_row['batch_size']}, epochs={best_row['epochs']}, "
              f"d_model={best_row['d_model']}, n_heads={best_row['n_heads']}")
    elif model_name == 'sarima':
        print(f"    Best {split_name} config: order=({best_row['p']}, {best_row['d']}, {best_row['q']}), "
              f"seasonal=({best_row['seasonal_p']}, {best_row['seasonal_d']}, "
              f"{best_row['seasonal_q']}, {best_row['seasonal_period']})")
    else:  # lstm, tcn
        print(f"    Best {split_name} config: lookback={best_row['lookback']}, "
              f"bs={best_row['batch_size']}, epochs={best_row['epochs']}")

def create_overall_comparison(model_names, results_dir='results_old_val'):
    """Create an overall comparison table across all models"""
    comparison_data = []
    
    for model_name in model_names:
        # Try to load validation results (preferred for model comparison)
        validation_file = Path(results_dir) / model_name / 'best_hyperparameters_validation.csv'
        train_file = Path(results_dir) / model_name / 'best_hyperparameters_train.csv'
        
        if validation_file.exists():
            df = pd.read_csv(validation_file)
            if len(df) > 0:
                best = df.iloc[0]
                comparison_data.append({
                    'model': model_name,
                    'split': 'validation',
                    'best_rmse_mean': best['RMSE_mean'],
                    'best_rmse_std': best['RMSE_std'],
                    'best_mae_mean': best['MAE_mean'],
                    'best_mae_std': best['MAE_std'],
                    'best_mape_mean': best['MAPE_mean'],
                    'best_mape_std': best['MAPE_std'],
                    'config_name': best['config_name']
                })
        
        if train_file.exists():
            df = pd.read_csv(train_file)
            if len(df) > 0:
                best = df.iloc[0]
                comparison_data.append({
                    'model': model_name,
                    'split': 'train',
                    'best_rmse_mean': best['RMSE_mean'],
                    'best_rmse_std': best['RMSE_std'],
                    'best_mae_mean': best['MAE_mean'],
                    'best_mae_std': best['MAE_std'],
                    'best_mape_mean': best['MAPE_mean'],
                    'best_mape_std': best['MAPE_std'],
                    'config_name': best['config_name']
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save overall comparison
        comparison_file = Path(results_dir) / 'model_comparison_summary.csv'
        comparison_df.to_csv(comparison_file, index=False)
        
        print("="*80)
        print("OVERALL MODEL COMPARISON (Validation Results)")
        print("="*80)
        
        # Show validation results sorted by RMSE
        val_results = comparison_df[comparison_df['split'] == 'validation'].sort_values('best_rmse_mean')
        if len(val_results) > 0:
            print("Validation RMSE Rankings:")
            for i, (_, row) in enumerate(val_results.iterrows(), 1):
                print(f"  {i}. {row['model'].upper()}: {row['best_rmse_mean']:.4f} ± {row['best_rmse_std']:.4f}")
        
        print(f"\nDetailed comparison saved to: {comparison_file}")
        
        return comparison_df
    
    return None

def process_all_models(model_names=None, results_dir='results_old_val'):
    """Process all specified models"""
    
    if model_names is None:
        # Auto-detect available models
        results_path = Path(results_dir)
        if results_path.exists():
            model_names = [d.name for d in results_path.iterdir() if d.is_dir()]
            print(f"Auto-detected models: {model_names}")
        else:
            print(f"Results directory {results_dir} does not exist!")
            return
    
    print("="*80)
    print("CONSOLIDATING HYPERPARAMETER RESULTS")
    print("="*80)
    
    all_train_results = {}
    all_validation_results = {}
    
    for model_name in model_names:
        train_results, validation_results = consolidate_single_model(model_name, results_dir)
        if train_results:
            all_train_results[model_name] = len(train_results)
        if validation_results:
            all_validation_results[model_name] = len(validation_results)
    
    # Create overall comparison
    create_overall_comparison(model_names, results_dir)
    
    print("="*80)
    print("CONSOLIDATION COMPLETE!")
    print("="*80)
    print("Files created for each model:")
    print("- best_hyperparameters_train.csv (sorted by train RMSE)")
    print("- best_hyperparameters_validation.csv (sorted by validation RMSE)")
    print("\nOverall comparison file:")
    print("- model_comparison_summary.csv")
    
    print(f"\nProcessed configurations:")
    for model, count in all_validation_results.items():
        print(f"  {model}: {count} configurations")
    
    print("\nTo use these results:")
    print("1. Check validation results for model selection")
    print("2. Use the best configuration for final training")
    print("3. Consider both RMSE mean and std for robustness")

# Main execution
if __name__ == "__main__":
    
    # Specify the models you want to process
    # Set to None to auto-detect all available models in the results directory
    MODEL_NAMES = ['lstm', 'seq2seq', 'seq2seq_attn', 'tcn', 'transformer', 'sarima']
    
    # You can also auto-detect by setting MODEL_NAMES = None
    # MODEL_NAMES = None
    
    # Process all models
    process_all_models(MODEL_NAMES, results_dir='results_old_val')
    
    # Or process a single model
    # consolidate_single_model('lstm', 'results_old_val')