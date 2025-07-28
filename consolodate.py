import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_hyperparams_from_path(path, model_name):
    """Extract hyperparameters from the folder path based on model type"""
    
    if model_name == 'seq2seq':
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
    else:
        # Original pattern for other models: lookback_12_bs_32_epochs_50
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
        # Extract mean and std values
        metrics = {}
        for col in df.columns:
            if col in ['RMSE', 'MAE', 'MAPE']:  # Only keep the main metrics
                metrics[f'{col}_mean'] = df.loc['mean', col]
                metrics[f'{col}_std'] = df.loc['std', col]
        return metrics
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def consolidate_single_model(model_name, results_dir='results'):
    """Consolidate hyperparameter results for a single model and save in model folder"""
    model_path = Path(results_dir) / model_name / 'fixed_seed_variability'
    
    if not model_path.exists():
        print(f"Path {model_path} does not exist")
        return
    
    train_results = []
    test_results = []
    
    print(f"Processing {model_name}...")
    
    # Iterate through all hyperparameter combination folders
    for config_folder in model_path.iterdir():
        if config_folder.is_dir():
            hyperparams = extract_hyperparams_from_path(config_folder.name, model_name)
            if hyperparams is None:
                print(f"  Warning: Could not parse hyperparameters from {config_folder.name}")
                continue
            
            # Load train metrics
            train_file = config_folder / 'summary_metrics_train.csv'
            if train_file.exists():
                train_metrics = load_summary_metrics(train_file)
                if train_metrics:
                    result_row = {**hyperparams, **train_metrics}
                    train_results.append(result_row)
            
            # Load test metrics
            test_file = config_folder / 'summary_metrics_test.csv'
            if test_file.exists():
                test_metrics = load_summary_metrics(test_file)
                if test_metrics:
                    result_row = {**hyperparams, **test_metrics}
                    test_results.append(result_row)
    
    # Convert to DataFrames and sort by RMSE_mean (best first)
    if train_results:
        train_df = pd.DataFrame(train_results)
        train_df = train_df.sort_values('RMSE_mean')
        
        # Save in model folder
        train_output_path = Path(results_dir) / model_name / 'hyperparameter_results_train.csv'
        train_df.to_csv(train_output_path, index=False)
        print(f"✓ Saved train results: {train_output_path}")
        
        # Print best config based on model type
        if model_name == 'seq2seq':
            print(f"  Best train config: lookback={train_df.iloc[0]['lookback']}, "
                  f"bs={train_df.iloc[0]['batch_size']}, epochs={train_df.iloc[0]['epochs']}, "
                  f"enc={train_df.iloc[0]['encoder_units']}, dec={train_df.iloc[0]['decoder_units']}, "
                  f"att={train_df.iloc[0]['attention']}, RMSE={train_df.iloc[0]['RMSE_mean']:.2f}")
        else:
            print(f"  Best train config: lookback={train_df.iloc[0]['lookback']}, "
                  f"bs={train_df.iloc[0]['batch_size']}, epochs={train_df.iloc[0]['epochs']}, "
                  f"RMSE={train_df.iloc[0]['RMSE_mean']:.2f}")
    
    if test_results:
        test_df = pd.DataFrame(test_results)
        test_df = test_df.sort_values('RMSE_mean')
        
        # Save in model folder
        test_output_path = Path(results_dir) / model_name / 'hyperparameter_results_test.csv'
        test_df.to_csv(test_output_path, index=False)
        print(f"✓ Saved test results: {test_output_path}")
        
        # Print best config based on model type
        if model_name == 'seq2seq':
            print(f"  Best test config: lookback={test_df.iloc[0]['lookback']}, "
                  f"bs={test_df.iloc[0]['batch_size']}, epochs={test_df.iloc[0]['epochs']}, "
                  f"enc={test_df.iloc[0]['encoder_units']}, dec={test_df.iloc[0]['decoder_units']}, "
                  f"att={test_df.iloc[0]['attention']}, RMSE={test_df.iloc[0]['RMSE_mean']:.2f}")
        else:
            print(f"  Best test config: lookback={test_df.iloc[0]['lookback']}, "
                  f"bs={test_df.iloc[0]['batch_size']}, epochs={test_df.iloc[0]['epochs']}, "
                  f"RMSE={test_df.iloc[0]['RMSE_mean']:.2f}")
    
    print()

def process_all_models(model_names, results_dir='results_updated'):
    """Process all specified models"""
    print("="*60)
    print("CONSOLIDATING HYPERPARAMETER RESULTS")
    print("="*60)
    
    for model_name in model_names:
        consolidate_single_model(model_name, results_dir)
    
    print("="*60)
    print("CONSOLIDATION COMPLETE!")
    print("="*60)
    print("Files created in each model folder:")
    print("- hyperparameter_results_train.csv")
    print("- hyperparameter_results_test.csv")
    print("\nTo find optimal hyperparameters:")
    print("1. Open the CSV file for your model")
    print("2. Sort by RMSE_mean (ascending) for best performance")
    print("3. Or sort by MAE_mean or MAPE_mean as needed")
    print("\nNote: seq2seq model includes additional hyperparameters:")
    print("- encoder_units: number of encoder units")
    print("- decoder_units: number of decoder units") 
    print("- attention: whether attention mechanism is used")

# Main execution
if __name__ == "__main__":
    
    MODEL_NAMES = ['lstm', 'seq2seq', 'seq2seq_attn', 'tcn', 'transformer']
    
    # Process all models
    process_all_models(MODEL_NAMES)

    # Process single model
    # consolidate_single_model('seq2seq')