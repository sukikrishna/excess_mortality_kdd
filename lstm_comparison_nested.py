import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Configuration
NUM_US_STATES = 50  # Variable for flexibility as you suggested
SEEDS = [42, 123, 456, 789, 321, 100, 200, 300, 400, 500]  # All seeds to test
TRIALS_PER_CONFIG = 30  # Multiple trials to assess variability
RESULTS_DIR = 'lstm_seeding_experiment_results_nested'

# Fixed hyperparameters for the experiment
LOOKBACK = 6  # Reduced from 12 given small training set (48 samples)
BATCH_SIZE = 8  # Reduced from 16 
EPOCHS = 50  # Reduced from 100 for faster testing
DROPOUT_RATE = 0.1  # Reduced dropout

def load_and_preprocess_data():
    """Load the preprocessed data - simplified version"""
    # Using your existing data loading logic
    df = pd.read_excel('data_updated/state_month_overdose_2015_2023.xlsx')
    
    if 'Sum of Deaths' in df.columns:
        df = df.rename(columns={'Sum of Deaths': 'Deaths'})
    
    if 'Year_Code' in df.columns and 'Month_Code' in df.columns:
        df['Month_Code'] = pd.to_numeric(df['Month_Code'], errors='coerce')
        df['Year_Code'] = pd.to_numeric(df['Year_Code'], errors='coerce')
        date_df = pd.DataFrame({
            'year': df['Year_Code'],
            'month': df['Month_Code'], 
            'day': 1
        })
        df['Month'] = pd.to_datetime(date_df)
    elif 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    
    if df['Deaths'].dtype == 'object':
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else float(x))
    else:
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
    
    df = df.dropna(subset=['Month', 'Deaths'])
    df = df[['Month', 'Deaths']].copy()
    df = df.sort_values('Month').reset_index(drop=True)
    
    return df

def create_train_val_test_split(df, train_end='2019-01-01', val_end='2020-01-01'):
    """Create proper train/validation/test splits for hyperparameter optimization"""
    train = df[df['Month'] < train_end]  # Up to 2018
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]  # 2019
    test = df[df['Month'] >= val_end]  # 2020 onwards (not used in this experiment)
    
    print(f"Train samples: {len(train)} ({train['Month'].min()} to {train['Month'].max()})")
    print(f"Validation samples: {len(validation)} ({validation['Month'].min()} to {validation['Month'].max()})")
    print(f"Test samples: {len(test)} ({test['Month'].min()} to {test['Month'].max()}) - NOT USED")
    
    return train, validation, test

def create_dataset(series, look_back):
    """Create dataset for supervised learning"""
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def run_lstm_with_layer_seeds(train, validation, lookback, batch_size, epochs, 
                             seed1, seed2, trial_id, global_seed_strategy=False):
    """
    Run LSTM with specified seeds for each layer
    
    Args:
        train: Training data
        validation: Validation data  
        lookback: Number of time steps to look back
        batch_size: Batch size for training
        epochs: Number of training epochs
        seed1: Seed for first LSTM layer
        seed2: Seed for second LSTM layer
        trial_id: Trial identifier
        global_seed_strategy: If True, set global seed instead of layer-specific seeds
    """
    
    # Debug: Check data lengths
    print(f"      Debug: Train length={len(train)}, Val length={len(validation)}, Lookback={lookback}")
    
    # Check if we have enough data
    if len(train) <= lookback:
        raise ValueError(f"Training data length ({len(train)}) must be > lookback ({lookback})")
    if len(validation) == 0:
        raise ValueError("Validation data is empty")
    
    if global_seed_strategy:
        # Global seed strategy - set global seed and let layers inherit
        np.random.seed(seed1)  # Use seed1 as the global seed
        tf.random.set_seed(seed1)
        
        model = Sequential([
            LSTM(units=NUM_US_STATES, 
                 input_shape=(lookback, 1), 
                 return_sequences=True,
                 dropout=DROPOUT_RATE,
                 recurrent_dropout=DROPOUT_RATE,
                 name=f'lstm1_global_seed_{seed1}_trial_{trial_id}'),
            LSTM(units=NUM_US_STATES, 
                 activation='relu',
                 dropout=DROPOUT_RATE,
                 recurrent_dropout=DROPOUT_RATE,
                 name=f'lstm2_global_seed_{seed1}_trial_{trial_id}'),
            Dense(1)
        ])
    else:
        # Layer-specific seed strategy
        np.random.seed(seed1)  # Set base seed
        tf.random.set_seed(seed1)
        
        # Create custom initializers with different seeds
        kernel_init_1 = tf.keras.initializers.GlorotUniform(seed=seed1)
        recurrent_init_1 = tf.keras.initializers.Orthogonal(seed=seed1)
        kernel_init_2 = tf.keras.initializers.GlorotUniform(seed=seed2)
        recurrent_init_2 = tf.keras.initializers.Orthogonal(seed=seed2)
        
        model = Sequential([
            LSTM(units=NUM_US_STATES, 
                 input_shape=(lookback, 1), 
                 return_sequences=True,
                 dropout=DROPOUT_RATE,
                 recurrent_dropout=DROPOUT_RATE,
                 kernel_initializer=kernel_init_1,
                 recurrent_initializer=recurrent_init_1,
                 name=f'lstm1_seed_{seed1}_trial_{trial_id}'),
            LSTM(units=NUM_US_STATES, 
                 activation='relu',
                 dropout=DROPOUT_RATE,
                 recurrent_dropout=DROPOUT_RATE,
                 kernel_initializer=kernel_init_2,
                 recurrent_initializer=recurrent_init_2,
                 name=f'lstm2_seed_{seed2}_trial_{trial_id}'),
            Dense(1)
        ])
    
    # Prepare training data
    X_train, y_train = create_dataset(train, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    print(f"      Debug: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    
    # Check if we have enough training samples
    if len(X_train) == 0:
        raise ValueError(f"No training samples created with lookback={lookback} and train_length={len(train)}")
    
    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=min(batch_size, len(X_train)), 
                       verbose=0, validation_split=0.1)
    
    # Generate training predictions (in-sample)
    train_preds = []
    for i in range(lookback, len(train)):
        input_seq = train[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    print(f"      Debug: Generated {len(train_preds)} training predictions")
    
    # Generate validation predictions (out-of-sample, autoregressive)
    current_input = train[-lookback:].reshape((1, lookback, 1))
    val_preds = []
    for _ in range(len(validation)):
        pred = model.predict(current_input, verbose=0)[0][0]
        val_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    print(f"      Debug: Generated {len(val_preds)} validation predictions")
    
    return (train[lookback:], np.array(train_preds), 
            validation, np.array(val_preds), history.history['loss'][-1])

def run_comprehensive_seeding_experiment():
    """Run comprehensive seeding experiment comparing all seed combinations"""
    print("="*80)
    print("COMPREHENSIVE LSTM SEEDING STRATEGY EXPERIMENT")
    print("="*80)
    print(f"Testing all combinations of layer seeds")
    print(f"Seeds to test: {SEEDS}")
    print(f"Total seed combinations: {len(SEEDS)} × {len(SEEDS)} = {len(SEEDS)**2}")
    print(f"Number of US states (units): {NUM_US_STATES}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Trials per configuration: {TRIALS_PER_CONFIG}")
    print(f"Evaluation on: 2019 validation data (training on data through 2018)")
    
    # Load data
    print("\nLoading data...")
    data = load_and_preprocess_data()
    train_data, validation_data, test_data = create_train_val_test_split(data)
    
    train_series = train_data['Deaths'].values
    validation_series = validation_data['Deaths'].values
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Storage for results
    all_results = []
    
    # Calculate total configurations
    total_configs = len(SEEDS) * len(SEEDS)
    config_count = 0
    
    print(f"\n" + "="*50)
    print("RUNNING ALL SEED COMBINATIONS")
    print("="*50)
    
    # Nested loop over all seed combinations
    for seed1 in SEEDS:
        for seed2 in SEEDS:
            config_count += 1
            
            # Determine strategy type
            if seed1 == seed2:
                strategy = 'same_seeds'
                strategy_detail = f'both_layers_seed_{seed1}'
            else:
                strategy = 'different_seeds'
                strategy_detail = f'layer1_{seed1}_layer2_{seed2}'
            
            print(f"\nConfiguration {config_count}/{total_configs}: {strategy_detail}")
            
            # Test both global seed strategy and layer-specific strategy when seeds are the same
            strategies_to_test = []
            if seed1 == seed2:
                strategies_to_test.append(('global_seed', True))
                strategies_to_test.append(('layer_specific_same', False))
            else:
                strategies_to_test.append(('layer_specific_different', False))
            
            for strategy_name, global_seed_flag in strategies_to_test:
                print(f"  Testing {strategy_name}")
                
                for trial in range(TRIALS_PER_CONFIG):
                    print(f"    Trial {trial+1}/{TRIALS_PER_CONFIG}")
                    
                    try:
                        y_train_true, y_train_pred, y_val_true, y_val_pred, final_loss = \
                            run_lstm_with_layer_seeds(train_series, validation_series, 
                                                    LOOKBACK, BATCH_SIZE, EPOCHS, 
                                                    seed1, seed2, trial, global_seed_flag)
                        
                        # Calculate metrics
                        train_metrics = evaluate_metrics(y_train_true, y_train_pred)
                        val_metrics = evaluate_metrics(y_val_true, y_val_pred)
                        
                        # Store results
                        result = {
                            'seed1': seed1,
                            'seed2': seed2,
                            'strategy': strategy,
                            'strategy_detail': strategy_detail,
                            'seeding_approach': strategy_name,
                            'global_seed_flag': global_seed_flag,
                            'trial': trial,
                            'final_loss': final_loss,
                            # Training metrics (in-sample)
                            'train_rmse': train_metrics['RMSE'],
                            'train_mae': train_metrics['MAE'],
                            'train_mape': train_metrics['MAPE'],
                            # Validation metrics (out-of-sample) - PRIMARY EVALUATION
                            'val_rmse': val_metrics['RMSE'],
                            'val_mae': val_metrics['MAE'],
                            'val_mape': val_metrics['MAPE'],
                            'is_diagonal': seed1 == seed2  # Flag for same seeds
                        }
                        all_results.append(result)
                        
                        # Save individual predictions for detailed analysis (separate files for train/val)
                        train_pred_df = pd.DataFrame({
                            'true': y_train_true,
                            'pred': y_train_pred
                        })
                        val_pred_df = pd.DataFrame({
                            'true': y_val_true,
                            'pred': y_val_pred
                        })
                        
                        train_filename = f'train_pred_{strategy_name}_s1_{seed1}_s2_{seed2}_trial_{trial}.csv'
                        val_filename = f'val_pred_{strategy_name}_s1_{seed1}_s2_{seed2}_trial_{trial}.csv'
                        
                        train_pred_df.to_csv(os.path.join(RESULTS_DIR, train_filename), index=False)
                        val_pred_df.to_csv(os.path.join(RESULTS_DIR, val_filename), index=False)
                        
                    except Exception as e:
                        print(f"      Error in trial {trial}: {e}")
                        continue
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'comprehensive_seeding_results.csv'), index=False)
    
    # Generate analysis
    generate_comprehensive_analysis(results_df)
    
    print(f"\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT COMPLETED")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)

def generate_comprehensive_analysis(results_df):
    """Generate comprehensive analysis and visualizations"""
    print("\n" + "="*50)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*50)
    
    # 1. Summary statistics by strategy
    summary_stats = results_df.groupby(['strategy', 'seeding_approach']).agg({
        'val_rmse': ['mean', 'std', 'min', 'max', 'count'],
        'val_mae': ['mean', 'std', 'min', 'max'],
        'val_mape': ['mean', 'std', 'min', 'max'],
        'final_loss': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\nSUMMARY STATISTICS BY STRATEGY:")
    print(summary_stats)
    summary_stats.to_csv(os.path.join(RESULTS_DIR, 'summary_statistics_by_strategy.csv'))
    
    # 2. Create seeding matrix analysis (for same seeds vs different seeds)
    diagonal_results = results_df[results_df['is_diagonal'] == True]
    off_diagonal_results = results_df[results_df['is_diagonal'] == False]
    
    print(f"\nDIAGONAL vs OFF-DIAGONAL ANALYSIS:")
    print(f"Same seeds (diagonal): {len(diagonal_results)} configurations")
    print(f"Different seeds (off-diagonal): {len(off_diagonal_results)} configurations")
    
    diagonal_stats = diagonal_results.groupby('seeding_approach')['val_rmse'].agg(['mean', 'std']).round(4)
    off_diagonal_stats = off_diagonal_results['val_rmse'].agg(['mean', 'std']).round(4)
    
    print(f"\nDiagonal (same seeds) validation RMSE by approach:")
    print(diagonal_stats)
    print(f"\nOff-diagonal (different seeds) validation RMSE:")
    print(f"  Mean: {off_diagonal_stats['mean']:.4f}")
    print(f"  Std:  {off_diagonal_stats['std']:.4f}")
    
    # 3. Create heat map of seed combinations
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive LSTM Seeding Analysis', fontsize=16, fontweight='bold')
    
    # Heat map of validation RMSE for all seed combinations
    mean_rmse_matrix = results_df[results_df['seeding_approach'] == 'layer_specific_different'].groupby(['seed1', 'seed2'])['val_rmse'].mean().unstack()
    
    # Add diagonal values from global_seed approach
    diagonal_values = results_df[results_df['seeding_approach'] == 'global_seed'].groupby('seed1')['val_rmse'].mean()
    for seed in diagonal_values.index:
        if seed in mean_rmse_matrix.index and seed in mean_rmse_matrix.columns:
            mean_rmse_matrix.loc[seed, seed] = diagonal_values[seed]
    
    sns.heatmap(mean_rmse_matrix, annot=True, fmt='.3f', cmap='viridis_r', ax=axes[0,0])
    axes[0,0].set_title('Mean Validation RMSE by Seed Combination\n(Diagonal: Global Seed, Off-diagonal: Layer-specific)')
    axes[0,0].set_xlabel('Layer 2 Seed')
    axes[0,0].set_ylabel('Layer 1 Seed')
    
    # Box plot comparing strategies
    strategy_comparison_data = []
    for _, row in results_df.iterrows():
        if row['seeding_approach'] == 'global_seed':
            strategy_comparison_data.append({'Strategy': 'Global Seed (Same)', 'Validation RMSE': row['val_rmse']})
        elif row['seeding_approach'] == 'layer_specific_same':
            strategy_comparison_data.append({'Strategy': 'Layer-Specific (Same)', 'Validation RMSE': row['val_rmse']})
        else:
            strategy_comparison_data.append({'Strategy': 'Layer-Specific (Different)', 'Validation RMSE': row['val_rmse']})
    
    strategy_df = pd.DataFrame(strategy_comparison_data)
    sns.boxplot(data=strategy_df, x='Strategy', y='Validation RMSE', ax=axes[0,1])
    axes[0,1].set_title('Validation RMSE by Seeding Strategy')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Variance analysis
    variance_by_seed_combo = results_df.groupby(['seed1', 'seed2', 'seeding_approach'])['val_rmse'].std().reset_index()
    variance_by_seed_combo['combination_type'] = variance_by_seed_combo.apply(
        lambda x: 'Same Seeds' if x['seed1'] == x['seed2'] else 'Different Seeds', axis=1)
    
    sns.boxplot(data=variance_by_seed_combo, x='combination_type', y='val_rmse', 
                hue='seeding_approach', ax=axes[1,0])
    axes[1,0].set_title('Variability (Std Dev) of Validation RMSE')
    axes[1,0].set_ylabel('Standard Deviation of RMSE')
    
    # Performance distribution
    axes[1,1].hist(diagonal_results[diagonal_results['seeding_approach'] == 'global_seed']['val_rmse'], 
                   alpha=0.7, label='Global Seed (Same)', bins=20)
    axes[1,1].hist(diagonal_results[diagonal_results['seeding_approach'] == 'layer_specific_same']['val_rmse'], 
                   alpha=0.7, label='Layer-Specific (Same)', bins=20)
    axes[1,1].hist(off_diagonal_results['val_rmse'], 
                   alpha=0.7, label='Layer-Specific (Different)', bins=20)
    axes[1,1].set_title('Distribution of Validation RMSE')
    axes[1,1].set_xlabel('Validation RMSE')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_seeding_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Statistical tests
    from scipy.stats import mannwhitneyu, kruskal
    
    print(f"\nSTATISTICAL ANALYSIS:")
    
    # Test 1: Global seed vs Layer-specific (same seeds)
    global_rmse = diagonal_results[diagonal_results['seeding_approach'] == 'global_seed']['val_rmse']
    layer_same_rmse = diagonal_results[diagonal_results['seeding_approach'] == 'layer_specific_same']['val_rmse']
    
    if len(global_rmse) > 0 and len(layer_same_rmse) > 0:
        stat, p_val = mannwhitneyu(global_rmse, layer_same_rmse, alternative='two-sided')
        print(f"\nGlobal Seed vs Layer-Specific (Same Seeds):")
        print(f"  Global seed RMSE: {global_rmse.mean():.4f} ± {global_rmse.std():.4f}")
        print(f"  Layer-specific RMSE: {layer_same_rmse.mean():.4f} ± {layer_same_rmse.std():.4f}")
        print(f"  Mann-Whitney U statistic: {stat:.4f}")
        print(f"  P-value: {p_val:.6f}")
        print(f"  Significant difference: {'Yes' if p_val < 0.05 else 'No'}")
    
    # Test 2: Same seeds vs Different seeds
    same_seeds_rmse = diagonal_results['val_rmse']
    diff_seeds_rmse = off_diagonal_results['val_rmse']
    
    stat2, p_val2 = mannwhitneyu(same_seeds_rmse, diff_seeds_rmse, alternative='two-sided')
    print(f"\nSame Seeds vs Different Seeds:")
    print(f"  Same seeds RMSE: {same_seeds_rmse.mean():.4f} ± {same_seeds_rmse.std():.4f}")
    print(f"  Different seeds RMSE: {diff_seeds_rmse.mean():.4f} ± {diff_seeds_rmse.std():.4f}")
    print(f"  Mann-Whitney U statistic: {stat2:.4f}")
    print(f"  P-value: {p_val2:.6f}")
    print(f"  Significant difference: {'Yes' if p_val2 < 0.05 else 'No'}")
    
    # 5. Best configurations analysis
    print(f"\nBEST CONFIGURATIONS:")
    
    best_configs = results_df.groupby(['seed1', 'seed2', 'seeding_approach'])['val_rmse'].mean().reset_index()
    best_configs = best_configs.sort_values('val_rmse').head(10)
    
    print(f"\nTop 10 configurations by validation RMSE:")
    for idx, row in best_configs.iterrows():
        print(f"  {row['seeding_approach']}: seed1={row['seed1']}, seed2={row['seed2']}, RMSE={row['val_rmse']:.4f}")
    
    best_configs.to_csv(os.path.join(RESULTS_DIR, 'best_configurations.csv'), index=False)
    
    # 6. Detailed analysis by seed
    seed_analysis = results_df.groupby(['seed1', 'seed2', 'seeding_approach']).agg({
        'val_rmse': ['mean', 'std', 'min', 'max'],
        'val_mae': ['mean', 'std'],
        'final_loss': ['mean', 'std']
    }).round(4)
    
    seed_analysis.to_csv(os.path.join(RESULTS_DIR, 'detailed_seed_analysis.csv'))
    
    print(f"\nAnalysis complete! Check the following files for detailed results:")
    print(f"  - comprehensive_seeding_results.csv: Raw results")
    print(f"  - summary_statistics_by_strategy.csv: Strategy summaries")
    print(f"  - best_configurations.csv: Top performing configurations")
    print(f"  - detailed_seed_analysis.csv: Detailed by-seed analysis")
    print(f"  - comprehensive_seeding_analysis.png: Visualizations")

if __name__ == "__main__":
    run_comprehensive_seeding_experiment()