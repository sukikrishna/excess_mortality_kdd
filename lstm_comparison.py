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
SEEDS_SHARED = [42, 123, 456, 789, 321]  # Seeds for shared randomness experiment
SEEDS_LAYER1 = [42, 123, 456, 789, 321]  # Seeds for layer 1 in independent experiment
SEEDS_LAYER2 = [100, 200, 300, 400, 500]  # Seeds for layer 2 in independent experiment
TRIALS_PER_CONFIG = 50  # Multiple trials to assess variability
RESULTS_DIR = 'lstm_seeding_experiment_results_50'

# Fixed hyperparameters for the experiment
LOOKBACK = 9
BATCH_SIZE = 8
EPOCHS = 100
DROPOUT_RATE = 0.2  # Add dropout to see effect of randomness

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

def create_train_test_split(df, train_end='2020-01-01'):
    """Create train/test splits"""
    train = df[df['Month'] < train_end]
    test = df[df['Month'] >= train_end]
    return train, test

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

def run_lstm_shared_seeds(train, test, lookback, batch_size, epochs, seed, trial_id):
    """Run LSTM with shared seed for both layers"""
    # Set global seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    X_train, y_train = create_dataset(train, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    # Both layers use the same seed
    model = Sequential([
        LSTM(units=NUM_US_STATES, 
             input_shape=(lookback, 1), 
             return_sequences=True,
             dropout=DROPOUT_RATE,
             recurrent_dropout=DROPOUT_RATE,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             name=f'lstm1_shared_seed_{seed}_trial_{trial_id}'),
        LSTM(units=NUM_US_STATES, 
             activation='relu',
             dropout=DROPOUT_RATE,
             recurrent_dropout=DROPOUT_RATE,
             kernel_initializer='glorot_uniform',
             recurrent_initializer='orthogonal',
             name=f'lstm2_shared_seed_{seed}_trial_{trial_id}'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                       verbose=0, validation_split=0.1)
    
    # Generate predictions
    train_preds = []
    for i in range(lookback, len(train)):
        input_seq = train[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    # Test predictions (autoregressive)
    current_input = train[-lookback:].reshape((1, lookback, 1))
    test_preds = []
    for _ in range(len(test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return (train[lookback:], np.array(train_preds), 
            test, np.array(test_preds), history.history['loss'][-1])

def run_lstm_independent_seeds(train, test, lookback, batch_size, epochs, 
                              seed1, seed2, trial_id):
    """Run LSTM with independent seeds for each layer"""
    # Set global seed for reproducibility of other operations
    np.random.seed(seed1)  # Use seed1 as base
    tf.random.set_seed(seed1)
    
    X_train, y_train = create_dataset(train, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
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
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                       verbose=0, validation_split=0.1)
    
    # Generate predictions (same as shared seeds)
    train_preds = []
    for i in range(lookback, len(train)):
        input_seq = train[i-lookback:i].reshape((1, lookback, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    current_input = train[-lookback:].reshape((1, lookback, 1))
    test_preds = []
    for _ in range(len(test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return (train[lookback:], np.array(train_preds), 
            test, np.array(test_preds), history.history['loss'][-1])

def run_experiment():
    """Run the complete seeding experiment"""
    print("="*80)
    print("LSTM SEEDING STRATEGY EXPERIMENT")
    print("="*80)
    print(f"Testing shared vs. independent layer seeds")
    print(f"Number of US states (units): {NUM_US_STATES}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Trials per configuration: {TRIALS_PER_CONFIG}")
    
    # Load data
    print("\nLoading data...")
    data = load_and_preprocess_data()
    train_data, test_data = create_train_test_split(data)
    
    train_series = train_data['Deaths'].values
    test_series = test_data['Deaths'].values
    
    print(f"Train samples: {len(train_series)}")
    print(f"Test samples: {len(test_series)}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Storage for results
    all_results = []
    
    # Experiment 1: Shared seeds
    print("\n" + "="*50)
    print("EXPERIMENT 1: SHARED SEEDS")
    print("="*50)
    
    for i, seed in enumerate(SEEDS_SHARED):
        print(f"\nShared seed {i+1}/{len(SEEDS_SHARED)}: {seed}")
        
        for trial in range(TRIALS_PER_CONFIG):
            print(f"  Trial {trial+1}/{TRIALS_PER_CONFIG}")
            
            try:
                y_train_true, y_train_pred, y_test_true, y_test_pred, final_loss = \
                    run_lstm_shared_seeds(train_series, test_series, LOOKBACK, 
                                        BATCH_SIZE, EPOCHS, seed, trial)
                
                # Calculate metrics
                train_metrics = evaluate_metrics(y_train_true, y_train_pred)
                test_metrics = evaluate_metrics(y_test_true, y_test_pred)
                
                # Store results
                result = {
                    'strategy': 'shared',
                    'seed_config': f'shared_{seed}',
                    'seed1': seed,
                    'seed2': seed,
                    'trial': trial,
                    'final_loss': final_loss,
                    'train_rmse': train_metrics['RMSE'],
                    'train_mae': train_metrics['MAE'],
                    'train_mape': train_metrics['MAPE'],
                    'test_rmse': test_metrics['RMSE'],
                    'test_mae': test_metrics['MAE'],
                    'test_mape': test_metrics['MAPE']
                }
                all_results.append(result)
                
                # Save individual predictions
                pred_df = pd.DataFrame({
                    'train_true': y_train_true,
                    'train_pred': y_train_pred,
                    'test_true': y_test_true,
                    'test_pred': y_test_pred
                })
                pred_df.to_csv(os.path.join(RESULTS_DIR, 
                    f'shared_seed_{seed}_trial_{trial}_predictions.csv'), index=False)
                
            except Exception as e:
                print(f"    Error in trial {trial}: {e}")
                continue
    
    # Experiment 2: Independent seeds
    print("\n" + "="*50)
    print("EXPERIMENT 2: INDEPENDENT SEEDS")
    print("="*50)
    
    for i, (seed1, seed2) in enumerate(zip(SEEDS_LAYER1, SEEDS_LAYER2)):
        print(f"\nIndependent seeds {i+1}/{len(SEEDS_LAYER1)}: Layer1={seed1}, Layer2={seed2}")
        
        for trial in range(TRIALS_PER_CONFIG):
            print(f"  Trial {trial+1}/{TRIALS_PER_CONFIG}")
            
            try:
                y_train_true, y_train_pred, y_test_true, y_test_pred, final_loss = \
                    run_lstm_independent_seeds(train_series, test_series, LOOKBACK, 
                                             BATCH_SIZE, EPOCHS, seed1, seed2, trial)
                
                # Calculate metrics
                train_metrics = evaluate_metrics(y_train_true, y_train_pred)
                test_metrics = evaluate_metrics(y_test_true, y_test_pred)
                
                # Store results
                result = {
                    'strategy': 'independent',
                    'seed_config': f'independent_{seed1}_{seed2}',
                    'seed1': seed1,
                    'seed2': seed2,
                    'trial': trial,
                    'final_loss': final_loss,
                    'train_rmse': train_metrics['RMSE'],
                    'train_mae': train_metrics['MAE'],
                    'train_mape': train_metrics['MAPE'],
                    'test_rmse': test_metrics['RMSE'],
                    'test_mae': test_metrics['MAE'],
                    'test_mape': test_metrics['MAPE']
                }
                all_results.append(result)
                
                # Save individual predictions
                pred_df = pd.DataFrame({
                    'train_true': y_train_true,
                    'train_pred': y_train_pred,
                    'test_true': y_test_true,
                    'test_pred': y_test_pred
                })
                pred_df.to_csv(os.path.join(RESULTS_DIR, 
                    f'independent_seeds_{seed1}_{seed2}_trial_{trial}_predictions.csv'), index=False)
                
            except Exception as e:
                print(f"    Error in trial {trial}: {e}")
                continue
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'all_experiment_results.csv'), index=False)
    
    # Generate summary statistics and visualizations
    generate_analysis_and_plots(results_df)
    
    print(f"\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)

def generate_analysis_and_plots(results_df):
    """Generate summary statistics and visualizations"""
    print("\n" + "="*50)
    print("GENERATING ANALYSIS AND PLOTS")
    print("="*50)
    
    # Summary statistics by strategy
    summary_stats = results_df.groupby('strategy').agg({
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'test_mae': ['mean', 'std', 'min', 'max'],
        'test_mape': ['mean', 'std', 'min', 'max'],
        'final_loss': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\nSUMMARY STATISTICS BY STRATEGY:")
    print(summary_stats)
    
    # Save summary statistics
    summary_stats.to_csv(os.path.join(RESULTS_DIR, 'summary_statistics.csv'))
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LSTM Seeding Strategy Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Test RMSE comparison
    sns.boxplot(data=results_df, x='strategy', y='test_rmse', ax=axes[0,0])
    axes[0,0].set_title('Test RMSE by Seeding Strategy')
    axes[0,0].set_ylabel('RMSE')
    
    # Plot 2: Test MAE comparison
    sns.boxplot(data=results_df, x='strategy', y='test_mae', ax=axes[0,1])
    axes[0,1].set_title('Test MAE by Seeding Strategy')
    axes[0,1].set_ylabel('MAE')
    
    # Plot 3: Training loss comparison
    sns.boxplot(data=results_df, x='strategy', y='final_loss', ax=axes[1,0])
    axes[1,0].set_title('Final Training Loss by Seeding Strategy')
    axes[1,0].set_ylabel('Loss')
    
    # Plot 4: Variability comparison (coefficient of variation)
    cv_data = results_df.groupby('strategy').agg({
        'test_rmse': lambda x: x.std() / x.mean(),
        'test_mae': lambda x: x.std() / x.mean(),
        'final_loss': lambda x: x.std() / x.mean()
    }).reset_index()
    
    cv_melted = cv_data.melt(id_vars=['strategy'], 
                            value_vars=['test_rmse', 'test_mae', 'final_loss'],
                            var_name='metric', value_name='coefficient_of_variation')
    
    sns.barplot(data=cv_melted, x='metric', y='coefficient_of_variation', 
                hue='strategy', ax=axes[1,1])
    axes[1,1].set_title('Coefficient of Variation by Metric')
    axes[1,1].set_ylabel('CV (std/mean)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'seeding_strategy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed analysis
    print("\nDETAILED ANALYSIS:")
    
    # Statistical significance test
    from scipy.stats import mannwhitneyu
    
    shared_rmse = results_df[results_df['strategy'] == 'shared']['test_rmse']
    independent_rmse = results_df[results_df['strategy'] == 'independent']['test_rmse']
    
    stat, p_value = mannwhitneyu(shared_rmse, independent_rmse, alternative='two-sided')
    print(f"\nMann-Whitney U test for Test RMSE:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Variability analysis
    shared_cv = shared_rmse.std() / shared_rmse.mean()
    independent_cv = independent_rmse.std() / independent_rmse.mean()
    
    print(f"\nVariability Analysis (Test RMSE):")
    print(f"  Shared seeding CV: {shared_cv:.4f}")
    print(f"  Independent seeding CV: {independent_cv:.4f}")
    print(f"  Difference: {abs(shared_cv - independent_cv):.4f}")
    
    # Performance analysis
    print(f"\nPerformance Analysis (Test RMSE):")
    print(f"  Shared seeding mean: {shared_rmse.mean():.4f} ± {shared_rmse.std():.4f}")
    print(f"  Independent seeding mean: {independent_rmse.mean():.4f} ± {independent_rmse.std():.4f}")
    
    # Create seed-specific analysis
    seed_analysis = results_df.groupby(['strategy', 'seed_config']).agg({
        'test_rmse': ['mean', 'std'],
        'test_mae': ['mean', 'std']
    }).round(4)
    
    print(f"\nSEED-SPECIFIC ANALYSIS:")
    print(seed_analysis)
    
    seed_analysis.to_csv(os.path.join(RESULTS_DIR, 'seed_specific_analysis.csv'))

if __name__ == "__main__":
    run_experiment()