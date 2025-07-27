import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import RepeatVector, Concatenate
from tensorflow.keras.layers import Input, Add, ReLU, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN  # pip install keras-tcn
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------ #
MODEL_TYPE = 'seq2seq_attn'  # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer'
TRIAL_MODE = 'fixed_seed'  # Options: 'fixed_seed', 'multi_seed'
SEEDS = [42] if TRIAL_MODE == 'fixed_seed' else [123, 456, 11, 245, 56712, 23467, 98, 38, 1506, 42]

# Number of US states - can be adjusted for substance-specific or age-specific models
NUM_US_STATES = 50

# Updated hyperparameter grids with expanded ranges
# Expanded lookbacks (1-12 plus larger values)
LOOKBACKS = list(range(1, 13)) + [18, 24]

# Batch sizes (added 48 for full series)
BATCH_SIZES = [8, 16, 32, 48]

# Epochs with wider coverage
EPOCHS_LIST = [50, 100, 150, 200]

# For seq2seq models
ENCODER_UNITS = [64, 128]
DECODER_UNITS = [64, 128]
USE_ATTENTION = [False, True]

# For transformer models
D_MODEL = [64, 128]
N_HEADS = [1, 2, 4]

# For SARIMA models - expanded orders and seasonals
SARIMA_ORDERS = [(1, 1, 1), (2, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 2)]
SARIMA_SEASONAL_ORDERS = [(1, 1, 1, 12), (1, 0, 1, 12), (2, 1, 1, 12)]

# Model-specific trial counts
def get_trials_per_config(model_type):
    """Return appropriate number of trials based on model stability"""
    if model_type == 'sarima':
        return 5  # SARIMA is stable, fewer trials needed
    else:
        return 30  # Neural models need more trials due to randomness

DATA_PATH = 'data_updated/state_month_overdose_2015_2023.xlsx'  # Updated path
RESULTS_DIR = 'results'

# ------------------ HELPER FUNCTIONS ------------------ #
def load_and_preprocess_data():
    """Load the preprocessed data from updated path with proper date handling"""
    df = pd.read_excel(DATA_PATH)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("First few rows:")
    print(df.head())
    
    # Handle different possible column names and structures
    if 'Sum of Deaths' in df.columns:
        df = df.rename(columns={'Sum of Deaths': 'Deaths'})
    
    # Create proper date column from Year_Code and Month_Code if available
    if 'Year_Code' in df.columns and 'Month_Code' in df.columns:
        # Convert Month_Code to integer if it's a string
        df['Month_Code'] = pd.to_numeric(df['Month_Code'], errors='coerce')
        df['Year_Code'] = pd.to_numeric(df['Year_Code'], errors='coerce')
        
        # Create proper datetime with correct column names
        date_df = pd.DataFrame({
            'year': df['Year_Code'],
            'month': df['Month_Code'], 
            'day': 1
        })
        df['Month'] = pd.to_datetime(date_df)
        print("✓ Created Month column from Year_Code and Month_Code")
    
    elif 'Month' in df.columns:
        # Try different approaches to handle the Month column
        try:
            # First, try direct conversion
            df['Month'] = pd.to_datetime(df['Month'])
            print("✓ Direct datetime conversion successful")
        except:
            try:
                # If direct conversion fails, try with specific format
                df['Month'] = pd.to_datetime(df['Month'], format='%m/%d/%Y')
                print("✓ Datetime conversion with format successful")
            except:
                try:
                    # Try converting from Excel serial date
                    df['Month'] = pd.to_datetime(df['Month'], unit='D', origin='1899-12-30')
                    print("✓ Excel serial date conversion successful")
                except:
                    # If all else fails, try parsing as string
                    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
                    print("✓ Coerced datetime conversion (some may be NaT)")
    else:
        print("✗ No suitable date column found!")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Cannot find or create a proper date column")
    
    # Handle Deaths column
    if 'Deaths' not in df.columns:
        print("✗ Deaths column not found!")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Deaths column not found")
    
    # Clean Deaths column
    if df['Deaths'].dtype == 'object':
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else float(x))
    else:
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
    
    # Remove any rows with invalid dates or deaths
    initial_rows = len(df)
    df = df.dropna(subset=['Month', 'Deaths'])
    final_rows = len(df)
    
    if initial_rows != final_rows:
        print(f"⚠ Removed {initial_rows - final_rows} rows with invalid data")
    
    # Keep only Month and Deaths columns
    df = df[['Month', 'Deaths']].copy()
    
    # Sort by date to ensure proper time series order
    df = df.sort_values('Month').reset_index(drop=True)
    
    print(f"✓ Final data shape: {df.shape}")
    print(f"✓ Date range: {df['Month'].min()} to {df['Month'].max()}")
    print(f"✓ Deaths range: {df['Deaths'].min()} to {df['Deaths'].max()}")
    print("✓ First few rows of processed data:")
    print(df.head())
    
    return df

def create_train_val_test_split(df, train_end='2019-01-01', val_end='2020-01-01'):
    """Create train/validation/test splits"""
    train = df[df['Month'] < train_end]
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
    test = df[df['Month'] >= val_end]
    
    print(f"Train samples: {len(train)} ({train['Month'].min()} to {train['Month'].max()})")
    print(f"Validation samples: {len(validation)} ({validation['Month'].min()} to {validation['Month'].max()})")
    print(f"Test samples: {len(test)} ({test['Month'].min()} to {test['Month'].max()})")
    
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
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'PI Width': 0,
        'CI Coverage': 0,
        'PI Overlap': 0
    }

def create_model_comparison_analysis(results_dir, model_type):
    """
    Create analysis comparing mean vs median RMSE rankings to identify outliers
    """
    import glob
    
    # Look for validation metrics files instead of test metrics
    summary_files = glob.glob(os.path.join(results_dir, model_type, '**/summary_metrics_validation.csv'), recursive=True)
    
    comparison_data = []
    for file_path in summary_files:
        config_name = os.path.basename(os.path.dirname(file_path))
        
        try:
            summary_df = pd.read_csv(file_path, index_col=0)
            
            if 'RMSE' in summary_df.columns:
                mean_rmse = summary_df.loc['mean', 'RMSE']
                median_rmse = summary_df.loc['median', 'RMSE']
                std_rmse = summary_df.loc['std', 'RMSE']
                
                # Calculate coefficient of variation to identify high variability
                cv_rmse = std_rmse / mean_rmse if mean_rmse > 0 else np.inf
                
                comparison_data.append({
                    'config': config_name,
                    'mean_rmse': mean_rmse,
                    'median_rmse': median_rmse,
                    'std_rmse': std_rmse,
                    'cv_rmse': cv_rmse,
                    'mean_median_diff': abs(mean_rmse - median_rmse),
                    'mean_median_ratio': mean_rmse / median_rmse if median_rmse > 0 else np.inf
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank by mean and median RMSE
        comparison_df['rank_by_mean'] = comparison_df['mean_rmse'].rank()
        comparison_df['rank_by_median'] = comparison_df['median_rmse'].rank()
        comparison_df['rank_difference'] = abs(comparison_df['rank_by_mean'] - comparison_df['rank_by_median'])
        
        # Sort by median RMSE (more robust to outliers)
        comparison_df = comparison_df.sort_values('median_rmse')
        
        # Save comparison analysis
        output_file = os.path.join(results_dir, f'{model_type}_mean_vs_median_analysis.csv')
        comparison_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"MEAN VS MEDIAN VALIDATION RMSE ANALYSIS FOR {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Analysis saved to: {output_file}")
        
        # Identify configs with high variability (outliers)
        high_variability = comparison_df[comparison_df['cv_rmse'] > comparison_df['cv_rmse'].quantile(0.9)]
        if len(high_variability) > 0:
            print(f"\nConfigurations with high RMSE variability (top 10%):")
            for _, row in high_variability.head(10).iterrows():
                print(f"  {row['config']}: CV = {row['cv_rmse']:.3f}")
        
        # Identify configs where ranking changes significantly between mean and median
        rank_changes = comparison_df[comparison_df['rank_difference'] > 5]
        if len(rank_changes) > 0:
            print(f"\nConfigurations with significant rank changes (>5 positions):")
            for _, row in rank_changes.head(10).iterrows():
                print(f"  {row['config']}: Mean rank = {row['rank_by_mean']:.0f}, Median rank = {row['rank_by_median']:.0f}")
        
        # Show top 10 by median (more robust)
        print(f"\nTop 10 configurations by MEDIAN VALIDATION RMSE:")
        top_median = comparison_df.head(10)
        for _, row in top_median.iterrows():
            print(f"  {row['config']}: Median = {row['median_rmse']:.4f}, Mean = {row['mean_rmse']:.4f}")
        
        return comparison_df
    
    return None

# ------------------ MODEL IMPLEMENTATIONS ------------------ #

def run_lstm(train, test, look_back, batch_size, epochs, seed):
    """Run LSTM model with state-based units and global seeding"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    X_train, y_train = create_dataset(train, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    
    model = Sequential([
        LSTM(units=NUM_US_STATES, input_shape=(look_back, 1), return_sequences=True),
        LSTM(units=NUM_US_STATES, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Generate training predictions
    train_preds = []
    for i in range(look_back, len(train)):
        input_seq = train[i-look_back:i].reshape((1, look_back, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)
    
    # Generate test predictions (autoregressive)
    current_input = train[-look_back:].reshape((1, look_back, 1))
    test_preds = []
    for _ in range(len(test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return train[look_back:], np.array(train_preds), test, np.array(test_preds)

def run_sarima(train_df, test_df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), seed=42):
    """Run SARIMA model"""
    np.random.seed(seed)
    
    train_series = train_df['Deaths'].astype(float)
    test_series = test_df['Deaths'].astype(float)

    try:
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=100)
        
        fitted = results.fittedvalues
        forecast = results.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
        
        return train_series.values, fitted.values, test_series.values, forecast.values
    except Exception as e:
        print(f"SARIMA failed with order {order}, seasonal {seasonal_order}: {e}")
        # Return simple forecast as fallback
        train_mean = train_series.mean()
        fitted = np.full_like(train_series, train_mean)
        forecast = np.full_like(test_series, train_mean)
        return train_series.values, fitted, test_series.values, forecast

def run_tcn(train, test, look_back, batch_size, epochs, seed):
    """Run TCN model"""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = create_dataset(train, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    model = Sequential([
        TCN(input_shape=(look_back, 1), dilations=[1, 2, 4, 8], 
            nb_filters=64, kernel_size=3, dropout_rate=0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Generate training predictions
    train_preds = []
    for i in range(look_back, len(train)):
        input_seq = train[i-look_back:i].reshape((1, look_back, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        train_preds.append(pred)

    # Generate test predictions (autoregressive)
    current_input = train[-look_back:].reshape((1, look_back, 1))
    test_preds = []
    for _ in range(len(test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        test_preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    return train[look_back:], np.array(train_preds), test, np.array(test_preds)

def build_seq2seq_model(lookback, encoder_units=64, decoder_units=64, use_attention=True):
    """Build seq2seq model with optional attention"""
    encoder_inputs = Input(shape=(lookback, 1), name='encoder_input')
    
    if use_attention:
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_outputs_proj = Dense(decoder_units, name='encoder_proj')(encoder_outputs)
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
        else:
            encoder_outputs_proj = encoder_outputs
            
        decoder_inputs = Input(shape=(1, 1), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention_layer = Attention(name='attention')
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(decoder_units, activation='relu', name='decoder_hidden')(decoder_combined)
        decoder_outputs = Dense(1, name='output_dense')(decoder_hidden)
    else:
        encoder_gru = GRU(encoder_units, return_state=True, name='encoder_gru')
        _, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
            
        decoder_inputs = Input(shape=(1, 1), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, name='decoder_gru')
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
        decoder_outputs = Dense(1, name='decoder_dense')(decoder_outputs)
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)

def run_seq2seq(train, test, look_back, batch_size, epochs, seed, 
                encoder_units=64, decoder_units=64, use_attention=True):
    """Run seq2seq model"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Scaling
    full_series = np.concatenate([train, test])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]
    
    # Prepare training data
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
    y_train = y_train.reshape((-1, 1, 1))
    
    # Build and train model
    model = build_seq2seq_model(look_back, encoder_units, decoder_units, use_attention)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit([X_train, decoder_input_train], y_train, epochs=epochs, batch_size=batch_size, 
              verbose=0, callbacks=[early_stopping], validation_split=0.1)
    
    # Generate training predictions
    train_preds_scaled = []
    for i in range(look_back, len(train)):
        encoder_input = train_scaled[i-look_back:i].reshape((1, look_back, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        train_preds_scaled.append(pred_scaled)
    
    # Generate test predictions (autoregressive)
    test_preds_scaled = []
    current_sequence = train_scaled[-look_back:].copy()
    
    for _ in range(len(test)):
        encoder_input = current_sequence.reshape((1, look_back, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        test_preds_scaled.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # Inverse transform predictions
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    
    return train[look_back:], train_preds_original, test, test_preds_original

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                if i+1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

def run_transformer(train, test, look_back, batch_size, epochs, seed, d_model=64, n_heads=2):
    """Run transformer model"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Scaling
    full_series = np.concatenate([train, test])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]
    
    # Prepare data
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    y_train = y_train.reshape((-1, 1))
    
    # Build transformer model
    inputs = Input(shape=(look_back, 1))
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Generate training predictions
    train_preds_scaled = []
    for i in range(look_back, len(train)):
        input_seq = train_scaled[i-look_back:i].reshape((1, look_back, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        train_preds_scaled.append(pred_scaled)
    
    # Generate test predictions (autoregressive)
    current_seq = train_scaled[-look_back:].copy()
    test_preds_scaled = []
    for _ in range(len(test)):
        input_seq = current_seq.reshape((1, look_back, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        test_preds_scaled.append(pred_scaled)
        current_seq = np.append(current_seq[1:], pred_scaled)
    
    # Inverse transform predictions
    train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
    test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    
    return train[look_back:], train_preds_original, test, test_preds_original

# ------------------ MAIN EXECUTION ------------------ #

def main():
    """Main grid search execution - uses only validation set for hyperparameter evaluation"""
    print("="*80)
    print(f"HYPERPARAMETER GRID SEARCH FOR {MODEL_TYPE.upper()}")
    print("="*80)
    
    # Load data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    train_data, validation_data, test_data = create_train_val_test_split(data)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # IMPORTANT: Only use train and validation for hyperparameter search
    # Test set (2020 onwards) is held out completely until final evaluation
    print(f"\n⚠️  HYPERPARAMETER SEARCH PROTOCOL:")
    print(f"   - Training on: {train_data['Month'].min()} to {train_data['Month'].max()}")
    print(f"   - Evaluating on: {validation_data['Month'].min()} to {validation_data['Month'].max()}")
    print(f"   - Test set HELD OUT: {test_data['Month'].min()} to {test_data['Month'].max()}")
    print(f"   - Test set will only be used for final model evaluation after hyperparameter selection")
    
    # Get model-specific trial count
    trials_per_config = get_trials_per_config(MODEL_TYPE)
    
    print(f"\nRunning grid search for {MODEL_TYPE} model...")
    print(f"Trial mode: {TRIAL_MODE}")
    print(f"Trials per configuration: {trials_per_config}")
    
    total_configs = 0
    
    for seed in SEEDS:
        print(f"\nProcessing seed: {seed}")
        
        if MODEL_TYPE == 'sarima':
            # SARIMA grid search
            for order in SARIMA_ORDERS:
                for seasonal_order in SARIMA_SEASONAL_ORDERS:
                    config_name = f'order_{order}_seasonal_{seasonal_order}'
                    total_configs += 1
                    
                    print(f"  Config {total_configs}: {config_name}")
                    
                    base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                        'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                        config_name)
                    os.makedirs(base_dir, exist_ok=True)
                    
                    metrics_all_test = []
                    metrics_all_train = []
                    
                    for trial in range(trials_per_config):
                        try:
                            # Train on training set, evaluate on validation set only
                            y_train_true, y_train_pred, y_val_true, y_val_pred = run_sarima(
                                train_data, validation_data, order, seasonal_order, seed)
                            
                            # Save predictions (train and validation only)
                            train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                            val_df = pd.DataFrame({'True': y_val_true, 'Pred': y_val_pred})
                            
                            train_df.to_csv(os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                            val_df.to_csv(os.path.join(base_dir, f'trial_{trial}_validation.csv'), index=False)
                            
                            # Calculate metrics for train and validation
                            metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                            metrics_val = evaluate_metrics(y_val_true, y_val_pred)
                            metrics_all_train.append(metrics_train)
                            metrics_all_test.append(metrics_val)  # Using 'test' variable for validation metrics
                            
                        except Exception as e:
                            print(f"    Error in trial {trial}: {e}")
                            continue
                    
                    # Save summary metrics with mean, median, and std
                    if metrics_all_train:
                        summary_train = pd.DataFrame(metrics_all_train).agg(['mean', 'median', 'std'])
                        summary_train.to_csv(os.path.join(base_dir, 'summary_metrics_train.csv'))
                    if metrics_all_test:
                        summary_val = pd.DataFrame(metrics_all_test).agg(['mean', 'median', 'std'])
                        summary_val.to_csv(os.path.join(base_dir, 'summary_metrics_validation.csv'))
        
        elif MODEL_TYPE in ['seq2seq', 'seq2seq_attn']:
            # Seq2seq grid search
            for look_back in LOOKBACKS:
                for bs in BATCH_SIZES:
                    for ep in EPOCHS_LIST:
                        for enc_units in ENCODER_UNITS:
                            for dec_units in DECODER_UNITS:
                                for use_att in USE_ATTENTION:
                                    if MODEL_TYPE == 'seq2seq' and use_att:
                                        continue  # Skip attention for basic seq2seq
                                    if MODEL_TYPE == 'seq2seq_attn' and not use_att:
                                        continue  # Skip non-attention for seq2seq_attn
                                    
                                    config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}_enc_{enc_units}_dec_{dec_units}_att_{use_att}'
                                    total_configs += 1
                                    
                                    print(f"  Config {total_configs}: {config_name}")
                                    
                                    base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                                        'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                                        config_name)
                                    os.makedirs(base_dir, exist_ok=True)
                                    
                                    metrics_all_test = []
                                    metrics_all_train = []
                                    
                                    for trial in range(trials_per_config):
                                        try:
                                            # Train on training set, evaluate on validation set only
                                            y_train_true, y_train_pred, y_val_true, y_val_pred = run_seq2seq(
                                                train_data['Deaths'].values, validation_data['Deaths'].values,
                                                look_back, bs, ep, seed, enc_units, dec_units, use_att)
                                            
                                            # Save predictions (train and validation only)
                                            train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                                            val_df = pd.DataFrame({'True': y_val_true, 'Pred': y_val_pred})
                                            
                                            train_df.to_csv(os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                                            val_df.to_csv(os.path.join(base_dir, f'trial_{trial}_validation.csv'), index=False)
                                            
                                            # Calculate metrics for train and validation
                                            metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                                            metrics_val = evaluate_metrics(y_val_true, y_val_pred)
                                            metrics_all_train.append(metrics_train)
                                            metrics_all_test.append(metrics_val)  # Using 'test' variable for validation metrics
                                            
                                        except Exception as e:
                                            print(f"    Error in trial {trial}: {e}")
                                            continue
                                    
                                    # Save summary metrics with mean, median, and std
                                    if metrics_all_train:
                                        summary_train = pd.DataFrame(metrics_all_train).agg(['mean', 'median', 'std'])
                                        summary_train.to_csv(os.path.join(base_dir, 'summary_metrics_train.csv'))
                                    if metrics_all_test:
                                        summary_val = pd.DataFrame(metrics_all_test).agg(['mean', 'median', 'std'])
                                        summary_val.to_csv(os.path.join(base_dir, 'summary_metrics_validation.csv'))
        
        elif MODEL_TYPE == 'transformer':
            # Transformer grid search
            for look_back in LOOKBACKS:
                for bs in BATCH_SIZES:
                    for ep in EPOCHS_LIST:
                        for d_model in D_MODEL:
                            for n_heads in N_HEADS:
                                config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}_dmodel_{d_model}_heads_{n_heads}'
                                total_configs += 1
                                
                                print(f"  Config {total_configs}: {config_name}")
                                
                                base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                                    'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                                    config_name)
                                os.makedirs(base_dir, exist_ok=True)
                                
                                metrics_all_test = []
                                metrics_all_train = []
                                
                                for trial in range(trials_per_config):
                                    try:
                                        # Train on training set, evaluate on validation set only
                                        y_train_true, y_train_pred, y_val_true, y_val_pred = run_transformer(
                                            train_data['Deaths'].values, validation_data['Deaths'].values,
                                            look_back, bs, ep, seed, d_model, n_heads)
                                        
                                        # Save predictions (train and validation only)
                                        train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                                        val_df = pd.DataFrame({'True': y_val_true, 'Pred': y_val_pred})
                                        
                                        train_df.to_csv(os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                                        val_df.to_csv(os.path.join(base_dir, f'trial_{trial}_validation.csv'), index=False)
                                        
                                        # Calculate metrics for train and validation
                                        metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                                        metrics_val = evaluate_metrics(y_val_true, y_val_pred)
                                        metrics_all_train.append(metrics_train)
                                        metrics_all_test.append(metrics_val)  # Using 'test' variable for validation metrics
                                        
                                    except Exception as e:
                                        print(f"    Error in trial {trial}: {e}")
                                        continue
                                
                                # Save summary metrics with mean, median, and std
                                if metrics_all_train:
                                    summary_train = pd.DataFrame(metrics_all_train).agg(['mean', 'median', 'std'])
                                    summary_train.to_csv(os.path.join(base_dir, 'summary_metrics_train.csv'))
                                if metrics_all_test:
                                    summary_val = pd.DataFrame(metrics_all_test).agg(['mean', 'median', 'std'])
                                    summary_val.to_csv(os.path.join(base_dir, 'summary_metrics_validation.csv'))
        
        else:
            # Standard grid search for LSTM, TCN
            for look_back in LOOKBACKS:
                for bs in BATCH_SIZES:
                    for ep in EPOCHS_LIST:
                        config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}'
                        total_configs += 1
                        
                        print(f"  Config {total_configs}: {config_name}")
                        
                        base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                            'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                            config_name)
                        os.makedirs(base_dir, exist_ok=True)
                        
                        metrics_all_test = []
                        metrics_all_train = []
                        
                        for trial in range(trials_per_config):
                            try:
                                if MODEL_TYPE == 'lstm':
                                    # Train on training set, evaluate on validation set only
                                    y_train_true, y_train_pred, y_val_true, y_val_pred = run_lstm(
                                        train_data['Deaths'].values, validation_data['Deaths'].values,
                                        look_back, bs, ep, seed)
                                elif MODEL_TYPE == 'tcn':
                                    # Train on training set, evaluate on validation set only
                                    y_train_true, y_train_pred, y_val_true, y_val_pred = run_tcn(
                                        train_data['Deaths'].values, validation_data['Deaths'].values,
                                        look_back, bs, ep, seed)
                                else:
                                    raise ValueError(f"Unknown model type: {MODEL_TYPE}")
                                
                                # Save predictions (train and validation only)
                                train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                                val_df = pd.DataFrame({'True': y_val_true, 'Pred': y_val_pred})
                                
                                train_df.to_csv(os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                                val_df.to_csv(os.path.join(base_dir, f'trial_{trial}_validation.csv'), index=False)
                                
                                # Calculate metrics for train and validation
                                metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                                metrics_val = evaluate_metrics(y_val_true, y_val_pred)
                                metrics_all_train.append(metrics_train)
                                metrics_all_test.append(metrics_val)  # Using 'test' variable for validation metrics
                                
                            except Exception as e:
                                print(f"    Error in trial {trial}: {e}")
                                continue
                        
                        # Save summary metrics with mean, median, and std
                        if metrics_all_train:
                            summary_train = pd.DataFrame(metrics_all_train).agg(['mean', 'median', 'std'])
                            summary_train.to_csv(os.path.join(base_dir, 'summary_metrics_train.csv'))
                        if metrics_all_test:
                            summary_val = pd.DataFrame(metrics_all_test).agg(['mean', 'median', 'std'])
                            summary_val.to_csv(os.path.join(base_dir, 'summary_metrics_validation.csv'))

    print(f"\nGrid search completed!")
    print(f"Total configurations tested: {total_configs}")
    print(f"Results saved to: {RESULTS_DIR}")
    
    # Create mean vs median analysis for the current model
    try:
        create_model_comparison_analysis(RESULTS_DIR, MODEL_TYPE)
    except Exception as e:
        print(f"Could not create comparison analysis: {e}")
    
    # Print summary of configurations tested
    print(f"\nConfiguration summary:")
    if MODEL_TYPE == 'sarima':
        print(f"  - SARIMA orders: {len(SARIMA_ORDERS)}")
        print(f"  - Seasonal orders: {len(SARIMA_SEASONAL_ORDERS)}")
        print(f"  - Total SARIMA configs: {len(SARIMA_ORDERS) * len(SARIMA_SEASONAL_ORDERS)}")
    elif MODEL_TYPE in ['seq2seq', 'seq2seq_attn']:
        valid_attention_configs = 1 if MODEL_TYPE == 'seq2seq' else 1 if MODEL_TYPE == 'seq2seq_attn' else 2
        seq2seq_configs = (len(LOOKBACKS) * len(BATCH_SIZES) * len(EPOCHS_LIST) * 
                          len(ENCODER_UNITS) * len(DECODER_UNITS) * valid_attention_configs)
        print(f"  - Lookbacks: {len(LOOKBACKS)}")
        print(f"  - Batch sizes: {len(BATCH_SIZES)}")
        print(f"  - Epochs: {len(EPOCHS_LIST)}")
        print(f"  - Encoder units: {len(ENCODER_UNITS)}")
        print(f"  - Decoder units: {len(DECODER_UNITS)}")
        print(f"  - Total seq2seq configs: {seq2seq_configs}")
    elif MODEL_TYPE == 'transformer':
        transformer_configs = (len(LOOKBACKS) * len(BATCH_SIZES) * len(EPOCHS_LIST) * 
                             len(D_MODEL) * len(N_HEADS))
        print(f"  - Lookbacks: {len(LOOKBACKS)}")
        print(f"  - Batch sizes: {len(BATCH_SIZES)}")
        print(f"  - Epochs: {len(EPOCHS_LIST)}")
        print(f"  - D_model: {len(D_MODEL)}")
        print(f"  - N_heads: {len(N_HEADS)}")
        print(f"  - Total transformer configs: {transformer_configs}")
    else:  # LSTM, TCN
        standard_configs = len(LOOKBACKS) * len(BATCH_SIZES) * len(EPOCHS_LIST)
        print(f"  - Lookbacks: {len(LOOKBACKS)}")
        print(f"  - Batch sizes: {len(BATCH_SIZES)}")
        print(f"  - Epochs: {len(EPOCHS_LIST)}")
        print(f"  - Total {MODEL_TYPE.upper()} configs: {standard_configs}")
    
    print(f"  - Seeds: {len(SEEDS)}")
    print(f"  - Trials per config: {trials_per_config}")
    
    # Recommendations for analysis
    print(f"\n{'='*60}")
    print("ANALYSIS RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. Compare mean vs median VALIDATION RMSE rankings to identify outliers")
    print("2. Look for configurations with high coefficient of variation (CV)")
    print("3. Check if best model is consistent between mean and median rankings")
    print("4. Investigate configurations with large rank differences")
    print("5. Consider using median RMSE for final model selection (more robust)")
    print(f"6. Analysis file: {RESULTS_DIR}/{MODEL_TYPE}_mean_vs_median_analysis.csv")
    print("")
    print("NEXT STEPS FOR FINAL EVALUATION:")
    print("1. Select optimal hyperparameters based on VALIDATION performance")
    print("2. Retrain model on TRAIN + VALIDATION data with optimal hyperparameters")
    print("3. Evaluate final model on TEST set (2020 onwards) for unbiased performance")
    print("4. Use final model for forecasting beyond 2023")

if __name__ == "__main__":
    main()