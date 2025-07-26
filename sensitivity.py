#!/usr/bin/env python3
"""
Sensitivity Analysis Script for Model Convergence Evaluation

This script performs sensitivity analysis to study convergence behavior of model metrics
as we vary:
1. Number of random seeds (seed sensitivity)
2. Number of trials per seed (trial sensitivity)

Usage:
    python sensitivity_analysis.py --model lstm --analysis seed --max_seeds 100 --eval_interval 5
    python sensitivity_analysis.py --model lstm --analysis trial --max_trials 200 --eval_interval 10

The script saves incremental results and creates convergence plots showing how metrics
stabilize as we increase the number of seeds or trials.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import json
import pickle
from pathlib import Path

# Model-specific imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import RepeatVector, Concatenate, Input, Add, ReLU, Lambda, Dropout, BatchNormalization
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Attention
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

try:
    from tcn import TCN
except ImportError:
    print("Warning: TCN not available. Install with: pip install keras-tcn")

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SensitivityAnalyzer:
    """Main class for conducting sensitivity analysis experiments"""
    
    def __init__(self, model_name, analysis_type, data_path='data/state_month_overdose.xlsx'):
        self.model_name = model_name
        self.analysis_type = analysis_type  # 'seed' or 'trial'
        self.data_path = data_path
        
        # Create output directories
        self.output_dir = f'sensitivity_analysis_{model_name}_{analysis_type}'
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        for dir_path in [self.output_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Load optimal hyperparameters
        self.optimal_params = self._get_optimal_params()
        
        # Initialize results storage
        self.convergence_results = []
        self.all_predictions = {}
        
        print(f"Initialized Sensitivity Analyzer for {model_name} - {analysis_type} analysis")
        print(f"Output directory: {self.output_dir}")
    
    def _get_optimal_params(self):
        """Get optimal hyperparameters for each model"""
        params = {
            'sarima': {'order': (1, 0, 0), 'seasonal_order': (2, 2, 2, 12)},
            'lstm': {'lookback': 9, 'batch_size': 8, 'epochs': 100},
            'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50},
            'seq2seq_attn': {'lookback': 11, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 128},
            'transformer': {'lookback': 5, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
        }
        return params[self.model_name]
    
    def load_and_preprocess_data(self):
        """Load and preprocess the overdose data"""
        df = pd.read_excel(self.data_path)
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
        return df
    
    def create_train_val_test_split(self, df, train_end='2019-01-01', val_end='2020-01-01'):
        """Create train/validation/test splits"""
        train = df[df['Month'] < train_end]
        validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
        test = df[df['Month'] >= val_end]
        
        # Combine train and validation for final training
        train_val = pd.concat([train, validation], ignore_index=True)
        return train_val, test
    
    def create_dataset(self, series, look_back):
        """Create dataset for supervised learning"""
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i+look_back])
            y.append(series[i+look_back])
        return np.array(X), np.array(y)
    
    def evaluate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mse = mean_squared_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MSE': mse}
    
    def calculate_prediction_intervals(self, actual, predictions, alpha=0.05):
        """Calculate prediction intervals"""
        residuals = actual - predictions
        std_residual = np.std(residuals)
        z_score = 1.96  # 95% confidence interval
        margin_of_error = z_score * std_residual
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        return lower_bound, upper_bound
    
    def calculate_pi_coverage(self, actual, lower_bound, upper_bound):
        """Calculate prediction interval coverage"""
        coverage = np.mean((actual >= lower_bound) & (actual <= upper_bound))
        return coverage * 100
    
    def run_single_model_trial(self, train_val_data, test_data, seed):
        """Run a single trial of the specified model"""
        
        if self.model_name == 'sarima':
            return self._run_sarima_trial(train_val_data, test_data, seed)
        elif self.model_name == 'lstm':
            return self._run_lstm_trial(train_val_data, test_data, seed)
        elif self.model_name == 'tcn':
            return self._run_tcn_trial(train_val_data, test_data, seed)
        elif self.model_name == 'seq2seq_attn':
            return self._run_seq2seq_trial(train_val_data, test_data, seed)
        elif self.model_name == 'transformer':
            return self._run_transformer_trial(train_val_data, test_data, seed)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _run_sarima_trial(self, train_val_df, test_df, seed):
        """Run SARIMA model trial"""
        np.random.seed(seed)
        
        train_val_series = train_val_df['Deaths'].reset_index(drop=True).astype(float)
        test_series = test_df['Deaths'].reset_index(drop=True).astype(float)
        
        model = SARIMAX(train_val_series, 
                        order=self.optimal_params['order'], 
                        seasonal_order=self.optimal_params['seasonal_order'],
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        
        train_predictions = results.fittedvalues.values
        test_predictions = results.predict(start=len(train_val_series), 
                                          end=len(train_val_series) + len(test_series) - 1).values
        
        return train_val_series.values, train_predictions, test_series.values, test_predictions
    
    def _run_lstm_trial(self, train_val_data, test_data, seed):
        """Run LSTM model trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        lookback = self.optimal_params['lookback']
        batch_size = self.optimal_params['batch_size']
        epochs = self.optimal_params['epochs']
        
        train_val_values = train_val_data['Deaths'].values
        test_values = test_data['Deaths'].values
        
        # Prepare training data
        X_train, y_train = self.create_dataset(train_val_values, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        
        # Build and train model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Generate training predictions
        train_preds = []
        for i in range(lookback, len(train_val_values)):
            input_seq = train_val_values[i-lookback:i].reshape((1, lookback, 1))
            pred = model.predict(input_seq, verbose=0)[0][0]
            train_preds.append(pred)
        
        # Generate test predictions (autoregressive)
        current_input = train_val_values[-lookback:].reshape((1, lookback, 1))
        test_preds = []
        for _ in range(len(test_values)):
            pred = model.predict(current_input, verbose=0)[0][0]
            test_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return (train_val_values[lookback:], np.array(train_preds), 
                test_values, np.array(test_preds))
    
    def _run_tcn_trial(self, train_val_data, test_data, seed):
        """Run TCN model trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        lookback = self.optimal_params['lookback']
        batch_size = self.optimal_params['batch_size']
        epochs = self.optimal_params['epochs']
        
        train_val_values = train_val_data['Deaths'].values
        test_values = test_data['Deaths'].values
        
        # Prepare training data
        X_train, y_train = self.create_dataset(train_val_values, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        
        # Build and train model
        model = Sequential([
            TCN(input_shape=(lookback, 1), dilations=[1, 2, 4, 8], 
                nb_filters=64, kernel_size=3, dropout_rate=0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Generate training predictions
        train_preds = []
        for i in range(lookback, len(train_val_values)):
            input_seq = train_val_values[i-lookback:i].reshape((1, lookback, 1))
            pred = model.predict(input_seq, verbose=0)[0][0]
            train_preds.append(pred)
        
        # Generate test predictions (autoregressive)
        current_input = train_val_values[-lookback:].reshape((1, lookback, 1))
        test_preds = []
        for _ in range(len(test_values)):
            pred = model.predict(current_input, verbose=0)[0][0]
            test_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return (train_val_values[lookback:], np.array(train_preds), 
                test_values, np.array(test_preds))
    
    def _run_seq2seq_trial(self, train_val_data, test_data, seed):
        """Run Seq2Seq with Attention model trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        lookback = self.optimal_params['lookback']
        batch_size = self.optimal_params['batch_size']
        epochs = self.optimal_params['epochs']
        encoder_units = self.optimal_params['encoder_units']
        decoder_units = self.optimal_params['decoder_units']
        
        train_val_values = train_val_data['Deaths'].values
        test_values = test_data['Deaths'].values
        
        # Scaling
        full_series = np.concatenate([train_val_values, test_values])
        scaler = MinMaxScaler()
        scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
        
        train_val_scaled = scaled_full[:len(train_val_values)]
        test_scaled = scaled_full[len(train_val_values):]
        
        # Prepare training data
        X_train, y_train = self.create_dataset(train_val_scaled, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
        y_train = y_train.reshape((-1, 1, 1))
        
        # Build model
        encoder_inputs = Input(shape=(lookback, 1))
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True)
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_outputs_proj = Dense(decoder_units)(encoder_outputs)
            encoder_state = Dense(decoder_units)(encoder_state)
        else:
            encoder_outputs_proj = encoder_outputs
            
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention_layer = Attention()
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(decoder_units, activation='relu')(decoder_combined)
        decoder_outputs = Dense(1)(decoder_hidden)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        model.fit([X_train, decoder_input_train], y_train, epochs=epochs, batch_size=batch_size, 
                  verbose=0, callbacks=[early_stopping], validation_split=0.1)
        
        # Generate training predictions
        train_preds_scaled = []
        for i in range(lookback, len(train_val_values)):
            encoder_input = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
            decoder_input = np.zeros((1, 1, 1))
            pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
            train_preds_scaled.append(pred_scaled)
        
        # Generate test predictions (autoregressive)
        test_preds_scaled = []
        current_sequence = train_val_scaled[-lookback:].copy()
        
        for _ in range(len(test_values)):
            encoder_input = current_sequence.reshape((1, lookback, 1))
            decoder_input = np.zeros((1, 1, 1))
            pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
            test_preds_scaled.append(pred_scaled)
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        # Inverse transform predictions
        train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
        
        return (train_val_values[lookback:], train_preds_original, 
                test_values, test_preds_original)
    
    def _run_transformer_trial(self, train_val_data, test_data, seed):
        """Run Transformer model trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        lookback = self.optimal_params['lookback']
        batch_size = self.optimal_params['batch_size']
        epochs = self.optimal_params['epochs']
        d_model = self.optimal_params['d_model']
        n_heads = self.optimal_params['n_heads']
        
        train_val_values = train_val_data['Deaths'].values
        test_values = test_data['Deaths'].values
        
        # Scaling
        full_series = np.concatenate([train_val_values, test_values])
        scaler = MinMaxScaler()
        scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
        
        train_val_scaled = scaled_full[:len(train_val_values)]
        test_scaled = scaled_full[len(train_val_values):]
        
        # Prepare data
        X_train, y_train = self.create_dataset(train_val_scaled, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        y_train = y_train.reshape((-1, 1))
        
        # Build transformer model
        class PositionalEncoding(tf.keras.layers.Layer):
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
        
        inputs = Input(shape=(lookback, 1))
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
        for i in range(lookback, len(train_val_values)):
            input_seq = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            train_preds_scaled.append(pred_scaled)
        
        # Generate test predictions (autoregressive)
        current_seq = train_val_scaled[-lookback:].copy()
        test_preds_scaled = []
        for _ in range(len(test_values)):
            input_seq = current_seq.reshape((1, lookback, 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            test_preds_scaled.append(pred_scaled)
            current_seq = np.append(current_seq[1:], pred_scaled)
        
        # Inverse transform predictions
        train_preds_original = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        test_preds_original = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
        
        return (train_val_values[lookback:], train_preds_original, 
                test_values, test_preds_original)
    
    def run_seed_sensitivity_analysis(self, max_seeds=100, eval_interval=5, fixed_trials=50):
        """Run sensitivity analysis varying number of seeds"""
        print(f"Starting seed sensitivity analysis for {self.model_name}")
        print(f"Max seeds: {max_seeds}, Evaluation interval: {eval_interval}, Fixed trials per seed: {fixed_trials}")
        
        # Load data
        data = self.load_and_preprocess_data()
        train_val_data, test_data = self.create_train_val_test_split(data)
        
        # Storage for all trial results
        all_trial_results = []
        evaluation_points = list(range(eval_interval, max_seeds + 1, eval_interval))
        
        # Run trials for all seeds
        for seed_idx in range(max_seeds):
            seed = 42 + seed_idx * 1000  # Generate different seeds
            print(f"Processing seed {seed_idx + 1}/{max_seeds} (seed={seed})")
            
            # Run multiple trials for this seed
            seed_trial_results = []
            for trial in range(fixed_trials):
                trial_seed = seed + trial
                
                try:
                    train_true, train_pred, test_true, test_pred = self.run_single_model_trial(
                        train_val_data, test_data, trial_seed)
                    
                    # Calculate metrics
                    train_metrics = self.evaluate_metrics(train_true, train_pred)
                    test_metrics = self.evaluate_metrics(test_true, test_pred)
                    
                    # Calculate prediction intervals and coverage
                    train_lower, train_upper = self.calculate_prediction_intervals(train_true, train_pred)
                    test_lower, test_upper = self.calculate_prediction_intervals(test_true, test_pred)
                    
                    train_coverage = self.calculate_pi_coverage(train_true, train_lower, train_upper)
                    test_coverage = self.calculate_pi_coverage(test_true, test_lower, test_upper)
                    
                    trial_result = {
                        'seed_idx': seed_idx,
                        'seed': seed,
                        'trial': trial,
                        'trial_seed': trial_seed,
                        'train_rmse': train_metrics['RMSE'],
                        'train_mae': train_metrics['MAE'],
                        'train_mape': train_metrics['MAPE'],
                        'train_pi_coverage': train_coverage,
                        'test_rmse': test_metrics['RMSE'],
                        'test_mae': test_metrics['MAE'],
                        'test_mape': test_metrics['MAPE'],
                        'test_pi_coverage': test_coverage
                    }
                    
                    seed_trial_results.append(trial_result)
                    all_trial_results.append(trial_result)
                    
                except Exception as e:
                    print(f"    Error in trial {trial}: {str(e)}")
                    continue
            
            # Evaluate at intervals
            if (seed_idx + 1) in evaluation_points:
                num_seeds_used = seed_idx + 1
                print(f"  Evaluating convergence at {num_seeds_used} seeds...")
                self._evaluate_seed_convergence(all_trial_results, num_seeds_used, fixed_trials)
        
        # Final evaluation and plotting
        print("Creating final convergence plots...")
        self._create_seed_convergence_plots(evaluation_points, fixed_trials)
        
        # Save all trial results
        results_df = pd.DataFrame(all_trial_results)
        results_df.to_csv(os.path.join(self.results_dir, 'all_seed_trial_results.csv'), index=False)
        
        print(f"Seed sensitivity analysis complete. Results saved to {self.output_dir}")
    
    def run_trial_sensitivity_analysis(self, max_trials=200, eval_interval=10, fixed_seed=42):
        """Run sensitivity analysis varying number of trials"""
        print(f"Starting trial sensitivity analysis for {self.model_name}")
        print(f"Max trials: {max_trials}, Evaluation interval: {eval_interval}, Fixed seed: {fixed_seed}")
        
        # Load data
        data = self.load_and_preprocess_data()
        train_val_data, test_data = self.create_train_val_test_split(data)
        
        # Storage for all trial results
        all_trial_results = []
        evaluation_points = list(range(eval_interval, max_trials + 1, eval_interval))
        
        # Run all trials
        for trial in range(max_trials):
            trial_seed = fixed_seed + trial * 1000
            print(f"Processing trial {trial + 1}/{max_trials} (seed={trial_seed})")
            
            try:
                train_true, train_pred, test_true, test_pred = self.run_single_model_trial(
                    train_val_data, test_data, trial_seed)
                
                # Calculate metrics
                train_metrics = self.evaluate_metrics(train_true, train_pred)
                test_metrics = self.evaluate_metrics(test_true, test_pred)
                
                # Calculate prediction intervals and coverage
                train_lower, train_upper = self.calculate_prediction_intervals(train_true, train_pred)
                test_lower, test_upper = self.calculate_prediction_intervals(test_true, test_pred)
                
                train_coverage = self.calculate_pi_coverage(train_true, train_lower, train_upper)
                test_coverage = self.calculate_pi_coverage(test_true, test_lower, test_upper)
                
                trial_result = {
                    'trial': trial,
                    'trial_seed': trial_seed,
                    'train_rmse': train_metrics['RMSE'],
                    'train_mae': train_metrics['MAE'],
                    'train_mape': train_metrics['MAPE'],
                    'train_pi_coverage': train_coverage,
                    'test_rmse': test_metrics['RMSE'],
                    'test_mae': test_metrics['MAE'],
                    'test_mape': test_metrics['MAPE'],
                    'test_pi_coverage': test_coverage
                }
                
                all_trial_results.append(trial_result)
                
            except Exception as e:
                print(f"    Error in trial {trial}: {str(e)}")
                continue
            
            # Evaluate at intervals
            if (trial + 1) in evaluation_points:
                num_trials_used = trial + 1
                print(f"  Evaluating convergence at {num_trials_used} trials...")
                self._evaluate_trial_convergence(all_trial_results, num_trials_used)
        
        # Final evaluation and plotting
        print("Creating final convergence plots...")
        self._create_trial_convergence_plots(evaluation_points)
        
        # Save all trial results
        results_df = pd.DataFrame(all_trial_results)
        results_df.to_csv(os.path.join(self.results_dir, 'all_trial_results.csv'), index=False)
        
        print(f"Trial sensitivity analysis complete. Results saved to {self.output_dir}")
    
    def _evaluate_seed_convergence(self, all_results, num_seeds, trials_per_seed):
        """Evaluate convergence metrics for seed analysis"""
        results_df = pd.DataFrame(all_results)
        
        # Filter to only use the first num_seeds
        filtered_results = results_df[results_df['seed_idx'] < num_seeds]
        
        # Calculate statistics across seeds (averaging trials within each seed first)
        seed_averages = filtered_results.groupby('seed_idx').agg({
            'train_rmse': 'mean',
            'train_mae': 'mean', 
            'train_mape': 'mean',
            'train_pi_coverage': 'mean',
            'test_rmse': 'mean',
            'test_mae': 'mean',
            'test_mape': 'mean',
            'test_pi_coverage': 'mean'
        }).reset_index()
        
        # Calculate confidence intervals across seeds
        convergence_stats = {}
        for metric in ['train_rmse', 'train_mae', 'train_mape', 'train_pi_coverage',
                      'test_rmse', 'test_mae', 'test_mape', 'test_pi_coverage']:
            values = seed_averages[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            lower_ci = mean_val - 1.96 * std_val / np.sqrt(len(values))
            upper_ci = mean_val + 1.96 * std_val / np.sqrt(len(values))
            
            convergence_stats[metric] = {
                'mean': mean_val,
                'std': std_val,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'ci_width': upper_ci - lower_ci
            }
        
        # Store convergence results
        convergence_result = {
            'num_seeds': num_seeds,
            'trials_per_seed': trials_per_seed,
            'timestamp': datetime.now().isoformat(),
            'metrics': convergence_stats
        }
        
        self.convergence_results.append(convergence_result)
        
        # Save incremental results
        with open(os.path.join(self.results_dir, f'seed_convergence_{num_seeds:03d}.json'), 'w') as f:
            json.dump(convergence_result, f, indent=2)
    
    def _evaluate_trial_convergence(self, all_results, num_trials):
        """Evaluate convergence metrics for trial analysis"""
        results_df = pd.DataFrame(all_results)
        
        # Filter to only use the first num_trials
        filtered_results = results_df.iloc[:num_trials]
        
        # Calculate cumulative statistics
        convergence_stats = {}
        for metric in ['train_rmse', 'train_mae', 'train_mape', 'train_pi_coverage',
                      'test_rmse', 'test_mae', 'test_mape', 'test_pi_coverage']:
            values = filtered_results[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            lower_ci = mean_val - 1.96 * std_val / np.sqrt(len(values))
            upper_ci = mean_val + 1.96 * std_val / np.sqrt(len(values))
            
            convergence_stats[metric] = {
                'mean': mean_val,
                'std': std_val,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'ci_width': upper_ci - lower_ci
            }
        
        # Store convergence results
        convergence_result = {
            'num_trials': num_trials,
            'timestamp': datetime.now().isoformat(),
            'metrics': convergence_stats
        }
        
        self.convergence_results.append(convergence_result)
        
        # Save incremental results
        with open(os.path.join(self.results_dir, f'trial_convergence_{num_trials:03d}.json'), 'w') as f:
            json.dump(convergence_result, f, indent=2)
    
    def _create_seed_convergence_plots(self, evaluation_points, trials_per_seed):
        """Create convergence plots for seed analysis"""
        
        # Load all convergence results
        convergence_data = []
        for num_seeds in evaluation_points:
            result_file = os.path.join(self.results_dir, f'seed_convergence_{num_seeds:03d}.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    convergence_data.append(data)
        
        if not convergence_data:
            print("No convergence data found for plotting")
            return
        
        # Extract data for plotting
        metrics_to_plot = ['test_rmse', 'test_mae', 'test_mape', 'test_pi_coverage']
        metric_labels = {'test_rmse': 'Test RMSE', 'test_mae': 'Test MAE', 
                        'test_mape': 'Test MAPE (%)', 'test_pi_coverage': 'Test PI Coverage (%)'}
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Extract data
            num_seeds_list = [d['num_seeds'] for d in convergence_data]
            means = [d['metrics'][metric]['mean'] for d in convergence_data]
            lower_cis = [d['metrics'][metric]['lower_ci'] for d in convergence_data]
            upper_cis = [d['metrics'][metric]['upper_ci'] for d in convergence_data]
            
            # Plot mean line
            ax.plot(num_seeds_list, means, 'b-', linewidth=2, label='Mean', marker='o')
            
            # Plot confidence interval
            ax.fill_between(num_seeds_list, lower_cis, upper_cis, alpha=0.3, color='blue', 
                           label='95% Confidence Interval')
            
            # Formatting
            ax.set_xlabel('Number of Random Seeds')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f'{metric_labels[metric]} Convergence\n({self.model_name.upper()}, {trials_per_seed} trials/seed)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add convergence info
            if len(means) > 1:
                final_ci_width = upper_cis[-1] - lower_cis[-1]
                ax.text(0.02, 0.98, f'Final CI Width: {final_ci_width:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'seed_convergence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create CI width convergence plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for metric in metrics_to_plot:
            num_seeds_list = [d['num_seeds'] for d in convergence_data]
            ci_widths = [d['metrics'][metric]['ci_width'] for d in convergence_data]
            ax.plot(num_seeds_list, ci_widths, marker='o', linewidth=2, label=metric_labels[metric])
        
        ax.set_xlabel('Number of Random Seeds')
        ax.set_ylabel('95% Confidence Interval Width')
        ax.set_title(f'Confidence Interval Width Convergence\n({self.model_name.upper()}, {trials_per_seed} trials/seed)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'seed_ci_width_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Seed convergence plots saved to {self.plots_dir}")
    
    def _create_trial_convergence_plots(self, evaluation_points):
        """Create convergence plots for trial analysis"""
        
        # Load all convergence results
        convergence_data = []
        for num_trials in evaluation_points:
            result_file = os.path.join(self.results_dir, f'trial_convergence_{num_trials:03d}.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    convergence_data.append(data)
        
        if not convergence_data:
            print("No convergence data found for plotting")
            return
        
        # Extract data for plotting
        metrics_to_plot = ['test_rmse', 'test_mae', 'test_mape', 'test_pi_coverage']
        metric_labels = {'test_rmse': 'Test RMSE', 'test_mae': 'Test MAE', 
                        'test_mape': 'Test MAPE (%)', 'test_pi_coverage': 'Test PI Coverage (%)'}
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Extract data
            num_trials_list = [d['num_trials'] for d in convergence_data]
            means = [d['metrics'][metric]['mean'] for d in convergence_data]
            lower_cis = [d['metrics'][metric]['lower_ci'] for d in convergence_data]
            upper_cis = [d['metrics'][metric]['upper_ci'] for d in convergence_data]
            
            # Plot mean line
            ax.plot(num_trials_list, means, 'r-', linewidth=2, label='Mean', marker='s')
            
            # Plot confidence interval
            ax.fill_between(num_trials_list, lower_cis, upper_cis, alpha=0.3, color='red', 
                           label='95% Confidence Interval')
            
            # Formatting
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f'{metric_labels[metric]} Convergence\n({self.model_name.upper()}, fixed seed)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add convergence info
            if len(means) > 1:
                final_ci_width = upper_cis[-1] - lower_cis[-1]
                ax.text(0.02, 0.98, f'Final CI Width: {final_ci_width:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'trial_convergence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create CI width convergence plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for metric in metrics_to_plot:
            num_trials_list = [d['num_trials'] for d in convergence_data]
            ci_widths = [d['metrics'][metric]['ci_width'] for d in convergence_data]
            ax.plot(num_trials_list, ci_widths, marker='s', linewidth=2, label=metric_labels[metric])
        
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('95% Confidence Interval Width')
        ax.set_title(f'Confidence Interval Width Convergence\n({self.model_name.upper()}, fixed seed)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'trial_ci_width_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trial convergence plots saved to {self.plots_dir}")
    
    def create_summary_report(self):
        """Create a summary report of the sensitivity analysis"""
        
        summary = {
            'model_name': self.model_name,
            'analysis_type': self.analysis_type,
            'optimal_parameters': self.optimal_params,
            'timestamp': datetime.now().isoformat(),
            'output_directory': self.output_dir
        }
        
        # Add analysis-specific information
        if self.analysis_type == 'seed':
            # Find the latest seed convergence result
            seed_files = [f for f in os.listdir(self.results_dir) if f.startswith('seed_convergence_')]
            if seed_files:
                latest_file = max(seed_files)
                with open(os.path.join(self.results_dir, latest_file), 'r') as f:
                    latest_result = json.load(f)
                summary['final_num_seeds'] = latest_result['num_seeds']
                summary['trials_per_seed'] = latest_result['trials_per_seed']
                summary['final_metrics'] = latest_result['metrics']
        
        elif self.analysis_type == 'trial':
            # Find the latest trial convergence result
            trial_files = [f for f in os.listdir(self.results_dir) if f.startswith('trial_convergence_')]
            if trial_files:
                latest_file = max(trial_files)
                with open(os.path.join(self.results_dir, latest_file), 'r') as f:
                    latest_result = json.load(f)
                summary['final_num_trials'] = latest_result['num_trials']
                summary['final_metrics'] = latest_result['metrics']
        
        # Save summary
        with open(os.path.join(self.output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary table
        if 'final_metrics' in summary:
            metrics_df = pd.DataFrame(summary['final_metrics']).T
            metrics_df = metrics_df.round(4)
            metrics_df.to_csv(os.path.join(self.output_dir, 'final_metrics_summary.csv'))
        
        print(f"Summary report saved to {self.output_dir}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Model Sensitivity Analysis')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['sarima', 'lstm', 'tcn', 'seq2seq_attn', 'transformer'],
                       help='Model to analyze')
    
    parser.add_argument('--analysis', type=str, required=True,
                       choices=['seed', 'trial'],
                       help='Type of sensitivity analysis')
    
    parser.add_argument('--max_seeds', type=int, default=50,
                       help='Maximum number of seeds for seed analysis (default: 50)')
    
    parser.add_argument('--max_trials', type=int, default=100,
                       help='Maximum number of trials for trial analysis (default: 100)')
    
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Evaluation interval (default: 5)')
    
    parser.add_argument('--fixed_trials', type=int, default=30,
                       help='Fixed number of trials per seed for seed analysis (default: 30)')
    
    parser.add_argument('--fixed_seed', type=int, default=42,
                       help='Fixed seed for trial analysis (default: 42)')
    
    parser.add_argument('--data_path', type=str, default='data/state_month_overdose.xlsx',
                       help='Path to data file (default: data/state_month_overdose.xlsx)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.analysis == 'seed' and args.max_seeds < args.eval_interval:
        print(f"Error: max_seeds ({args.max_seeds}) must be >= eval_interval ({args.eval_interval})")
        return
    
    if args.analysis == 'trial' and args.max_trials < args.eval_interval:
        print(f"Error: max_trials ({args.max_trials}) must be >= eval_interval ({args.eval_interval})")
        return
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    print("="*80)
    print(f"SENSITIVITY ANALYSIS: {args.model.upper()} - {args.analysis.upper()}")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Analysis Type: {args.analysis}")
    print(f"Data Path: {args.data_path}")
    
    if args.analysis == 'seed':
        print(f"Max Seeds: {args.max_seeds}")
        print(f"Fixed Trials per Seed: {args.fixed_trials}")
    else:
        print(f"Max Trials: {args.max_trials}")
        print(f"Fixed Seed: {args.fixed_seed}")
    
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(args.model, args.analysis, args.data_path)
    
    try:
        # Run analysis
        if args.analysis == 'seed':
            analyzer.run_seed_sensitivity_analysis(
                max_seeds=args.max_seeds,
                eval_interval=args.eval_interval,
                fixed_trials=args.fixed_trials
            )
        else:
            analyzer.run_trial_sensitivity_analysis(
                max_trials=args.max_trials,
                eval_interval=args.eval_interval,
                fixed_seed=args.fixed_seed
            )
        
        # Create summary report
        analyzer.create_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {analyzer.output_dir}")
        print("Generated files:")
        print(f"  ├── results/")
        if args.analysis == 'seed':
            print(f"  │   ├── seed_convergence_XXX.json")
            print(f"  │   └── all_seed_trial_results.csv")
        else:
            print(f"  │   ├── trial_convergence_XXX.json")
            print(f"  │   └── all_trial_results.csv")
        print(f"  ├── plots/")
        if args.analysis == 'seed':
            print(f"  │   ├── seed_convergence_analysis.png")
            print(f"  │   └── seed_ci_width_convergence.png")
        else:
            print(f"  │   ├── trial_convergence_analysis.png")
            print(f"  │   └── trial_ci_width_convergence.png")
        print(f"  ├── analysis_summary.json")
        print(f"  └── final_metrics_summary.csv")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
