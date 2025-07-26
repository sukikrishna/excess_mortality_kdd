#!/usr/bin/env python3
"""
Model Ranking Sensitivity Analysis Script

This script performs comprehensive sensitivity analysis to determine the minimum
number of random seeds and trials needed for stable model rankings on the validation set.

The analysis creates a grid/heatmap over (number of trials, number of seeds) combinations
and evaluates:
1. Conditional on number of seeds, how many trials are required for ranking stability?
2. Conditional on number of trials, how many seeds are required for ranking stability?
3. What is the optimal (trials, seeds) combination for computational efficiency?

Usage:
    python ranking_sensitivity_analysis.py --max_seeds 20 --max_trials 100 --eval_interval 5
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
import time
from itertools import product
from scipy.stats import spearmanr, kendalltau

# Model-specific imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.layers import Input, Add, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

class ModelRankingSensitivityAnalyzer:
    """
    Analyzes sensitivity of model rankings to number of seeds and trials
    """
    
    def __init__(self, data_path='data_updated/state_month_overdose_2015_2023.xlsx'):
        self.data_path = data_path
        
        # Create output directories
        self.output_dir = 'ranking_sensitivity_analysis'
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        for dir_path in [self.output_dir, self.results_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model configurations to test
        self.model_configs = {
            'lstm_small': {'model': 'lstm', 'lookback': 5, 'batch_size': 16, 'epochs': 50},
            'lstm_medium': {'model': 'lstm', 'lookback': 9, 'batch_size': 8, 'epochs': 100},
            'lstm_large': {'model': 'lstm', 'lookback': 12, 'batch_size': 32, 'epochs': 50},
            'tcn_small': {'model': 'tcn', 'lookback': 3, 'batch_size': 32, 'epochs': 50},
            'tcn_medium': {'model': 'tcn', 'lookback': 5, 'batch_size': 16, 'epochs': 100},
            'sarima_1': {'model': 'sarima', 'order': (1, 0, 0), 'seasonal_order': (1, 1, 1, 12)},
            'sarima_2': {'model': 'sarima', 'order': (1, 1, 1), 'seasonal_order': (2, 2, 2, 12)},
        }
        
        # Storage for results
        self.all_results = {}
        self.ranking_stability_results = {}
        
        print(f"Initialized Model Ranking Sensitivity Analyzer")
        print(f"Output directory: {self.output_dir}")
        print(f"Model configurations: {len(self.model_configs)}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the overdose data"""
        df = pd.read_excel(self.data_path)
        
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle different possible column names
        if 'Sum of Deaths' in df.columns:
            df = df.rename(columns={'Sum of Deaths': 'Deaths'})
        
        # Create proper date column
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
        
        # Handle Deaths column
        if df['Deaths'].dtype == 'object':
            df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else float(x))
        else:
            df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['Month', 'Deaths'])
        df = df[['Month', 'Deaths']].copy()
        df = df.sort_values('Month').reset_index(drop=True)
        
        print(f"✓ Final data shape: {df.shape}")
        print(f"✓ Date range: {df['Month'].min()} to {df['Month'].max()}")
        
        return df
    
    def create_train_val_split(self, df, val_start='2019-01-01', val_end='2020-01-01'):
        """Create train/validation splits for ranking evaluation"""
        train = df[df['Month'] < val_start]
        validation = df[(df['Month'] >= val_start) & (df['Month'] < val_end)]
        
        print(f"Train samples: {len(train)} ({train['Month'].min()} to {train['Month'].max()})")
        print(f"Validation samples: {len(validation)} ({validation['Month'].min()} to {validation['Month'].max()})")
        
        return train, validation
    
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
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    def run_lstm_trial(self, train_data, val_data, config, seed):
        """Run single LSTM trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        lookback = config['lookback']
        batch_size = config['batch_size']
        epochs = config['epochs']
        
        train_values = train_data['Deaths'].values
        val_values = val_data['Deaths'].values
        
        # Prepare training data
        X_train, y_train = self.create_dataset(train_values, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        
        # Build and train model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Generate validation predictions (autoregressive)
        current_input = train_values[-lookback:].reshape((1, lookback, 1))
        val_preds = []
        for _ in range(len(val_values)):
            pred = model.predict(current_input, verbose=0)[0][0]
            val_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return val_values, np.array(val_preds)
    
    def run_tcn_trial(self, train_data, val_data, config, seed):
        """Run single TCN trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        lookback = config['lookback']
        batch_size = config['batch_size']
        epochs = config['epochs']
        
        train_values = train_data['Deaths'].values
        val_values = val_data['Deaths'].values
        
        # Prepare training data
        X_train, y_train = self.create_dataset(train_values, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        
        # Build and train model
        model = Sequential([
            TCN(input_shape=(lookback, 1), dilations=[1, 2, 4, 8], 
                nb_filters=64, kernel_size=3, dropout_rate=0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Generate validation predictions
        current_input = train_values[-lookback:].reshape((1, lookback, 1))
        val_preds = []
        for _ in range(len(val_values)):
            pred = model.predict(current_input, verbose=0)[0][0]
            val_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return val_values, np.array(val_preds)
    
    def run_sarima_trial(self, train_data, val_data, config, seed):
        """Run single SARIMA trial"""
        np.random.seed(seed)
        
        train_series = train_data['Deaths'].astype(float)
        val_series = val_data['Deaths'].astype(float)
        
        try:
            model = SARIMAX(train_series, 
                            order=config['order'], 
                            seasonal_order=config['seasonal_order'],
                            enforce_stationarity=False, 
                            enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=50)
            
            val_predictions = results.predict(start=len(train_series), 
                                            end=len(train_series) + len(val_series) - 1).values
            
            return val_series.values, val_predictions
            
        except Exception as e:
            print(f"SARIMA failed: {e}")
            # Return simple forecast as fallback
            train_mean = train_series.mean()
            val_predictions = np.full_like(val_series, train_mean)
            return val_series.values, val_predictions
    
    def run_single_model_trial(self, train_data, val_data, config, seed):
        """Run a single trial for any model type"""
        model_type = config['model']
        
        if model_type == 'lstm':
            return self.run_lstm_trial(train_data, val_data, config, seed)
        elif model_type == 'tcn':
            return self.run_tcn_trial(train_data, val_data, config, seed)
        elif model_type == 'sarima':
            return self.run_sarima_trial(train_data, val_data, config, seed)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_trials_for_config(self, train_data, val_data, config_name, config, num_trials, num_seeds):
        """Run multiple trials for a single configuration"""
        all_metrics = []
        
        seeds_to_use = [42 + i * 1000 for i in range(num_seeds)]
        
        for seed_idx, base_seed in enumerate(seeds_to_use):
            for trial in range(num_trials):
                trial_seed = base_seed + trial
                
                try:
                    start_time = time.time()
                    val_true, val_pred = self.run_single_model_trial(train_data, val_data, config, trial_seed)
                    end_time = time.time()
                    
                    metrics = self.evaluate_metrics(val_true, val_pred)
                    metrics['config_name'] = config_name
                    metrics['seed_idx'] = seed_idx
                    metrics['trial'] = trial
                    metrics['trial_seed'] = trial_seed
                    metrics['compute_time'] = end_time - start_time
                    
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"Error in {config_name}, seed {seed_idx}, trial {trial}: {e}")
                    continue
        
        return all_metrics
    
    def calculate_model_rankings(self, metrics_df, metric='RMSE'):
        """Calculate model rankings based on a specific metric"""
        # Group by config and calculate mean performance
        config_performance = metrics_df.groupby('config_name')[metric].mean()
        
        # Rank models (lower is better for RMSE, MAE, MAPE)
        rankings = config_performance.rank(ascending=True).to_dict()
        
        return rankings
    
    def evaluate_ranking_stability(self, all_metrics, num_trials_range, num_seeds_range, metric='RMSE'):
        """Evaluate ranking stability across different (trials, seeds) combinations"""
        stability_results = {}
        
        metrics_df = pd.DataFrame(all_metrics)
        
        for num_trials in num_trials_range:
            for num_seeds in num_seeds_range:
                # Filter data to use only specified number of trials and seeds
                filtered_df = metrics_df[
                    (metrics_df['trial'] < num_trials) & 
                    (metrics_df['seed_idx'] < num_seeds)
                ]
                
                if len(filtered_df) == 0:
                    continue
                
                # Calculate rankings
                rankings = self.calculate_model_rankings(filtered_df, metric)
                
                # Calculate total compute time
                total_compute_time = filtered_df['compute_time'].sum()
                
                stability_results[(num_trials, num_seeds)] = {
                    'rankings': rankings,
                    'total_compute_time': total_compute_time,
                    'num_configs_evaluated': len(rankings),
                    'metric_used': metric
                }
        
        return stability_results
    
    def calculate_ranking_correlation(self, stability_results, reference_key=None):
        """Calculate ranking correlation across different (trials, seeds) combinations"""
        if reference_key is None:
            # Use the largest (trials, seeds) combination as reference
            reference_key = max(stability_results.keys(), key=lambda x: (x[0], x[1]))
        
        reference_rankings = stability_results[reference_key]['rankings']
        
        correlation_results = {}
        
        for key, result in stability_results.items():
            current_rankings = result['rankings']
            
            # Ensure same models are compared
            common_models = set(reference_rankings.keys()) & set(current_rankings.keys())
            if len(common_models) < 2:
                continue
            
            ref_ranks = [reference_rankings[model] for model in common_models]
            cur_ranks = [current_rankings[model] for model in common_models]
            
            # Calculate correlations
            spearman_corr, spearman_p = spearmanr(ref_ranks, cur_ranks)
            kendall_corr, kendall_p = kendalltau(ref_ranks, cur_ranks)
            
            correlation_results[key] = {
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'kendall_correlation': kendall_corr,
                'kendall_p_value': kendall_p,
                'num_models_compared': len(common_models),
                'total_compute_time': result['total_compute_time']
            }
        
        return correlation_results, reference_key
    
    def run_comprehensive_analysis(self, max_trials=50, max_seeds=10, eval_interval=5, metric='RMSE'):
        """Run comprehensive ranking sensitivity analysis"""
        print("="*80)
        print("COMPREHENSIVE MODEL RANKING SENSITIVITY ANALYSIS")
        print("="*80)
        
        # Load data
        print("Loading and preprocessing data...")
        data = self.load_and_preprocess_data()
        train_data, val_data = self.create_train_val_split(data)
        
        print(f"\nRunning trials for all model configurations...")
        print(f"Max trials: {max_trials}, Max seeds: {max_seeds}")
        print(f"Model configurations: {list(self.model_configs.keys())}")
        
        # Run trials for all configurations
        all_metrics = []
        total_configs = len(self.model_configs)
        
        for config_idx, (config_name, config) in enumerate(self.model_configs.items()):
            print(f"\nProcessing configuration {config_idx + 1}/{total_configs}: {config_name}")
            print(f"  Config: {config}")
            
            config_metrics = self.run_trials_for_config(
                train_data, val_data, config_name, config, max_trials, max_seeds
            )
            
            all_metrics.extend(config_metrics)
            print(f"  Completed {len(config_metrics)} trials")
        
        # Save all raw results
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(os.path.join(self.results_dir, 'all_trial_results.csv'), index=False)
        
        print(f"\nEvaluating ranking stability...")
        
        # Define evaluation ranges
        trials_range = list(range(eval_interval, max_trials + 1, eval_interval))
        seeds_range = list(range(1, max_seeds + 1))
        
        # Evaluate ranking stability
        stability_results = self.evaluate_ranking_stability(
            all_metrics, trials_range, seeds_range, metric
        )
        
        # Calculate ranking correlations
        correlation_results, reference_key = self.calculate_ranking_correlation(stability_results)
        
        # Save results
        with open(os.path.join(self.results_dir, 'stability_results.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for key, value in stability_results.items():
                serializable_results[f"{key[0]}_{key[1]}"] = {
                    'rankings': {k: float(v) for k, v in value['rankings'].items()},
                    'total_compute_time': float(value['total_compute_time']),
                    'num_configs_evaluated': int(value['num_configs_evaluated']),
                    'metric_used': value['metric_used']
                }
            json.dump(serializable_results, f, indent=2)
        
        with open(os.path.join(self.results_dir, 'correlation_results.json'), 'w') as f:
            serializable_corr = {}
            for key, value in correlation_results.items():
                serializable_corr[f"{key[0]}_{key[1]}"] = {
                    'spearman_correlation': float(value['spearman_correlation']),
                    'spearman_p_value': float(value['spearman_p_value']),
                    'kendall_correlation': float(value['kendall_correlation']),
                    'kendall_p_value': float(value['kendall_p_value']),
                    'num_models_compared': int(value['num_models_compared']),
                    'total_compute_time': float(value['total_compute_time'])
                }
            json.dump(serializable_corr, f, indent=2)
        
        # Store results for plotting
        self.stability_results = stability_results
        self.correlation_results = correlation_results
        self.reference_key = reference_key
        
        print(f"Analysis complete! Reference configuration: {reference_key}")
        
        return stability_results, correlation_results
    
    def create_ranking_stability_plots(self):
        """Create comprehensive plots for ranking stability analysis"""
        
        if not hasattr(self, 'correlation_results'):
            print("No correlation results found. Run analysis first.")
            return
        
        # Extract data for plotting
        trials_list = []
        seeds_list = []
        spearman_corr_list = []
        kendall_corr_list = []
        compute_times = []
        
        for (trials, seeds), result in self.correlation_results.items():
            trials_list.append(trials)
            seeds_list.append(seeds)
            spearman_corr_list.append(result['spearman_correlation'])
            kendall_corr_list.append(result['kendall_correlation'])
            compute_times.append(result['total_compute_time'])
        
        # Create DataFrame for easier plotting
        plot_data = pd.DataFrame({
            'trials': trials_list,
            'seeds': seeds_list,
            'spearman_correlation': spearman_corr_list,
            'kendall_correlation': kendall_corr_list,
            'compute_time': compute_times
        })
        
        # 1. Heatmap of Spearman correlations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Pivot data for heatmaps
        spearman_pivot = plot_data.pivot(index='seeds', columns='trials', values='spearman_correlation')
        kendall_pivot = plot_data.pivot(index='seeds', columns='trials', values='kendall_correlation')
        compute_pivot = plot_data.pivot(index='seeds', columns='trials', values='compute_time')
        
        # Spearman correlation heatmap
        sns.heatmap(spearman_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0,0], cbar_kws={'label': 'Spearman Correlation'})
        axes[0,0].set_title('Ranking Stability: Spearman Correlation')
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Number of Seeds')
        
        # Kendall correlation heatmap
        sns.heatmap(kendall_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0,1], cbar_kws={'label': 'Kendall Correlation'})
        axes[0,1].set_title('Ranking Stability: Kendall Correlation')
        axes[0,1].set_xlabel('Number of Trials')
        axes[0,1].set_ylabel('Number of Seeds')
        
        # Compute time heatmap
        sns.heatmap(compute_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[1,0], cbar_kws={'label': 'Total Compute Time (s)'})
        axes[1,0].set_title('Computational Cost')
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('Number of Seeds')
        
        # Efficiency plot (correlation vs compute time)
        scatter = axes[1,1].scatter(plot_data['compute_time'], plot_data['spearman_correlation'], 
                                  c=plot_data['trials'], s=plot_data['seeds']*10, 
                                  alpha=0.7, cmap='viridis')
        axes[1,1].set_xlabel('Total Compute Time (s)')
        axes[1,1].set_ylabel('Spearman Correlation')
        axes[1,1].set_title('Efficiency: Correlation vs Compute Time\n(Color=Trials, Size=Seeds)')
        
        # Add colorbar for scatter plot
        cbar = plt.colorbar(scatter, ax=axes[1,1])
        cbar.set_label('Number of Trials')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'ranking_stability_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Line plots showing convergence
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Group by seeds and plot trials vs correlation
        unique_seeds = sorted(plot_data['seeds'].unique())
        for seed_count in unique_seeds:
            seed_data = plot_data[plot_data['seeds'] == seed_count].sort_values('trials')
            axes[0,0].plot(seed_data['trials'], seed_data['spearman_correlation'], 
                          marker='o', label=f'{seed_count} seeds')
        
        axes[0,0].set_xlabel('Number of Trials')
        axes[0,0].set_ylabel('Spearman Correlation')
        axes[0,0].set_title('Ranking Stability vs Trials (by Seeds)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% threshold')
        
        # Group by trials and plot seeds vs correlation
        unique_trials = sorted(plot_data['trials'].unique())
        for trial_count in unique_trials:
            trial_data = plot_data[plot_data['trials'] == trial_count].sort_values('seeds')
            axes[0,1].plot(trial_data['seeds'], trial_data['spearman_correlation'], 
                          marker='s', label=f'{trial_count} trials')
        
        axes[0,1].set_xlabel('Number of Seeds')
        axes[0,1].set_ylabel('Spearman Correlation')
        axes[0,1].set_title('Ranking Stability vs Seeds (by Trials)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% threshold')
        
        # Compute time vs trials
        for seed_count in unique_seeds:
            seed_data = plot_data[plot_data['seeds'] == seed_count].sort_values('trials')
            axes[1,0].plot(seed_data['trials'], seed_data['compute_time'], 
                          marker='o', label=f'{seed_count} seeds')
        
        axes[1,0].set_xlabel('Number of Trials')
        axes[1,0].set_ylabel('Total Compute Time (s)')
        axes[1,0].set_title('Compute Time vs Trials (by Seeds)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Compute time vs seeds
        for trial_count in unique_trials:
            trial_data = plot_data[plot_data['trials'] == trial_count].sort_values('seeds')
            axes[1,1].plot(trial_data['seeds'], trial_data['compute_time'], 
                          marker='s', label=f'{trial_count} trials')
        
        axes[1,1].set_xlabel('Number of Seeds')
        axes[1,1].set_ylabel('Total Compute Time (s)')
        axes[1,1].set_title('Compute Time vs Seeds (by Trials)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'ranking_convergence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Optimal configuration identification
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Find configurations that meet stability threshold (e.g., correlation > 0.9)
        stability_threshold = 0.9
        stable_configs = plot_data[plot_data['spearman_correlation'] >= stability_threshold]
        
        if len(stable_configs) > 0:
            # Find the most efficient stable configuration (minimum compute time)
            optimal_config = stable_configs.loc[stable_configs['compute_time'].idxmin()]
            
            # Plot all configurations
            scatter = ax.scatter(plot_data['compute_time'], plot_data['spearman_correlation'], 
                               c=plot_data['trials'], s=plot_data['seeds']*20, 
                               alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
            
            # Highlight optimal configuration
            ax.scatter(optimal_config['compute_time'], optimal_config['spearman_correlation'], 
                      color='red', s=200, marker='*', edgecolors='black', linewidth=2,
                      label=f'Optimal: {int(optimal_config["trials"])} trials, {int(optimal_config["seeds"])} seeds')
            
            # Add threshold line
            ax.axhline(y=stability_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Stability threshold ({stability_threshold})')
            
            ax.set_xlabel('Total Compute Time (s)')
            ax.set_ylabel('Spearman Correlation')
            ax.set_title('Optimal Configuration Identification\n(Color=Trials, Size=Seeds)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Trials')
            
            print(f"\nOptimal configuration found:")
            print(f"  Trials: {int(optimal_config['trials'])}")
            print(f"  Seeds: {int(optimal_config['seeds'])}")
            print(f"  Spearman correlation: {optimal_config['spearman_correlation']:.3f}")
            print(f"  Compute time: {optimal_config['compute_time']:.1f} seconds")
        else:
            ax.text(0.5, 0.5, f'No configurations meet\nstability threshold ({stability_threshold})', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'optimal_configuration.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Ranking stability plots saved to {self.plots_dir}")
    
    def create_model_performance_comparison(self):
        """Create plots comparing model performance across different configurations"""
        
        if not hasattr(self, 'stability_results'):
            print("No stability results found. Run analysis first.")
            return
        
        # Extract model rankings for different (trials, seeds) combinations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Select a few key combinations to show
        key_combinations = [
            (min(k[0] for k in self.stability_results.keys()), 1),  # Min trials, 1 seed
            (max(k[0] for k in self.stability_results.keys()), 1),  # Max trials, 1 seed
            (min(k[0] for k in self.stability_results.keys()), max(k[1] for k in self.stability_results.keys())),  # Min trials, max seeds
            (max(k[0] for k in self.stability_results.keys()), max(k[1] for k in self.stability_results.keys()))   # Max trials, max seeds
        ]
        
        titles = [
            f"Min Trials ({key_combinations[0][0]}), 1 Seed",
            f"Max Trials ({key_combinations[1][0]}), 1 Seed", 
            f"Min Trials ({key_combinations[2][0]}), Max Seeds ({key_combinations[2][1]})",
            f"Max Trials ({key_combinations[3][0]}), Max Seeds ({key_combinations[3][1]})"
        ]
        
        for idx, (key, title) in enumerate(zip(key_combinations, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if key in self.stability_results:
                rankings = self.stability_results[key]['rankings']
                
                models = list(rankings.keys())
                ranks = list(rankings.values())
                
                # Create bar plot
                bars = ax.bar(range(len(models)), ranks, alpha=0.7)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel('Ranking (lower is better)')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
                # Color bars based on ranking
                for bar, rank in zip(bars, ranks):
                    if rank == 1:
                        bar.set_color('gold')
                    elif rank == 2:
                        bar.set_color('silver')
                    elif rank == 3:
                        bar.set_color('#CD7F32')  # bronze
                    else:
                        bar.set_color('lightblue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_rankings_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model performance comparison saved to {self.plots_dir}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        
        if not hasattr(self, 'correlation_results'):
            print("No results found. Run analysis first.")
            return
        
        # Find optimal configuration
        plot_data = []
        for (trials, seeds), result in self.correlation_results.items():
            plot_data.append({
                'trials': trials,
                'seeds': seeds,
                'spearman_correlation': result['spearman_correlation'],
                'compute_time': result['total_compute_time']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Find configurations meeting stability threshold
        stability_threshold = 0.9
        stable_configs = plot_df[plot_df['spearman_correlation'] >= stability_threshold]
        
        report = {
            'analysis_summary': {
                'total_configurations_tested': len(self.model_configs),
                'model_configurations': list(self.model_configs.keys()),
                'stability_threshold': stability_threshold,
                'reference_configuration': f"{self.reference_key[0]} trials, {self.reference_key[1]} seeds"
            },
            'stability_analysis': {
                'min_trials_tested': min(plot_df['trials']),
                'max_trials_tested': max(plot_df['trials']),
                'min_seeds_tested': min(plot_df['seeds']),
                'max_seeds_tested': max(plot_df['seeds']),
                'total_combinations_evaluated': len(plot_df)
            },
            'key_findings': {}
        }
        
        if len(stable_configs) > 0:
            optimal_config = stable_configs.loc[stable_configs['compute_time'].idxmin()]
            
            report['key_findings']['optimal_configuration'] = {
                'trials': int(optimal_config['trials']),
                'seeds': int(optimal_config['seeds']),
                'spearman_correlation': float(optimal_config['spearman_correlation']),
                'compute_time_seconds': float(optimal_config['compute_time']),
                'efficiency_score': float(optimal_config['spearman_correlation'] / optimal_config['compute_time'])
            }
            
            # Find minimum requirements for each dimension
            min_trials_stable = stable_configs['trials'].min()
            min_seeds_stable = stable_configs['seeds'].min()
            
            report['key_findings']['minimum_requirements'] = {
                'min_trials_for_stability': int(min_trials_stable),
                'min_seeds_for_stability': int(min_seeds_stable),
                'total_stable_configurations': len(stable_configs)
            }
        else:
            report['key_findings']['optimal_configuration'] = "No configuration meets stability threshold"
            report['key_findings']['minimum_requirements'] = "Consider increasing trials or seeds"
        
        # Add ranking stability analysis
        correlations = [result['spearman_correlation'] for result in self.correlation_results.values()]
        report['correlation_statistics'] = {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'min_correlation': float(np.min(correlations)),
            'max_correlation': float(np.max(correlations)),
            'correlations_above_threshold': int(sum(1 for c in correlations if c >= stability_threshold))
        }
        
        # Save report
        with open(os.path.join(self.results_dir, 'summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create readable summary
        summary_text = f"""
RANKING SENSITIVITY ANALYSIS SUMMARY REPORT
==========================================

Analysis Configuration:
- Model configurations tested: {len(self.model_configs)}
- Models: {', '.join(self.model_configs.keys())}
- Stability threshold: {stability_threshold} (Spearman correlation)
- Reference: {self.reference_key[0]} trials, {self.reference_key[1]} seeds

Key Findings:
"""
        
        if len(stable_configs) > 0:
            optimal = report['key_findings']['optimal_configuration']
            minimum = report['key_findings']['minimum_requirements']
            
            summary_text += f"""
✓ OPTIMAL CONFIGURATION FOUND:
  - Trials: {optimal['trials']}
  - Seeds: {optimal['seeds']}
  - Ranking stability: {optimal['spearman_correlation']:.3f}
  - Compute time: {optimal['compute_time_seconds']:.1f} seconds
  - Efficiency score: {optimal['efficiency_score']:.6f}

✓ MINIMUM REQUIREMENTS:
  - Minimum trials for stability: {minimum['min_trials_for_stability']}
  - Minimum seeds for stability: {minimum['min_seeds_for_stability']}
  - Total stable configurations: {minimum['total_stable_configurations']}
"""
        else:
            summary_text += f"""
⚠ NO OPTIMAL CONFIGURATION FOUND:
  - No combination of trials and seeds achieved {stability_threshold} correlation
  - Consider increasing the maximum number of trials or seeds
  - Current best correlation: {max(correlations):.3f}
"""
        
        corr_stats = report['correlation_statistics']
        summary_text += f"""
Correlation Statistics:
- Mean correlation: {corr_stats['mean_correlation']:.3f}
- Standard deviation: {corr_stats['std_correlation']:.3f}
- Range: {corr_stats['min_correlation']:.3f} - {corr_stats['max_correlation']:.3f}
- Configurations above threshold: {corr_stats['correlations_above_threshold']}/{len(correlations)}

Recommendations:
"""
        
        if len(stable_configs) > 0:
            optimal = report['key_findings']['optimal_configuration']
            summary_text += f"""
1. Use {optimal['trials']} trials and {optimal['seeds']} seeds for hyperparameter search
2. This configuration provides stable rankings with reasonable computational cost
3. Model rankings are reliable for selecting best hyperparameters
4. Can proceed with confidence to full grid search using these parameters
"""
        else:
            summary_text += f"""
1. Increase maximum trials and/or seeds in the analysis
2. Current analysis suggests rankings are not yet stable
3. Consider using the best available configuration: {plot_df.loc[plot_df['spearman_correlation'].idxmax(), 'trials']:.0f} trials, {plot_df.loc[plot_df['spearman_correlation'].idxmax(), 'seeds']:.0f} seeds
4. Monitor ranking stability in future hyperparameter searches
"""
        
        summary_text += f"""
Output Files:
- Raw results: {self.results_dir}/all_trial_results.csv
- Stability results: {self.results_dir}/stability_results.json
- Correlation results: {self.results_dir}/correlation_results.json
- Summary report: {self.results_dir}/summary_report.json
- Plots: {self.plots_dir}/
"""
        
        with open(os.path.join(self.output_dir, 'SUMMARY_REPORT.txt'), 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\nDetailed summary report saved to {self.output_dir}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Model Ranking Sensitivity Analysis')
    
    parser.add_argument('--max_trials', type=int, default=50,
                       help='Maximum number of trials to test (default: 50)')
    
    parser.add_argument('--max_seeds', type=int, default=10,
                       help='Maximum number of seeds to test (default: 10)')
    
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Evaluation interval for trials (default: 5)')
    
    parser.add_argument('--metric', type=str, default='RMSE',
                       choices=['RMSE', 'MAE', 'MAPE'],
                       help='Metric to use for ranking (default: RMSE)')
    
    parser.add_argument('--data_path', type=str, 
                       default='data_updated/state_month_overdose_2015_2023.xlsx',
                       help='Path to data file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_trials < args.eval_interval:
        print(f"Error: max_trials ({args.max_trials}) must be >= eval_interval ({args.eval_interval})")
        return
    
    if args.max_seeds < 1:
        print(f"Error: max_seeds ({args.max_seeds}) must be >= 1")
        return
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    print("="*80)
    print("MODEL RANKING SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Data Path: {args.data_path}")
    print(f"Max Trials: {args.max_trials}")
    print(f"Max Seeds: {args.max_seeds}")
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Ranking Metric: {args.metric}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize analyzer
    analyzer = ModelRankingSensitivityAnalyzer(args.data_path)
    
    try:
        # Run comprehensive analysis
        stability_results, correlation_results = analyzer.run_comprehensive_analysis(
            max_trials=args.max_trials,
            max_seeds=args.max_seeds,
            eval_interval=args.eval_interval,
            metric=args.metric
        )
        
        # Create plots
        print("\nGenerating plots...")
        analyzer.create_ranking_stability_plots()
        analyzer.create_model_performance_comparison()
        
        # Generate summary report
        print("\nGenerating summary report...")
        analyzer.generate_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {analyzer.output_dir}")
        print("Generated files:")
        print(f"  ├── results/")
        print(f"  │   ├── all_trial_results.csv")
        print(f"  │   ├── stability_results.json")
        print(f"  │   ├── correlation_results.json")
        print(f"  │   └── summary_report.json")
        print(f"  ├── plots/")
        print(f"  │   ├── ranking_stability_heatmaps.png")
        print(f"  │   ├── ranking_convergence_analysis.png")
        print(f"  │   ├── optimal_configuration.png")
        print(f"  │   └── model_rankings_comparison.png")
        print(f"  └── SUMMARY_REPORT.txt")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()