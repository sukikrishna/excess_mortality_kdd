# Optimal hyperparameters from grid search results
OPTIMAL_PARAMS = {
    'lstm': {'lookback': 5, 'batch_size': 8, 'epochs': 50},
    'seq2seq': {'lookback': 7, 'batch_size': 16, 'epochs': 100, 'encoder_units': 64, 'decoder_units': 64},
    'seq2seq_attn': {'lookback': 5, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 64},
    'tcn': {'lookback': 7, 'batch_size': 8, 'epochs': 100},
    'transformer': {'lookback': 7, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2},
    'sarima': {'order': (1, 0, 0), 'seasonal_order': (1, 1, 1, 12)},
}
