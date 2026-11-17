def compute_signal(history):
    return {symbol: (prices[-1] / prices[-21] - 1.0) if len(prices) > 20 else 0.0
            for symbol, prices in history.items()}
