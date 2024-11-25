import yfinance as yf
import numpy as np
import pandas as pd

class StockDataPreprocessor:
    def __init__(self, bin_size=50, min_return=-10000, max_return=10000, noise_level=0.05):
        self.bin_size = bin_size
        self.min_return = min_return
        self.max_return = max_return
        self.bins = np.arange(min_return, max_return + bin_size, bin_size)
        self.noise_level = noise_level

    def discretize(self, returns):
        # Add random noise to the returns for data augmentation
        noise = np.random.normal(0, self.noise_level, size=returns.shape)
        noisy_returns = returns + noise
        # Discretize with noise
        bins_indices = np.digitize(noisy_returns, self.bins) - 1
        return bins_indices

    def process_data(self, stock_data):
        returns = stock_data.pct_change().dropna() * 10000  # Convert to basis points
        discretized_returns = self.discretize(returns)
        return discretized_returns


def get_stock_data(tickers, start, end):
    # Use yfinance to download stock data for multiple tickers
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def prepare_data(tickers, start, end):
    # Fetch the stock data for all tickers
    stock_data = get_stock_data(tickers, start, end)
    
    preprocessor = StockDataPreprocessor()
    all_discretized_data = []

    # Process each ticker separately
    for ticker in stock_data.columns:
        ticker_data = stock_data[ticker].dropna()
        discretized_data = preprocessor.process_data(ticker_data)
        all_discretized_data.extend(discretized_data)

    all_discretized_data = np.array(all_discretized_data)

    # Balance the dataset to avoid overrepresentation of frequent tokens
    token_counts = np.bincount(all_discretized_data, minlength=402)  # Assuming 402 possible tokens
    target_count = int(np.median(token_counts[token_counts > 0]))

    balanced_data = []
    for token, count in enumerate(token_counts):
        indices = np.where(all_discretized_data == token)[0]

        if count > 0:
            if count > target_count:
                # Undersample if token is overrepresented
                selected_indices = np.random.choice(indices, min(target_count, len(indices)), replace=False)
            else:
                # Oversample if token is underrepresented
                selected_indices = np.random.choice(indices, target_count, replace=True)

            balanced_data.extend(all_discretized_data[selected_indices])

    return np.array(balanced_data)

# Example usage to get the preprocessed data for multiple tickers
if __name__ == "__main__":
    start_date = "20cdc20-01-01"
    end_date = "2022-01-01"
    tickers = ["AAPL", "MSFT", "GOOG"]  # Example list of tickers
    discretized_data = prepare_data(tickers, start_date, end_date)
