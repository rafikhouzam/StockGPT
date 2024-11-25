import torch
from torch.utils.data import DataLoader
from model.stock_gpt import StockGPT
from data.dataset import StockDataset
from training.train import train_model_with_early_stopping
from prediction.predict import predict_next_n_days
from data.preprocessing import prepare_data

if __name__ == "__main__":
    # Set device to MPS for MacBook with M1 or CUDA for other GPU options
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data
    start_date = "1970-01-01"
    end_date = "2024-10-01"
    tickers = [
    'AAPL',   # Apple Inc.
    'MSFT',   # Microsoft Corporation
    'GOOGL',  # Alphabet Inc. (Google)
    'AMZN',   # Amazon.com, Inc.
    'TSLA',   # Tesla, Inc.
    'NVDA',   # NVIDIA Corporation
    'META',   # Meta Platforms, Inc. (Facebook)
    'JPM',    # JPMorgan Chase & Co.
    'V',      # Visa Inc.
    'JNJ',    # Johnson & Johnson
    'UNH',    # UnitedHealth Group Incorporated
    'PG',     # Procter & Gamble Co.
    'DIS',    # The Walt Disney Company
    'MA',     # Mastercard Incorporated
    'HD',     # The Home Depot, Inc.
    'VZ',     # Verizon Communications Inc.
    'KO',     # The Coca-Cola Company
    'PEP',    # PepsiCo, Inc.
    'XOM',    # Exxon Mobil Corporation
    'NFLX'    # Netflix, Inc.
]

    discretized_data = prepare_data(tickers, start_date, end_date)

    # Dataset and Data Loaders
    seq_length = 20
    dataset = StockDataset(discretized_data, seq_length)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Instantiate model
    model = StockGPT().to(device)

    # Train the model
    train_model_with_early_stopping(model, train_loader, val_loader, device, epochs=10, patience=5)

    # Predict the next 10 days
    input_sequence = discretized_data[-seq_length:]
    predictions = predict_next_n_days(model, input_sequence, device, n=10)
    print("Predicted Tokens for the Next 10 Days:", predictions)
