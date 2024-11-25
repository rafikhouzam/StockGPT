import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model.stock_gpt import StockGPT

# Label smoothing cross entropy
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

# Training function with early stopping
def train_model_with_early_stopping(model, train_loader, val_loader, device, epochs=10, lr=1e-3, patience=5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=2000, mode='triangular')
    criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.15)  # Increased label smoothing

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                logits = logits[:, -1, :]
                loss = criterion(logits, y)
                loss.backward()

                # Add gradient clipping here
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()  # Use a cyclic learning rate
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate_model_with_logging(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break


# Evaluation function with logging
def evaluate_model_with_logging(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    token_predictions = np.zeros(402)  # Assuming 402 possible tokens in total

    with tqdm(data_loader, desc="Evaluating", unit="batch") as pbar:
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                logits = logits[:, -1, :]  # Predict only the next token
                loss = criterion(logits, y)
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                # Log the predicted tokens
                predicted_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
                for token in predicted_tokens:
                    token_predictions[token] += 1

    avg_val_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Token Predictions Distribution: {token_predictions}")

    return avg_val_loss
