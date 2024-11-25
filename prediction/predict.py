import torch
from tqdm import tqdm

def predict_next_n_days(model, input_sequence, device, n=10):
    model.eval()
    current_sequence = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(n):
            logits = model(current_sequence)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            predictions.append(next_token.item())
            current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)[:, -20:]

    return predictions