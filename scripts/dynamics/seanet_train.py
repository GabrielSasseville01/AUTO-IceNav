import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
from typing import Optional
import numpy as np
from seanet import upsample_dataset, SeaCurrentRNN

print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Load dataset
with open("ECCO4.pkl", "rb") as f:
    # (299, 4, 80, 140)
    # (T, (U, V, DU, DV), H, W
    raw_dataset = pickle.load(f)
    # Convert JAX array to numpy, then to PyTorch
    raw_dataset = np.array(raw_dataset) if not isinstance(raw_dataset, np.ndarray) else raw_dataset
    dataset = torch.from_numpy(raw_dataset[:, :2, :, :]).float()
    f.close()

# Set random seed
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

series = upsample_dataset(dataset, factor=30)  # T, C, H, W
series = series.permute(0, 2, 3, 1)  # T, H, W, C
input_series = series[:-1]
target_series = series[1:]

print(f"inputs shape {input_series.shape}, targets shape {target_series.shape}")

# Initialization
hidden_size = 128
input_height, input_width = input_series.shape[1], input_series.shape[2]
model = SeaCurrentRNN(hidden_size=hidden_size, input_height=input_height, input_width=input_width)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 128
batch_size = 128
nun_batches = math.ceil(input_series.shape[0] / batch_size)
for epoch in range(1, num_epochs + 1):

    epoch_loss = []
    for batch_idx in range(nun_batches):
        batch_X = input_series[batch_idx * batch_size : (batch_idx+1) * batch_size].to(device)
        batch_Y = target_series[batch_idx * batch_size : (batch_idx+1) * batch_size].to(device)

        preds = model(batch_X)
        # preds shape: (T, H, W, C), targets shape: (T, H, W, C)
        loss = torch.mean((preds - batch_Y) ** 2)

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {epoch:03d} | loss {torch.tensor(epoch_loss).mean().item():.6f}")
    torch.save(model.state_dict(), "seanet.pth")

# Final evaluation
with torch.no_grad():
    preds = model(input_series)
    mae = torch.mean(torch.abs(preds - target_series))
    print(f"final MAE: {float(mae.item()):.6f}")
    print(f"Output shape: {preds.shape}, Target shape: {target_series.shape}")
    print("Last timestep prediction (first 5 spatial locations, both channels):")
    print(preds[-1, :5, :5, :])
    print("Last timestep target (first 5 spatial locations, both channels):")
    print(target_series[-1, :5, :5, :])

# Save model
torch.save(model.state_dict(), "seanet.pth")
