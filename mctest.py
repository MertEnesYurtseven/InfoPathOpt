#!/usr/bin/env python
"""
Script 2: Deployment of the Multi–Channel Reservoir in a Time Series Network with Channel Pooling

This script loads a pre-trained multi–channel reservoir weight tensor
(saved as "W2_tensor_multi.npy") and integrates it into a network for a time series task.
After running T reservoir updates, the reservoir state is pooled along the channel dimension
(via average pooling) before being flattened and passed to an MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

#######################################
# Global Parameters (must match Script 1)
#######################################
n = 16              # Grid side length (should match training)
r = 2               # Manhattan radius
channels = 4       # Number of channels (features) per neuron
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################
# Locally-Connected Multi-Channel Update
#######################################
def multi_channel_reservoir_update(S, W_tensor, r):
    """
    Performs the multi–channel locally–connected update.
    
    Args:
        S (Tensor): Reservoir state of shape [batch, channels, n, n].
        W_tensor (Tensor): Weight tensor of shape [n, n, channels, channels, 2*r+1, 2*r+1].
        r (int): Receptive field radius.
    
    Returns:
        Tensor: Updated reservoir state of shape [batch, channels, n, n].
    """
    batch, C, n1, n2 = S.shape  # Expect n1 == n2 == n and C == channels.
    kernel_size = 2 * r + 1

    # Unfold S to extract local patches.
    # Resulting shape: [batch, C * (kernel_size^2), n*n]
    patches = F.unfold(S, kernel_size=(kernel_size, kernel_size), padding=r)
    # Reshape into [batch, n*n, C, kernel_size^2]
    patches = patches.view(batch, C, kernel_size * kernel_size, n*n)
    patches = patches.permute(0, 3, 1, 2)  # [batch, n*n, C, kernel_size^2]
    # Reshape to [batch, n*n, C, kernel_size, kernel_size]
    patches = patches.view(batch, n*n, C, kernel_size, kernel_size)
    
    # Flatten the spatial weight tensor:
    # W_tensor has shape [n, n, channels, channels, kernel_size, kernel_size]
    # We reshape it to [n*n, channels, channels, kernel_size, kernel_size]
    W_flat = W_tensor.view(n*n, channels, channels, kernel_size, kernel_size)
    
    # For each spatial location l, compute:
    # out[b, l, oc] = sum_{ic, m, n} patches[b, l, ic, m, n] * W_flat[l, oc, ic, m, n]
    out = torch.einsum('blimn, l o i m n -> b l o', patches, W_flat)  # [batch, n*n, channels]
    
    # Reshape back to [batch, channels, n, n]
    out = out.view(batch, n, n, channels).permute(0, 3, 1, 2)
    return out

#######################################
# ReservoirLayer: Wraps the Multi–Channel Reservoir Update with Channel Pooling
#######################################
class ReservoirLayer(nn.Module):
    """
    A reservoir layer that:
      1. Projects an input vector (of length input_dim) into a spatial grid with
         shape [batch, channels, n, n] (by padding or projecting).
      2. Runs T reservoir update steps using the multi–channel locally–connected update.
      3. Pools over the channel dimension (here via average pooling) to reduce the state
         from [batch, channels, n, n] to [batch, n, n].
      4. Flattens the pooled state to a vector of length (n*n) for further processing.
    """
    def __init__(self, input_dim, reservoir_n, reservoir_d, reservoir_r, T, W_numpy_tensor):
        super().__init__()
        self.reservoir_n = reservoir_n   # n
        self.reservoir_d = reservoir_d   # usually 2
        self.reservoir_r = reservoir_r   # r
        self.T = T                       # number of reservoir update steps
        self.n = reservoir_n
        self.channels = channels         # number of channels
        N = reservoir_n ** reservoir_d   # total number of spatial locations

        # Convert the pre-trained weight tensor to a torch parameter.
        W_tensor = torch.from_numpy(W_numpy_tensor).float()
        expected_shape = (reservoir_n, reservoir_n, channels, channels, 2*reservoir_r+1, 2*reservoir_r+1)
        assert W_tensor.shape == expected_shape, f"Expected W shape {expected_shape}, got {W_tensor.shape}"
        self.W = nn.Parameter(W_tensor, requires_grad=True)
        # Create a connectivity mask so that during training only the originally–nonzero weights are updated.
        self.connectivity_mask = (W_tensor != 0).float().to(device)
        self.W.register_hook(lambda grad: grad * self.connectivity_mask)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch, input_dim]. It is assumed that input_dim <= n*n*channels.
        
        Returns:
            Tensor: Flattened pooled reservoir state of shape [batch, n*n].
        """
        batch = x.size(0)
        N = self.n ** self.reservoir_d
        if x.dim() == 2:
            # Pad x to length N * channels if necessary.
            padded = F.pad(x, (0, N * self.channels - x.size(1)))
            S = padded.view(batch, self.channels, self.n, self.n)
        else:
            S = x  # Already in shape [batch, channels, n, n].
        for _ in range(self.T):
            S = multi_channel_reservoir_update(S, self.W, self.reservoir_r)
            S = F.leaky_relu(S, negative_slope=0.1)
        # Pool over the channels: e.g. average pooling.
        S_pooled = S.mean(dim=1)  # Shape becomes [batch, n, n]
        return S_pooled.view(batch, -1)  # Flatten to [batch, n*n]

#######################################
# ReservoirNetwork: Stacks ReservoirLayer(s) and an MLP head.
#######################################
class ReservoirNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, reservoir_layers, mlp_hidden_dims):
        super().__init__()
        self.reservoir_layers = nn.ModuleList(reservoir_layers)
        mlp_layers = []
        # The final reservoir layer now outputs a flattened vector of length n*n (channels have been pooled)
        prev_dim = reservoir_layers[-1].n ** reservoir_layers[-1].reservoir_d
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        for layer in self.reservoir_layers:
            x = layer(x)
        return self.mlp(x)

#######################################
# A Simple MLP for Comparison
#######################################
class MLP(nn.Module):
    def __init__(self, input_dim=18, output_dim=18):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 324),
            nn.ReLU(),
            nn.Linear(324, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.model(x)

#######################################
# Utility: Create a Time Series Dataset from a DataFrame
#######################################
def create_time_series_dataset(df, feature_column, window_size):
    """
    Given a DataFrame of time series data, this function creates a supervised dataset.
    Each sample consists of a window of length 'window_size' from the specified feature column,
    and the target is the next value.
    """
    X, y = [], []
    df['prev_close'] = df['Close'].shift(1)
    df[feature_column] = ((df['Close'] - df['prev_close']) / df['prev_close'])
    for i in range(window_size, len(df) - window_size):
        window = df[feature_column].iloc[i : i + window_size].values
        target = df[feature_column].iloc[i + window_size]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

#######################################
# Main: Load Pre-Trained Weights & Train the Time Series Network
#######################################
if __name__ == "__main__":
    # Load the pre-trained multi-channel weight tensor.
    W2_tensor = np.load('W2_tensor_multi.npy')  # Expected shape: [n, n, channels, channels, 2*r+1, 2*r+1]
    print(f"W2_tensor shape: {W2_tensor.shape}")
    reservoir_n = int(W2_tensor.shape[0])  # Should be 20 based on training
    window = reservoir_n  # Set the time series window size equal to the grid side length

    # Load time series data.
    df = pd.read_csv('BTC-EUR.csv').dropna()
    X, y = create_time_series_dataset(df, 'Close', window)
    
    # Create PyTorch datasets.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Compute standard deviation of targets (for monitoring).
    real_std = np.std(y[train_size:])
    
    # Create reservoir layers.
    reservoir_layers = [
        ReservoirLayer(input_dim=window, reservoir_n=reservoir_n, reservoir_d=2, 
                       reservoir_r=r, T=window//2, W_numpy_tensor=W2_tensor)
    ]
    
    # Create the full network (Reservoir + MLP head) and also a baseline MLP.
    reservoir_model = ReservoirNetwork(
        input_dim=window,
        output_dim=1,
        reservoir_layers=reservoir_layers,
        mlp_hidden_dims=[128, 64]
    ).to(device)
    
    mlp_model = MLP(input_dim=window, output_dim=1).to(device)
    
    criterion = nn.MSELoss()
    
    def train_model(model, name, num_epochs=50):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_losses = []
        test_losses = []
        output_stds = []
        best_outputs = []
        best_targets = []
        best_test_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            outputs_list = []
            targets_list = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
                outputs_list.extend(outputs.detach().cpu().numpy().flatten())
                targets_list.extend(targets.detach().cpu().numpy().flatten())
            train_loss = epoch_loss / len(train_dataset)
            train_losses.append(train_loss)
            output_std = np.std(outputs_list)
            output_stds.append(output_std)
            
            model.eval()
            test_loss = 0
            epoch_outputs = []
            epoch_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item() * inputs.size(0)
                    epoch_outputs.extend(outputs.cpu().numpy().flatten())
                    epoch_targets.extend(targets.cpu().numpy().flatten())
            test_loss = test_loss / len(test_dataset)
            test_losses.append(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_outputs = epoch_outputs
                best_targets = epoch_targets
                
            print(f"{name} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4e} | Test Loss: {test_loss:.4e}")
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'output_stds': output_stds,
            'best_outputs': best_outputs,
            'best_targets': best_targets
        }
    
    # Train both models.
    reservoir_results = train_model(reservoir_model, "Reservoir", num_epochs=50)
    mlp_results = train_model(mlp_model, "MLP", num_epochs=50)
    
    # Plot results.
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(reservoir_results['train_losses'], label='Reservoir Train')
    plt.plot(reservoir_results['test_losses'], label='Reservoir Test')
    plt.plot(mlp_results['train_losses'], label='MLP Train')
    plt.plot(mlp_results['test_losses'], label='MLP Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.plot(reservoir_results['output_stds'], label='Reservoir Output')
    plt.plot(mlp_results['output_stds'], label='MLP Output')
    plt.axhline(real_std, color='r', linestyle='--', label='Real Target')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.title('Output Standard Deviation')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(reservoir_results['best_targets'], bins=50, alpha=0.7, label='Real Targets', color='blue')
    plt.title('Real Targets')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(reservoir_results['best_outputs'], bins=50, alpha=0.7, label='Reservoir Outputs', color='green')
    plt.title('Reservoir Outputs')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(mlp_results['best_outputs'], bins=50, alpha=0.7, label='MLP Outputs', color='orange')
    plt.title('MLP Outputs')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(reservoir_results['best_targets'], bins=50, alpha=0.7, label='Real Targets', color='blue')
    plt.title('Real Targets')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
