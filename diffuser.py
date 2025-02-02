# assume we have preoptimized W matrices
# network architecture follows:
# for ni dimensional vector input, we have n d=2,r pretrained liquid network in square mesh structure
#diffuse ni to n, meaning fill start of the square structure (first n, ni=n or project ni to n)
'''
state = 
[0,0,0
 0,0,0
 0,0,0]

if the input is 3 dimensional, <x,y,z>
initial_state = 
[x,y,z
 0,0,0
 0,0,0]
'''
#run resarvoir n steps
#flatten and feed trough an mlp
#optionally you can get the states and diffuse them to a bigger resarvoir and so on
#assume you read the weight matrices as numpy arrays.
# we run the resarvoir like:
'''
 S = S0  # list of [batch_size, N]
    for _ in range(T):
        S_prev = states_list[-1]  # [batch_size, N]
        S_next = S_prev @ W#F.leaky_relu(, negative_slope=0.1)
        states_list.append(S_next)
    return states_list[-1]
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class ReservoirLayer(nn.Module):
    """A single reservoir layer with pre-trained weights and optional input projection."""
    def __init__(self, input_dim, reservoir_n, reservoir_d, reservoir_r, T, W_numpy):
        super().__init__()
        self.reservoir_n = reservoir_n
        self.reservoir_d = reservoir_d
        self.reservoir_r = reservoir_r
        self.T = T  # Number of reservoir update steps
        self.N = reservoir_n ** reservoir_d  # Reservoir size (n^d)

        # Convert numpy weight matrix to PyTorch buffer
        W_tensor = torch.from_numpy(W_numpy).float()
        assert W_tensor.shape == (self.N, self.N), \
            f"Expected W shape {(self.N, self.N)}, got {W_tensor.shape}"
        self.W =nn.Parameter(W_tensor, requires_grad=True)  # Make W a trainable parameter

        # Create connectivity mask
        self.connectivity_mask = (torch.tensor(W_numpy) != 0).float().to('cuda')

        # Register backward hook to modify gradients
        self.W.register_hook(lambda grad: grad * self.connectivity_mask)



        
    def forward(self, x):
        """Process input through the reservoir.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Final reservoir state of shape [batch_size, N]
        """

        S = F.pad(x, (0, self.N - x.size(1))) 
        
        # Run reservoir updates
        for _ in range(self.T):
            S = F.leaky_relu(S @ self.W, negative_slope=0.1)  # Reservoir state update
            
        return S#[:,-self.reservoir_n:]

class ReservoirNetwork(nn.Module):
    """A network composed of multiple reservoir layers followed by an MLP head."""
    def __init__(self, input_dim, output_dim, reservoir_layers, mlp_hidden_dims):
        super().__init__()
        self.reservoir_layers = nn.ModuleList(reservoir_layers)
        

            
        mlp_layers = []
        prev_dim = reservoir_layers[-1].N
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        """Process input through the entire network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Pass through reservoir layers
        for layer in self.reservoir_layers:
            x = layer(x)
        
        # Pass through MLP head
        return self.mlp(x)

# Example usage --------------------------------------------------------------


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
from thop import profile


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from synthdata import SyntheticTransformDataset
def create_time_series_dataset(df, feature_column, window_size):
    """
    Convert a time series DataFrame into a supervised learning dataset.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing the time series.
      feature_column (str): The column name to use for features/target.
      window_size (int): Number of time steps to include in each feature window.
    
    Returns:
      X (np.array): 2D array where each row is a window of features.
      y (np.array): 1D array of target values.
    """
    X, y = [], []
    df['prev_close'] = df['Close'].shift(1)

    # Calculate the percentage change from the previous close to the current close
    df[feature_column] = ((df['Close'] - df['prev_close']) / df['prev_close'])

    # Loop over the DataFrame such that we have enough data for the window and target
    for i in range(window_size,len(df) - window_size):
        window = df[feature_column].iloc[i : i + window_size].values
        target = df[feature_column].iloc[i + window_size]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # 1. Load pre-trained weight matrices (example)
    W1_numpy = np.load('W1.npy')  # 10x10 grid (n=10, d=2)
    W2_numpy = np.load('W2.npy')  # 10x10 grid (n=10, d=2)
    window = int((W2_numpy.shape[0])**0.5)
    X,y = create_time_series_dataset(pd.read_csv('BTC-EUR.csv').dropna(),'Close',window)
    print(W2_numpy.shape)
    
    #W1_numpy = (np.random.rand(100, 100)-0.5)*2

    # 2. Create reservoir layers
    reservoir_layers = [
        ReservoirLayer(input_dim=window, reservoir_n=window, reservoir_d=2, 
                      reservoir_r=2, T=window//2, W_numpy=W2_numpy),
    ]

    # 3. Create full network
    # model = ReservoirNetwork(
    #     input_dim=window, 
    #     output_dim=1,
    #     reservoir_layers=reservoir_layers,
    #     mlp_hidden_dims=[128, 64]
    # ).to('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim=window,output_dim=1).to('cuda' if torch.cuda.is_available() else 'cpu')

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a TensorDataset from X and y
    dataset = TensorDataset(X_tensor, y_tensor)

    # Determine the split index (80% training, 20% testing)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    # Alternatively, if you want to avoid random splitting for time series,
    # simply slice the dataset rather than using random_split.
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset  = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    # Create DataLoaders
    # For training, you might want to shuffle the data.
    batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # For testing, do not shuffle as you want to preserve the time order.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Create dataloaders
    
    # 7. Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50

    # 8. Training loop
    best_test_loss = float('inf')
    device = next(model.parameters()).device
    print(device)

    import torch
    import torch
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt

    # Load data and create datasets
    W2_numpy = np.load('W2.npy')
    window = int(W2_numpy.shape[0]**0.5)
    df = pd.read_csv('BTC-EUR.csv').dropna()
    X, y = create_time_series_dataset(df, 'Close', window)
    
    # Create datasets
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate real target std once
    real_std = np.std(y[train_size:])
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    reservoir_layers = [
        ReservoirLayer(window, window, 2, 2, window//2, W2_numpy)
    ]
    
    reservoir_model = ReservoirNetwork(
        input_dim=window,
        output_dim=1,
        reservoir_layers=reservoir_layers,
        mlp_hidden_dims=[128, 64]
    ).to(device)
    
    mlp_model = MLP(input_dim=window, output_dim=1).to(device)
    
    # Training function
    def train_model(model, name):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        train_losses = []
        test_losses = []
        output_stds = []
        best_outputs = []
        best_targets = []
        best_test_loss = float('inf')
        
        for epoch in range(50):
            # Training
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
            
            # Validation
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
            
            # Save best outputs
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_outputs = epoch_outputs
                best_targets = epoch_targets
            
            print(f"{name} Epoch {epoch+1}/50 | Train Loss: {train_loss:.4e} | Test Loss: {test_loss:.4e}")

        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'output_stds': output_stds,
            'best_outputs': best_outputs,
            'best_targets': best_targets
        }

    # Train both models
    reservoir_results = train_model(reservoir_model, "Reservoir")
    mlp_results = train_model(mlp_model, "MLP")
    
    # Plotting
    plt.figure(figsize=(18, 5))
    
    # Loss plot
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
    
    # STD plot
    plt.subplot(1, 3, 2)
    plt.plot(reservoir_results['output_stds'], label='Reservoir Output')
    plt.plot(mlp_results['output_stds'], label='MLP Output')
    plt.axhline(real_std, color='r', linestyle='--', label='Real Target')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.title('Output Standard Deviation')
    plt.legend()
    
    # Histogram plot (side by side)
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