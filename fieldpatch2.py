import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # For Ampere GPUs
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from line_profiler import profile
###############################################################################
# 1) Coordinate & Field Utilities
###############################################################################
@profile
def index_to_coords(idx: int, n: int, d: int):
    """
    Convert a 1D index (idx) into dD coordinates in [0, n-1].
    """
    coords = []
    tmp = idx
    for _ in range(d):
        coords.append(tmp % n)
        tmp //= n
    return coords
@profile
def coords_to_index(coords, n: int):
    """
    Convert dD coordinates back into a single index in [0..n^d - 1].
    """
    d = len(coords)
    idx = 0
    base = 1
    for c in coords:
        idx += c * base
        base *= n
    return idx
@profile
def get_coords_tensor(n, d, device='cpu'):
    """
    Create a [N, d] tensor of coordinates for all N = n^d neurons.
    """
    N = n**d
    all_coords = [index_to_coords(i, n, d) for i in range(N)]
    coords_tensor = torch.tensor(all_coords, device=device, dtype=torch.long)
    return coords_tensor
@profile
def get_local_field(center_idx, r, coords_tensor, n, d):
    """
    Return a list of neuron indices that are within
    Manhattan distance <= r of the center_idx neuron.
    """
    c_i = coords_tensor[center_idx]
    dist = torch.sum(torch.abs(coords_tensor - c_i), dim=1)
    mask = (dist <= r)
    field_indices = mask.nonzero(as_tuple=True)[0].tolist()
    return field_indices
@profile
def get_centers_in_range(center_idx, dist_lo, dist_hi, coords_tensor):
    """
    Return a list of neuron indices whose L1 distance from center_idx
    is in [dist_lo, dist_hi].
    """
    c_i = coords_tensor[center_idx]
    dist = torch.sum(torch.abs(coords_tensor - c_i), dim=1)
    mask = (dist >= dist_lo) & (dist <= dist_hi)
    candidates = mask.nonzero(as_tuple=True)[0].tolist()
    return candidates
@profile
def create_weight_matrix_naive(n: int, d: int, r: int, device='cpu'):
    """
    Create an N x N weight matrix (N = n^d) for a d-dimensional grid,
    with Manhattan distance <= r, randomly initialized from N(0,1).
    """
    N = n**d
    W = torch.zeros((N, N), device=device)
    coords_tensor = get_coords_tensor(n, d, device=device)
    for i in range(N):
        c_i = coords_tensor[i]
        dist = torch.sum(torch.abs(coords_tensor - c_i), dim=1)
        mask = (dist <= r)
        W[i, mask] = torch.randn(mask.sum(), device=device)
    return W

###############################################################################
# 2) Reservoir Update
###############################################################################
@profile
def run_reservoir_batch(W, S0, T=5):
    """
    Recurrently update S_{t+1} = tanh(W @ S_t), starting from S0.
    Supports batch processing.

    Args:
        W (Tensor): Weight matrix of shape [N, N].
        S0 (Tensor): Initial states of shape [batch_size, N].
        T (int): Number of time steps.

    Returns:
        Tensor: Reservoir states of shape [T+1, batch_size, N].
    """
    states_list = [S0]  # list of [batch_size, N]
    for _ in range(T):
        S_prev = states_list[-1]  # [batch_size, N]
        #S_next = torch.tanh(S_prev @ W.t())  # [batch_size, N]
        S_next = S_prev @ W#F.leaky_relu(, negative_slope=0.1)
        states_list.append(S_next)
    return torch.stack(states_list, dim=0)  # [T+1, batch_size, N]

###############################################################################
# 3) ScoreNet for InfoNCE
###############################################################################

class ScoreNet(nn.Module):
    """
    A simple feed-forward network that outputs a scalar given cat([x,y]).
    """
    def __init__(self, input_dim=2, hidden_dim=128):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        h = F.relu(self.fc1(x))
        hh = F.relu(self.fc2(h))
        out = self.fc3(hh)
        return out

from tqdm import tqdm
@profile
def infoNCE_loss(anchor, pos, neg, score_net, batch_size=512):

    device = anchor.device
    B, d = anchor.shape
    K_neg = neg.shape[1]
    total_loss = 0.0

    # Process in chunks to avoid OOM
    for i in range(0, B, batch_size):
        # Extract chunk for current batch
        anchor_chunk = anchor[i:i+batch_size]
        pos_chunk = pos[i:i+batch_size]
        neg_chunk = neg[i:i+batch_size]
        chunk_size = anchor_chunk.size(0)

        # 1) Positive scores [chunk_size]
        pos_score = score_net(torch.cat([anchor_chunk, pos_chunk], dim=1)).squeeze(1)


        neg_flat = neg_chunk.view(-1, d)  # [chunk_size*K_neg, d]
        anchor_repeat = anchor_chunk.unsqueeze(1).repeat(1, K_neg, 1).view(-1, d)
        neg_score = score_net(torch.cat([anchor_repeat, neg_flat], dim=1)).squeeze(1)

        # 3) Logsumexp and loss for this chunk
        pos_score_exp = pos_score.unsqueeze(1)  # [chunk_size, 1]
        neg_score_exp = neg_score.view(chunk_size, K_neg)  # [chunk_size, K_neg]
        all_scores = torch.cat([pos_score_exp, neg_score_exp], dim=1)  # [chunk_size, 1+K_neg]
        logsumexp = torch.logsumexp(all_scores, dim=1)  # [chunk_size]
        chunk_loss = - (pos_score - logsumexp).mean()  # Mean over chunk

        # Weight by chunk proportion of total samples
        total_loss += chunk_loss * (chunk_size / B)

    return total_loss
###############################################################################
# 5) Building Anchor, Pos, Neg from A, B, C
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ... [Previous utility functions remain unchanged] ...
@profile
def precompute_fields(n, d, r, coords_tensor):
    """
    Precompute fields and max_neighbors.
    Returns local_fields, b_centers, c_centers, padded_fields, mask.
    """
    N = n**d
    local_fields = []
    b_centers = []
    c_centers = []
    for i in range(N):
        local_fields.append(get_local_field(i, r, coords_tensor, n, d))
    for i in range(N):
        b_centers.append(get_centers_in_range(i, 2*r, 2*r, coords_tensor))
        c_centers.append(get_centers_in_range(i, 4*r, 4*r, coords_tensor))
    
    # Compute max_neighbors and create padded fields
    max_neighbors = max(len(field) for field in local_fields)
    padded_fields = []
    for field in local_fields:
        padded = field.copy()
        if len(padded) < max_neighbors:
            padded += [-1] * (max_neighbors - len(padded))
        padded_fields.append(padded)
    padded_fields = torch.tensor(padded_fields, dtype=torch.long, device=coords_tensor.device)
    mask = (padded_fields != -1).to(dtype=torch.float32)
    padded_fields = torch.where(padded_fields == -1, torch.tensor(0, device=padded_fields.device), padded_fields)
    
    return local_fields, b_centers, c_centers, padded_fields, mask

@profile
def precompute_triplets(local_fields, b_centers, c_centers, N, device='cuda'):
    anchors = []
    positives = []
    negatives = []
    for i in range(N):
        A_indices = local_fields[i]
        if not A_indices:
            continue
        for b_idx in b_centers[i]:
            if not local_fields[b_idx]:
                continue
            for c_idx in c_centers[i]:
                if not local_fields[c_idx]:
                    continue
                for a_idx in A_indices:
                    anchors.append(a_idx)
                    positives.append(b_idx)
                    negatives.append(c_idx)
    return anchors, positives, negatives

@profile
def build_anchor_pos_neg_for_batch(
    S_history, ai, bi, ci, padded_fields, mask, r=2, device='cuda'
):
    Tplus1, batch_size, N = S_history.shape
    T = Tplus1 - 1
    max_neighbors = padded_fields.shape[1]
    
    ai_tensor = torch.tensor(ai, device=device)
    bi_tensor = torch.tensor(bi, device=device)
    ci_tensor = torch.tensor(ci, device=device)
    
    anchor_fields = padded_fields[ai_tensor]
    pos_fields = padded_fields[bi_tensor]
    neg_fields = padded_fields[ci_tensor]
    
    anchor_masks = mask[ai_tensor]
    pos_masks = mask[bi_tensor]
    neg_masks = mask[ci_tensor]

    def gather_states(time, fields, masks):
        fields_expanded = fields.unsqueeze(0).expand(batch_size, -1, -1)
        states = torch.gather(
            S_history[time].unsqueeze(1).expand(-1, fields.size(0), -1),
            dim=2, index=fields_expanded
        )
        masks_expanded = masks.unsqueeze(0).expand(batch_size, -1, -1)
        return states * masks_expanded

    # Forward direction times
    A_time_1, B_time_1, C_time_1 = 0, 2*r, 4*r
    A_states_1 = gather_states(A_time_1, anchor_fields, anchor_masks)
    B_states_1 = gather_states(B_time_1, pos_fields, pos_masks)
    C_states_1 = gather_states(C_time_1, neg_fields, neg_masks)
    
    # # Reverse direction times
    # A_time_2, B_time_2, C_time_2 = T, T - 2*r, 0
    # A_states_2 = gather_states(A_time_2, anchor_fields, anchor_masks)
    # B_states_2 = gather_states(B_time_2, pos_fields, pos_masks)
    # C_states_2 = gather_states(C_time_2, neg_fields, neg_masks)

    # Concatenate directions
    anchor_cat =A_states_1# torch.cat([A_states_1, A_states_2], dim=1)
    pos_cat = B_states_1#torch.cat([B_states_1, B_states_2], dim=1)
    neg_cat = C_states_1#torch.cat([C_states_1, C_states_2], dim=1)
    # Reshape
    B_total = batch_size * anchor_cat.size(1)  # 2 * 5080 = 10160
    anchor_flat = anchor_cat.view(B_total, max_neighbors)  # [10160, 5]
    pos_flat = pos_cat.view(B_total, max_neighbors)        # [10160, 5]
    neg_flat = neg_cat.view(B_total, max_neighbors).unsqueeze(1)
    
    return anchor_flat, pos_flat, neg_flat
import matplotlib.pyplot as plt
def plot_weight_histogram(W, title="Weight Matrix Histogram"):
    """
    Plots a histogram of the weight matrix values.
    
    Args:
        W (torch.Tensor): The weight matrix of shape [N, N].
        title (str): Title for the histogram plot.
    """
    # Convert weights to a numpy array and flatten
    weights = W.detach().cpu().numpy().ravel()
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(weights, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
# Update ScoreNet initialization in main code
from torchviz import make_dot

if __name__ == "__main__":
  
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Setup grid
    n = 18
    d = 2
    r = 2
    N = n**d
    coords_tensor = get_coords_tensor(n, d, device='cpu')

    # 2. Create initial W
    W_init = create_weight_matrix_naive(n, d, r=r, device=device)
    connectivity_mask = (W_init != 0).float().to(device)  # Shape [N, N]
    
    #W_init = W_init *(torch.ones_like(W_init ).to(device) - torch.eye(W_init.shape[0] ).to(device))
    W = nn.Parameter(W_init.clone(), requires_grad=True)
    W_opt     = torch.optim.Adam([W], lr=1e-4)
    # 3. ScoreNet + optim
    
    

    # 4. Reservoir steps
    T = 4 * r 

    local_fields, b_centers, c_centers, padded_fields, mask = precompute_fields(n, d, r, coords_tensor)
    padded_fields = padded_fields.to(device)
    mask = mask.to(device)
    max_neighbors = padded_fields.shape[1]
    
    # Initialize ScoreNet with input_dim=2*max_neighbors
    score_net = torch.compile(ScoreNet(input_dim=2*max_neighbors, hidden_dim=256).to(device))
    score_opt = torch.optim.Adam(score_net.parameters(), lr=1e-4)
    # During training loop
    ai, bi, ci = precompute_triplets(local_fields, b_centers, c_centers, N, device)
    # 6. Define batch size
    batch_size = 2  # Adjust based on your GPU memory

    # 7. Check precomputed fields
    total_b_centers = sum(1 for b in b_centers if len(b) > 0)
    total_c_centers = sum(1 for c in c_centers if len(c) > 0)
    print(f"Total neurons with at least one B-center: {total_b_centers}/{N}")
    print(f"Total neurons with at least one C-center: {total_c_centers}/{N}")
    print(f"Total triplets: {len(ai)}")

    # 8. Setup-phase: train ScoreNet alone
    setup_steps = 10
    print("\n--- Setup Phase: Training ScoreNet Alone ---")
    B=int(2*4096)
    setup_losses = []

    for step in range(setup_steps):
        # Sample a batch of random initial states
        S0 = torch.randn(batch_size, N, device=device)  # [batch_size, N]
        #print('runnin')
        S_history = run_reservoir_batch(W, S0, T=T)    # [T+1, batch_size, N]
        #print('states ready, getting triplets')
        anchor, pos, neg = build_anchor_pos_neg_for_batch(S_history, ai, bi, ci, padded_fields, mask, r, device)
        #print('got triplets, calculating loss')
        loss = infoNCE_loss(anchor, pos, neg, score_net,batch_size=B)
        #print('calculated loss, backwarding')

        # Optimize ScoreNet only

        score_opt.zero_grad()

        loss.backward()
      
        score_opt.step()

        print(f"[Setup] step={step+1}/{setup_steps}, infoNCE_loss={loss.item():.4f}")
        setup_losses.append(loss.item())

    # 9. Warmup-phase: train both ScoreNet + W
    warmup_steps = 10
    print("\n--- Warmup Phase: Training Both ScoreNet and W ---")
    warmup_losses = []
    for step in range(warmup_steps):
        # Sample a batch of random initial states
        S0 = torch.randn(batch_size, N, device=device)  # [batch_size, N]
        S_history = run_reservoir_batch(W, S0, T=T)    # [T+1, batch_size, N]

        anchor, pos, neg = build_anchor_pos_neg_for_batch(S_history, ai, bi, ci, padded_fields, mask, r, device)
        loss = infoNCE_loss(anchor, pos, neg, score_net,batch_size=B)
#         params = dict(score_net.named_parameters())
#         params["W"] = W  # Include the weight matrix if it's part of the graph
#         graph = make_dot(
#             loss, 
#             params=params,
#             show_attrs=True,    # Show parameter attributes
#             show_saved=True     # Show saved tensors (for backward pass)
#         )

# # Save the graph to a file (e.g., PNG)
#         graph.render("computational_graph", format="png")
#         exit()

        # Optimize both ScoreNet and W
        #plot_weight_histogram(W, title="Initial Weight Matrix Distribution")
        score_opt.zero_grad()
        W_opt.zero_grad()
        loss.backward()
        score_opt.step()
        
        W.grad *= connectivity_mask
        W_opt.step()

        print(f"[Warmup] step={step+1}/{warmup_steps}, infoNCE_loss={loss.item():.4f}")
        warmup_losses.append(loss.item())
    numpy_array = W.detach().cpu().numpy()
    np.save("W2.npy", numpy_array)

