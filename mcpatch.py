#!/usr/bin/env python
"""
Script 1: Pre-training a spatially-structured reservoir using multi–channel locally–connected updates
and contrastive InfoNCE loss. The reservoir weight tensor is built with shape:
    [n, n, C, C, 2*r+1, 2*r+1],
where C is the number of channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from line_profiler import profile
from tqdm import tqdm

#######################################
# Global Parameters
#######################################
n = 16              # Grid side length so that N = n*n neurons.
r = 2               # Manhattan radius.
channels = 4        # Number of channels (features) per neuron.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################
# 1) Coordinate & Field Utilities (unchanged)
#######################################
@profile
def index_to_coords(idx: int, n: int, d: int):
    # For a 2D grid (row-major):
    if d == 2:
        return [idx // n, idx % n]
    else:
        coords = []
        for _ in range(d):
            coords.insert(0, idx % n)
            idx //= n
        return coords

@profile
def coords_to_index(coords, n: int):
    if len(coords) == 2:
        return coords[0] * n + coords[1]
    else:
        idx = 0
        for c in coords:
            idx = idx * n + c
        return idx

@profile
def get_coords_tensor(n, d, device='cpu'):
    N = n**d
    all_coords = [index_to_coords(i, n, d) for i in range(N)]
    coords_tensor = torch.tensor(all_coords, device=device, dtype=torch.long)
    return coords_tensor

#######################################
# 2) Build the Multi-Channel Reservoir Weight Tensor
#######################################
@profile
def create_weight_tensor(n: int, r: int, channels: int, device='cpu'):
    """
    Create a weight tensor of shape [n, n, channels, channels, 2*r+1, 2*r+1].
    For each target neuron (i,j) and each offset (di, dj) with |di|+|dj| <= r,
    if the source neuron (i-di, j-dj) is within bounds, assign a random matrix ~N(0,1)
    of shape [channels, channels]; else leave as zero.
    """
    kernel_size = 2 * r + 1
    W_tensor = torch.zeros((n, n, channels, channels, kernel_size, kernel_size), device=device)
    for i in range(n):
        for j in range(n):
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if abs(di) + abs(dj) <= r:
                        src_i = i - di
                        src_j = j - dj
                        if 0 <= src_i < n and 0 <= src_j < n:
                            # Random weight matrix for each valid offset.
                            W_tensor[i, j, :, :, di + r, dj + r] = torch.randn(channels, channels)
    return W_tensor

#######################################
# 3) Multi-Channel Reservoir Update via Locally-Connected Operation
#######################################
def multi_channel_reservoir_update(S, W_tensor, r):
    """
    Update the reservoir state S (of shape [batch, channels, n, n]) using the multi-channel
    weight tensor W_tensor of shape [n, n, channels, channels, 2*r+1, 2*r+1].
    
    Returns the updated state of shape [batch, channels, n, n].
    """
    batch, C, n1, n2 = S.shape  # Here, C == channels and n1 == n2 == n.
    kernel_size = 2 * r + 1
    
    # Use F.unfold to extract patches.
    # For multi-channel input, unfold returns shape: [batch, C*kernel_size^2, n*n].
    patches = F.unfold(S, kernel_size=(kernel_size, kernel_size), padding=r)
    # Reshape patches into [batch, n*n, C, kernel_size, kernel_size]
    patches = patches.view(batch, C, kernel_size * kernel_size, n*n)
    patches = patches.permute(0, 3, 1, 2)  # [batch, n*n, C, kernel_size^2]
    patches = patches.view(batch, n*n, C, kernel_size, kernel_size)  # [batch, n*n, C, kernel_size, kernel_size]
    
    # Flatten spatial locations in the weight tensor.
    # W_tensor has shape [n, n, channels (out), channels (in), kernel_size, kernel_size]
    W_flat = W_tensor.view(n*n, channels, channels, kernel_size, kernel_size)  # [n*n, Cout, Cin, k, k]
    
    # For each spatial location l, perform:
    # out[b, l, oc] = sum_{ic, m, n} patches[b, l, ic, m, n] * W_flat[l, oc, ic, m, n]
    out = torch.einsum('blimn, l o i m n -> b l o', patches, W_flat)  # [batch, n*n, channels]
    
    # Reshape back to [batch, channels, n, n]
    out = out.view(batch, n, n, channels).permute(0, 3, 1, 2)
    return out

#######################################
# 4) Reservoir Batch Runner (Multi-Channel)
#######################################
@profile
def run_reservoir_batch(W_tensor, S0, T=5, r=2, n=20, channels=channels):
    """
    Recurrently update the reservoir state.
    S0 is given as [batch, channels, n, n] and returns a tensor of shape
    [T+1, batch, n*n, channels].
    """
    batch = S0.shape[0]
    S = S0.reshape(batch, channels, n, n)  # Use reshape instead of view.
    states_list = [S.reshape(batch, n*n, channels)]
    for _ in range(T):
        S = multi_channel_reservoir_update(S, W_tensor, r)
        S = F.leaky_relu(S, negative_slope=0.1)
        states_list.append(S.reshape(batch, n*n, channels))
    return torch.stack(states_list, dim=0)  # [T+1, batch, n*n, channels]

#######################################
# 5) Field & Triplet Precomputation Utilities (unchanged)
#######################################
@profile
def get_local_field(center_idx, r, coords_tensor, n, d):
    c_i = coords_tensor[center_idx]
    dist = torch.sum(torch.abs(coords_tensor - c_i), dim=1)
    mask = (dist <= r)
    field_indices = mask.nonzero(as_tuple=True)[0].tolist()
    return field_indices

@profile
def get_centers_in_range(center_idx, dist_lo, dist_hi, coords_tensor):
    c_i = coords_tensor[center_idx]
    dist = torch.sum(torch.abs(coords_tensor - c_i), dim=1)
    mask = (dist >= dist_lo) & (dist <= dist_hi)
    candidates = mask.nonzero(as_tuple=True)[0].tolist()
    return candidates

@profile
def precompute_fields(n, d, r, coords_tensor):
    N = n**d
    local_fields = []
    b_centers = []
    c_centers = []
    for i in range(N):
        local_fields.append(get_local_field(i, r, coords_tensor, n, d))
    for i in range(N):
        b_centers.append(get_centers_in_range(i, 2*r, 2*r, coords_tensor))
        c_centers.append(get_centers_in_range(i, 4*r, 4*r, coords_tensor))
    
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
    anchors, positives, negatives = [], [], []
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

#######################################
# 6) Contrastive Learning: ScoreNet & InfoNCE Loss (updated for multi-channel fields)
#######################################
class ScoreNet(nn.Module):
    """
    A simple feed-forward network that outputs a scalar given the concatenation of
    two fields. For each field, the features are flattened.
    """
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)
@profile
def infoNCE_loss(anchor, pos, neg, score_net, batch_size=512, sub_batch_size=256):
    """
    Computes the InfoNCE loss in two levels of batching:
    
    - Outer loop batches the total number of triplets (anchor, pos, neg) into chunks of size batch_size.
    - Inner (sub-)batch loop processes the ScoreNet inputs in smaller pieces (of size sub_batch_size).
    
    Args:
        anchor (Tensor): [B, d]
        pos (Tensor): [B, d]
        neg (Tensor): [B, K_neg, d]
        score_net (nn.Module): the scoring network.
        batch_size (int): batch size for outer loop over triplets.
        sub_batch_size (int): batch size for ScoreNet sub–batch processing.
        
    Returns:
        loss (Tensor): scalar loss.
    """
    device = anchor.device
    B, d = anchor.shape
    K_neg = neg.shape[1]
    total_loss = 0.0

    for i in range(0, B, batch_size):
        # Outer loop: process a chunk of triplets
        anchor_chunk = anchor[i:i+batch_size]
        pos_chunk = pos[i:i+batch_size]
        neg_chunk = neg[i:i+batch_size]
        current_chunk_size = anchor_chunk.size(0)
        
        # Helper function to run ScoreNet in sub–batches.
        def run_score_net(x):
            outs = []
            for j in range(0, x.size(0), sub_batch_size):
                outs.append(score_net(x[j:j+sub_batch_size]).squeeze(1))
            return torch.cat(outs, dim=0)
        
        # Process positive pairs:
        pos_input = torch.cat([anchor_chunk, pos_chunk], dim=1)  # shape: [chunk_size, 2*d]
        pos_score = run_score_net(pos_input)  # [chunk_size]
        
        # Process negative pairs:
        # First, flatten neg_chunk: currently shape [chunk_size, K_neg, d]
        neg_flat = neg_chunk.reshape(-1, d)  # [chunk_size*K_neg, d]
        # Repeat anchor_chunk for each negative sample:
        anchor_repeat = anchor_chunk.unsqueeze(1).repeat(1, K_neg, 1).reshape(-1, d)  # [chunk_size*K_neg, d]
        neg_input = torch.cat([anchor_repeat, neg_flat], dim=1)  # [chunk_size*K_neg, 2*d]
        neg_score = run_score_net(neg_input)  # [chunk_size*K_neg]
        
        # Reshape negative scores:
        neg_score = neg_score.reshape(current_chunk_size, K_neg)
        
        # Combine scores for InfoNCE loss:
        pos_score_exp = pos_score.unsqueeze(1)  # [chunk_size, 1]
        all_scores = torch.cat([pos_score_exp, neg_score], dim=1)  # [chunk_size, 1+K_neg]
        logsumexp = torch.logsumexp(all_scores, dim=1)  # [chunk_size]
        chunk_loss = -(pos_score - logsumexp).mean()
        
        total_loss += chunk_loss * (current_chunk_size / B)
    return total_loss

#######################################
# 7) Build Anchor, Positive, Negative Triplets for a Batch
#######################################
@profile
def build_anchor_pos_neg_for_batch(S_history, ai, bi, ci, padded_fields, mask, r=2, device='cuda'):
    """
    S_history: [T+1, batch, n*n, channels]
    padded_fields, mask: tensors of shape [n*n, max_neighbors]
    
    For each triplet (anchor, positive, negative), we gather the corresponding neuron features
    (which are vectors of length `channels`) from the reservoir history.
    Then, we flatten the gathered features (of shape [max_neighbors, channels])
    to a vector of length (max_neighbors * channels).
    """
    Tplus1, batch_size, N, C = S_history.shape
    max_neighbors = padded_fields.shape[1]
    
    # Convert the triplet lists into tensors.
    ai_tensor = torch.tensor(ai, device=device)  # shape: [num_triplets]
    bi_tensor = torch.tensor(bi, device=device)
    ci_tensor = torch.tensor(ci, device=device)
    
    def gather_states(time, fields, masks):
        # S_history[time]: [batch, N, C]
        # fields: [num_triplets, max_neighbors] (neuron indices)
        # masks: [num_triplets, max_neighbors]
        
        # Expand fields and masks to include the batch dimension.
        # New shape: [batch, num_triplets, max_neighbors]
        fields_expanded = fields.unsqueeze(0).expand(batch_size, -1, -1)
        masks_expanded = masks.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)  # -> [batch, num_triplets, max_neighbors, 1]
        
        # Create a batch index tensor of shape [batch, num_triplets, max_neighbors]
        batch_idx = torch.arange(batch_size, device=S_history.device).view(batch_size, 1, 1).expand_as(fields_expanded)
        
        # Use advanced indexing to gather from S_history[time]
        # S_history[time] has shape [batch, N, C]
        # The result will have shape [batch, num_triplets, max_neighbors, C]
        gathered = S_history[time][batch_idx, fields_expanded]
        
        return gathered * masks_expanded
    
    # Use fixed time delays for anchor, positive, and negative.
    A_time, B_time, C_time = 0, 2*r, 4*r
    A_states = gather_states(A_time, padded_fields[ai_tensor], mask[ai_tensor])
    B_states = gather_states(B_time, padded_fields[bi_tensor], mask[bi_tensor])
    C_states = gather_states(C_time, padded_fields[ci_tensor], mask[ci_tensor])
    
    # Flatten the last two dimensions so that each field becomes a vector of length (max_neighbors * C).
    A_flat = A_states.reshape(batch_size * A_states.size(1), max_neighbors * C)
    B_flat = B_states.reshape(batch_size * A_states.size(1), max_neighbors * C)
    C_flat = C_states.reshape(batch_size * A_states.size(1), max_neighbors * C)
    
    # For the InfoNCE loss, negatives are expected with an extra dimension.
    C_flat = C_flat.unsqueeze(1)
    return A_flat, B_flat, C_flat


#######################################
# 8) Main Pre-training Loop
#######################################
if __name__ == "__main__":
    print(f"Using device: {device}")
    d = 2
    N = n**d
    coords_tensor = get_coords_tensor(n, d, device='cpu')

    # Create the multi-channel reservoir weight tensor.
    W_tensor = create_weight_tensor(n, r, channels, device=device)
    # Create a connectivity mask (to preserve sparsity during training).
    connectivity_mask = (W_tensor != 0).float().to(device)
    # Wrap W_tensor as a trainable parameter.
    W = nn.Parameter(W_tensor.clone(), requires_grad=True)
    W_opt = torch.optim.Adam([W], lr=1e-4)
    
    # Define the number of reservoir update steps.
    T = 4 * r
    local_fields, b_centers, c_centers, padded_fields, mask = precompute_fields(n, d, r, coords_tensor)
    padded_fields = padded_fields.to(device)
    mask = mask.to(device)
    
    # For ScoreNet: each field (of max_neighbors neurons) now has dimension (max_neighbors * channels),
    # so ScoreNet input is 2*(max_neighbors * channels)
    max_neighbors = padded_fields.shape[1]
    score_net = torch.compile(ScoreNet(input_dim=2 * max_neighbors * channels, hidden_dim=256).to(device))
    score_opt = torch.optim.Adam(score_net.parameters(), lr=1e-4)
    
    # Precompute triplets from the fields.
    ai, bi, ci = precompute_triplets(local_fields, b_centers, c_centers, N, device)
    batch_size = 2
    print(f"Total neurons with at least one B-center: {sum(1 for b in b_centers if len(b) > 0)}/{N}")
    print(f"Total neurons with at least one C-center: {sum(1 for c in c_centers if len(c) > 0)}/{N}")
    print(f"Total triplets: {len(ai)}")
    
    # --- Setup Phase: Train ScoreNet alone ---
    setup_steps = 10
    print("\n--- Setup Phase: Training ScoreNet Alone ---")
    B = int(2 * 4096)
    setup_losses = []
    for step in range(setup_steps):
        # S0 now is of shape [batch, channels, n, n].
        S0 = torch.randn(batch_size, channels, n, n, device=device)
        S_history = run_reservoir_batch(W, S0, T=T, r=r, n=n, channels=channels)  # shape: [T+1, batch, n*n, channels]
        anchor, pos, neg = build_anchor_pos_neg_for_batch(S_history, ai, bi, ci, padded_fields, mask, r, device)
        loss = infoNCE_loss(anchor, pos, neg, score_net, batch_size=B)
        score_opt.zero_grad()
        loss.backward()
        score_opt.step()
        print(f"[Setup] step={step+1}/{setup_steps}, infoNCE_loss={loss.item():.4f}")
        setup_losses.append(loss.item())
    
    # --- Warmup Phase: Train both ScoreNet and Reservoir W ---
    warmup_steps = 10
    print("\n--- Warmup Phase: Training Both ScoreNet and Reservoir W ---")
    warmup_losses = []
    for step in range(warmup_steps):
        S0 = torch.randn(batch_size, channels, n, n, device=device)
        S_history = run_reservoir_batch(W, S0, T=T, r=r, n=n, channels=channels)
        anchor, pos, neg = build_anchor_pos_neg_for_batch(S_history, ai, bi, ci, padded_fields, mask, r, device)
        loss = infoNCE_loss(anchor, pos, neg, score_net, batch_size=B)
        score_opt.zero_grad()
        W_opt.zero_grad()
        loss.backward()
        score_opt.step()
        # Enforce sparsity of W by masking its gradient.
        W.grad *= connectivity_mask
        W_opt.step()
        print(f"[Warmup] step={step+1}/{warmup_steps}, infoNCE_loss={loss.item():.4f}")
        warmup_losses.append(loss.item())
    
    np.save("W2_tensor_multi.npy", W.detach().cpu().numpy())
