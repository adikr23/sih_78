"""
Message-passing GNN that scores each mesh point as anomalous / not anomalous.

Upgrades over a minimal 2-layer dense version:
  - sparse adjacency (torch.sparse.mm) instead of a dense (P,P) matrix @ features,
    which is the difference between O(P^2) and O(nnz) per layer -- essential once
    P grows past a few hundred (a subdivisions=4 icosphere already has 2562 nodes,
    dense would be a 2562x2562 matmul every layer for no reason)
  - configurable depth (gnn_layers) with residual connections + LayerNorm so it
    doesn't degrade when made deeper
  - takes ALL forecast variables as input features (t2m, rainfall, wind_u, wind_v)
    plus the per-variable EFI as extra channels, instead of a single raw value
"""
import numpy as np
import torch
import torch.nn as nn

from config import settings
from src.mesh import scipy_csr_to_torch_sparse


class SparseGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.norm = nn.LayerNorm(out_feats)

    def forward(self, features, adjacency_sparse):
        agg = torch.sparse.mm(adjacency_sparse, features)
        out = self.linear(agg)
        return self.norm(torch.relu(out))


class AnomalyGNN(nn.Module):
    def __init__(self, in_feats: int, hidden: int = None, n_layers: int = None):
        super().__init__()
        hidden = hidden or settings.gnn_hidden
        n_layers = n_layers or settings.gnn_layers

        self.input_proj = nn.Linear(in_feats, hidden)
        self.layers = nn.ModuleList([SparseGCNLayer(hidden, hidden) for _ in range(n_layers)])
        self.output_head = nn.Linear(hidden, 1)

    def forward(self, features, adjacency_sparse):
        h = torch.relu(self.input_proj(features))
        for layer in self.layers:
            h = h + layer(h, adjacency_sparse)  # residual
        return torch.sigmoid(self.output_head(h)).squeeze(-1)


def build_training_tensors(mesh_features: np.ndarray, efi_combined: np.ndarray, adjacency_csr):
    """mesh_features: (n_points, n_vars); efi_combined: (n_points,) -> labels."""
    features = torch.tensor(mesh_features, dtype=torch.float32)
    labels = torch.tensor((efi_combined > settings.efi_anomaly_threshold).astype(np.float32))
    adjacency_sparse = scipy_csr_to_torch_sparse(adjacency_csr)
    return features, labels, adjacency_sparse


def train_gnn(features: torch.Tensor, labels: torch.Tensor, adjacency_sparse, epochs: int = None):
    epochs = epochs or settings.gnn_epochs
    model = AnomalyGNN(in_feats=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.gnn_lr, weight_decay=1e-5)

    # class-balanced BCE: anomalous points are rare, plain BCE would collapse to "predict nothing"
    n_pos = labels.sum().clamp(min=1)
    n_neg = (labels.numel() - n_pos).clamp(min=1)
    pos_weight = (n_neg / n_pos).item()
    loss_fn = nn.BCELoss(reduction="none")

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(features, adjacency_sparse)
        raw_loss = loss_fn(pred, labels)
        weights = torch.where(labels > 0.5, pos_weight, 1.0)
        loss = (raw_loss * weights).mean()
        loss.backward()
        optimizer.step()
        if epoch % max(1, epochs // 6) == 0:
            history.append(round(loss.item(), 4))

    return model, history


if __name__ == "__main__":
    from scipy import sparse

    vertices = np.load(f"{settings.data_dir}/mesh_vertices.npy")
    mesh_values = np.load(f"{settings.data_dir}/mesh_values.npy")  # (n_points, n_vars)
    adjacency = sparse.load_npz(f"{settings.data_dir}/mesh_adjacency.npz")

    from src.efi import compute_efi_multivar
    baseline = np.load(f"{settings.data_dir}/baseline.npy")
    efi = compute_efi_multivar(mesh_values, baseline, vertices)

    features, labels, adjacency_sparse = build_training_tensors(mesh_values, efi["combined_max"], adjacency)
    model, history = train_gnn(features, labels, adjacency_sparse)

    torch.save(model.state_dict(), f"{settings.data_dir}/gnn_model.pt")
    with torch.no_grad():
        preds = model(features, adjacency_sparse).numpy()
    np.save(f"{settings.data_dir}/gnn_predictions.npy", preds)
    print("Loss history:", history)
    print("Most anomalous point:", int(preds.argmax()), "score:", float(preds.max()))
