"""
XLA-compatible GNN for molecular property prediction.

SchNet-style architecture using only native PyTorch ops (scatter_add,
index operations) — no torch_scatter dependency, fully TPU/XLA safe.

Architecture mirrors the ScalableSurrogate spec from the FlashOptim
discovery: configurable hidden_dim and num_blocks for the 1x vs 2x
scaling experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_radius_graph_local(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Build edge_index for all pairs within cutoff. Pure PyTorch, XLA-safe."""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # [N, N, 3]
    dist = diff.norm(dim=-1)                      # [N, N]
    mask = (dist < cutoff) & (dist > 0)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)         # [2, E]


class GaussianSmearing(nn.Module):
    """Encode interatomic distances as Gaussian basis functions."""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [E] → [E, num_gaussians]
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist.pow(2))


class InteractionBlock(nn.Module):
    """
    One round of continuous-filter message passing (SchNet-style).

    Uses per-molecule torch.bmm for aggregation — a native TPU matmul that
    XLA compiles in seconds. Operates on [B, MAX_ATOMS/MAX_EDGES, hidden_dim]
    tensors to keep all shapes static and avoid scatter_add.
    """

    def __init__(self, hidden_dim: int, num_gaussians: int = 50):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,                 # [B, MAX_ATOMS, hidden_dim]
        edge_src: torch.Tensor,           # [B, MAX_EDGES] local source indices
        edge_attr: torch.Tensor,          # [B, MAX_EDGES, num_gaussians]
        assign_mat: torch.Tensor,         # [B, MAX_EDGES, MAX_ATOMS] precomputed assignment
        edge_valid: torch.Tensor = None,  # [B, MAX_EDGES] bool
    ) -> torch.Tensor:                    # [B, MAX_ATOMS, hidden_dim]
        B         = h.shape[0]
        hid       = h.shape[2]
        MAX_EDGES = edge_src.shape[1]

        # Filter weights from distance encoding
        W = self.filter_net(edge_attr)               # [B, MAX_EDGES, hidden_dim]

        # Gather source atom embeddings — torch.gather → XLA Gather HLO
        h_idx = edge_src.unsqueeze(-1).expand(B, MAX_EDGES, hid)
        h_src = torch.gather(h, 1, h_idx)            # [B, MAX_EDGES, hidden_dim]

        # Messages
        msg = self.msg_linear(h_src) * W             # [B, MAX_EDGES, hidden_dim]

        # Zero out padding edges
        if edge_valid is not None:
            msg = msg * edge_valid.unsqueeze(-1).to(msg.dtype)

        # Aggregate via bmm over precomputed assignment matrix.
        # assign_mat[b, e, a] = 1 if edge e's destination is atom a, else 0.
        # Precomputed on CPU in dataset — never appears as a dynamic comparison
        # inside the XLA compiled graph, cutting HLO instructions by ~100x.
        agg = torch.bmm(assign_mat.permute(0, 2, 1), msg)   # [B, MAX_ATOMS, hidden_dim]

        return self.norm(h + self.update_net(agg))


class MolecularGNN(nn.Module):
    """
    Molecular property prediction GNN, XLA-compatible.

    Scale the model by varying hidden_dim:
      - Condition A/B (baseline):  hidden_dim=256, ~3M params
      - Condition C (2x wider):    hidden_dim=512, ~12M params  (~4x params for 2x width)
    """

    def __init__(
        self,
        num_atom_types: int = 9,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        num_targets: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff

        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.smearing = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.blocks = nn.ModuleList([
            InteractionBlock(hidden_dim, num_gaussians)
            for _ in range(num_blocks)
        ])

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(
        self,
        z: torch.Tensor,                 # [B, MAX_ATOMS] atom type indices
        pos: torch.Tensor,               # [B, MAX_ATOMS, 3] 3D positions
        edge_src: torch.Tensor,          # [B, MAX_EDGES] per-molecule source index
        edge_dst: torch.Tensor,          # [B, MAX_EDGES] per-molecule dest index
        assign_mat: torch.Tensor,        # [B, MAX_EDGES, MAX_ATOMS] precomputed assignment
        num_graphs: int,                 # B
        edge_valid: torch.Tensor = None, # [B, MAX_EDGES] bool
        atom_valid: torch.Tensor = None, # [B, MAX_ATOMS] bool
    ) -> torch.Tensor:                   # [B] predicted property
        B         = z.shape[0]
        MAX_EDGES = edge_src.shape[1]

        # Interatomic distances — use torch.gather (maps to XLA Gather HLO,
        # avoids fancy-index b_idx which may inhibit constant-folding).
        # Expand edge indices to position dimension [B, MAX_EDGES, 3]
        eidx_3  = edge_src.unsqueeze(-1).expand(B, MAX_EDGES, 3)
        src_pos = torch.gather(pos, 1, eidx_3)                      # [B, MAX_EDGES, 3]
        eidx_3d = edge_dst.unsqueeze(-1).expand(B, MAX_EDGES, 3)
        dst_pos = torch.gather(pos, 1, eidx_3d)                     # [B, MAX_EDGES, 3]
        dist      = (dst_pos - src_pos).norm(dim=-1)                 # [B, MAX_EDGES]
        edge_attr = self.smearing(dist)                              # [B, MAX_EDGES, num_gaussians]

        # Atom embeddings
        h = self.embedding(z)                                        # [B, MAX_ATOMS, hidden_dim]

        # Message passing rounds
        for block in self.blocks:
            h = block(h, edge_src, edge_attr, assign_mat, edge_valid)

        # Mask padding atoms
        if atom_valid is not None:
            h = h * atom_valid.unsqueeze(-1).to(h.dtype)            # [B, MAX_ATOMS, hidden_dim]

        # Graph-level readout: sum valid atoms per molecule
        out = h.sum(dim=1)                                           # [B, hidden_dim]

        return self.output_net(out).squeeze(-1)                      # [B]

    def embed(
        self,
        z: torch.Tensor,                 # [B, MAX_ATOMS]
        pos: torch.Tensor,               # [B, MAX_ATOMS, 3]
        edge_src: torch.Tensor,          # [B, MAX_EDGES]
        edge_dst: torch.Tensor,          # [B, MAX_EDGES]
        assign_mat: torch.Tensor,        # [B, MAX_EDGES, MAX_ATOMS]
        num_graphs: int,
        edge_valid: torch.Tensor = None, # [B, MAX_EDGES]
        atom_valid: torch.Tensor = None, # [B, MAX_ATOMS]
    ) -> torch.Tensor:                   # [B, hidden_dim] — graph embedding before output_net
        """
        Same as forward() but returns the pooled graph embedding before output_net.
        Used by Phase 2 binding affinity head to add a task-specific regression layer.
        """
        B         = z.shape[0]
        MAX_EDGES = edge_src.shape[1]
        eidx_3    = edge_src.unsqueeze(-1).expand(B, MAX_EDGES, 3)
        src_pos   = torch.gather(pos, 1, eidx_3)
        eidx_3d   = edge_dst.unsqueeze(-1).expand(B, MAX_EDGES, 3)
        dst_pos   = torch.gather(pos, 1, eidx_3d)
        dist      = (dst_pos - src_pos).norm(dim=-1)
        edge_attr = self.smearing(dist)
        h = self.embedding(z)
        for block in self.blocks:
            h = block(h, edge_src, edge_attr, assign_mat, edge_valid)
        if atom_valid is not None:
            h = h * atom_valid.unsqueeze(-1).to(h.dtype)
        return h.sum(dim=1)                                          # [B, hidden_dim]

    def forward_embed(
        self,
        z: torch.Tensor,    # [N] atom type indices
        pos: torch.Tensor,  # [N, 3] 3D positions
        batch: torch.Tensor,# [N] graph assignment
    ) -> torch.Tensor:      # [num_graphs, hidden_dim] — graph embedding before output_net
        """
        CPU-only inference helper that builds edge_index internally.
        NOT used in TPU training (incompatible with batched forward signature).
        Raises NotImplementedError — kept as a stub for future adaptation.
        """
        raise NotImplementedError(
            "forward_embed is incompatible with the padded-batch InteractionBlock "
            "introduced for XLA compatibility. Use forward() with precomputed "
            "assign_mat from batch_to_graph() instead."
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(condition: str, device: torch.device) -> MolecularGNN:
    """
    Build model for the given experimental condition.

    A — FP32 baseline:           hidden_dim=256
    B — BF16 same size:          hidden_dim=256
    C — BF16 2x wider:           hidden_dim=512  (~4x parameters)
    """
    hidden_dim = 512 if condition == 'C' else 256
    model = MolecularGNN(
        num_atom_types=9,
        hidden_dim=hidden_dim,
        num_blocks=6,
        num_gaussians=50,
        cutoff=5.0,
        num_targets=1,
    ).to(device)
    print(f"Model (condition {condition}): hidden_dim={hidden_dim}, "
          f"params={model.parameter_count():,}")
    return model
