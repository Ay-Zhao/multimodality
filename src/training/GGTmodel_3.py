import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, Sequential, GraphNorm,
    global_mean_pool, global_max_pool,
)
from transformers import AutoModelForMaskedLM
from src.branch_3D.model_unimol import UniMolModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Valid modality + fusion combinations
# Reference: Data Flow Table
#
# Single:
#   "1d"                       -> [B, 1, 767] -> 767
#   "2d"                       -> [B, 2, 767] -> 1534
#   "3d"                       -> [B, 1, 767] -> 767
#
# Dual 1D+2D:
#   "1d_2d" + "concat"         -> [B, 3, 767] -> 2301
#   "1d_2d" + "1d_query_2d"    -> [B, 1, 767] -> 767
#   "1d_2d" + "2d_query_1d"    -> [B, 2, 767] -> 1534
#
# Dual 1D+3D:
#   "1d_3d" + "concat"         -> [B, 2, 767] -> 1534
#   "1d_3d" + "add"            -> [B, 1, 767] -> 767
#   "1d_3d" + "1d_query_3d"    -> [B, 1, 767] -> 767
#   "1d_3d" + "3d_query_1d"    -> [B, 1, 767] -> 767
#
# Dual 2D+3D:
#   "2d_3d" + "concat"         -> [B, 3, 767] -> 2301
#   "2d_3d" + "2d_query_3d"    -> [B, 2, 767] -> 1534
#   "2d_3d" + "3d_query_2d"    -> [B, 1, 767] -> 767
#
# Triple 1D+2D+3D:
#   "1d_2d_3d" + "concat"      -> [B, 4, 767] -> 3068
#   "1d_2d_3d" + "plus"        -> [B, 2, 767] -> 1534
# =============================================================================

VALID_COMBOS = {
    # single
    "1d":      {None},
    "2d":      {None},
    "3d":      {None},
    # dual
    "1d_2d":   {"concat", "1d_query_2d", "2d_query_1d"},
    "1d_3d":   {"concat", "add", "1d_query_3d", "3d_query_1d"},
    "2d_3d":   {"concat", "2d_query_3d", "3d_query_2d"},
    # triple
    "1d_2d_3d": {"concat", "plus"},
}

# Flattened input size to MLP for each (modality, fusion) combo
FLAT_SIZE = {
    ("1d",      None):           767,
    ("2d",      None):          1534,
    ("3d",      None):           767,
    ("1d_2d",   "concat"):      2301,
    ("1d_2d",   "1d_query_2d"):  767,
    ("1d_2d",   "2d_query_1d"): 1534,
    ("1d_3d",   "concat"):      1534,
    ("1d_3d",   "add"):          767,
    ("1d_3d",   "1d_query_3d"):  767,
    ("1d_3d",   "3d_query_1d"):  767,
    ("2d_3d",   "concat"):      2301,
    ("2d_3d",   "2d_query_3d"): 1534,
    ("2d_3d",   "3d_query_2d"):  767,
    ("1d_2d_3d","concat"):      3068,
    ("1d_2d_3d","plus"):        1534,
}


# =============================================================================
# 2D Branch
# =============================================================================

class GNN_branch_with_GAT(torch.nn.Module):
    def __init__(self):
        super(GNN_branch_with_GAT, self).__init__()
        self.graphconv = Sequential('x,edge_index,batch', [
            (GATConv(9, 128),    'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(128),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(128, 256),  'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(256),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(256, 512),  'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(512),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(512, 767),  'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(767),
        ])

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        graph_representation = self.graphconv(x, edge_index, batch)

        mean = global_mean_pool(graph_representation, batch).view(-1, 1, 767)
        max_ = global_max_pool(graph_representation, batch).view(-1, 1, 767)

        # [B, 2, 767]
        return torch.cat([mean, max_], dim=1)


# =============================================================================
# Main Model
# =============================================================================

class Net(torch.nn.Module):
    """
    Multi-modality fusion model supporting 16 modality+fusion combinations.

    Branch outputs:
        1D (ChemBERTa) : CLS token      -> [B, 1, 767]
        2D (GAT)       : mean + max pool -> [B, 2, 767]
        3D (UniMol)    : first token     -> [B, 1, 767]

    ┌─────────────┬──────────────────┬──────────────────────────────────────────┬──────────┐
    │  Modality   │  Fusion          │  Operation                               │ Flat dim │
    ├─────────────┼──────────────────┼──────────────────────────────────────────┼──────────┤
    │ 1d          │ None             │ CLS token                                │  767     │
    │ 2d          │ None             │ mean + max pool                          │  1534    │
    │ 3d          │ None             │ first token                              │  767     │
    ├─────────────┼──────────────────┼──────────────────────────────────────────┼──────────┤
    │ 1d_2d       │ concat           │ cat([B,1,767], [B,2,767], dim=1)         │  2301    │
    │ 1d_2d       │ 1d_query_2d      │ CrossAttn(Q=1D, KV=2D)                   │  767     │
    │ 1d_2d       │ 2d_query_1d      │ CrossAttn(Q=2D, KV=1D)                   │  1534    │
    ├─────────────┼──────────────────┼──────────────────────────────────────────┼──────────┤
    │ 1d_3d       │ concat           │ cat([B,1,767], [B,1,767], dim=1)         │  1534    │
    │ 1d_3d       │ add              │ [B,1,767] + [B,1,767]                    │  767     │
    │ 1d_3d       │ 1d_query_3d      │ CrossAttn(Q=1D, KV=3D)                   │  767     │
    │ 1d_3d       │ 3d_query_1d      │ CrossAttn(Q=3D, KV=1D)                   │  767     │
    ├─────────────┼──────────────────┼──────────────────────────────────────────┼──────────┤
    │ 2d_3d       │ concat           │ cat([B,2,767], [B,1,767], dim=1)         │  2301    │
    │ 2d_3d       │ 2d_query_3d      │ CrossAttn(Q=2D, KV=3D)                   │  1534    │
    │ 2d_3d       │ 3d_query_2d      │ CrossAttn(Q=3D, KV=2D)                   │  767     │
    ├─────────────┼──────────────────┼──────────────────────────────────────────┼──────────┤
    │ 1d_2d_3d    │ concat           │ cat([B,2,767],[B,1,767],[B,1,767], dim=1)│  3068    │
    │ 1d_2d_3d    │ plus             │ CrossAttn(Q=2D,KV=1D)+CrossAttn(Q=2D,KV=3D)│ 1534  │
    └─────────────┴──────────────────┴──────────────────────────────────────────┴──────────┘

    Usage:
        model = Net(modality="1d_2d_3d", fusion="concat", n_output_layers=1)
        model = Net(modality="1d_2d",    fusion="1d_query_2d")
        model = Net(modality="2d",       fusion=None)
    """

    def __init__(self, n_output_layers=1, modality="1d_2d_3d", fusion="concat"):
        super().__init__()

        # ── Validate modality + fusion combo ──────────────────────────────
        if modality not in VALID_COMBOS:
            raise ValueError(
                f"Unknown modality: '{modality}'.\n"
                f"Valid options: {set(VALID_COMBOS.keys())}"
            )
        if fusion not in VALID_COMBOS[modality]:
            raise ValueError(
                f"Invalid fusion '{fusion}' for modality '{modality}'.\n"
                f"Valid fusions for '{modality}': {VALID_COMBOS[modality]}"
            )

        self.hidden_dim      = 767
        self.n_output_layers = n_output_layers
        self.modality        = modality
        self.fusion          = fusion

        # ── Branches (only instantiate what is needed) ────────────────────
        self.use_1d = "1d" in modality
        self.use_2d = "2d" in modality
        self.use_3d = "3d" in modality

        if self.use_1d:
            self.chemberta = AutoModelForMaskedLM.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1"
            )

        if self.use_2d:
            self.gnn_branch = GNN_branch_with_GAT()

        if self.use_3d:
            self.geometric_branch = UniMolModel(
                output_dim=self.hidden_dim,
                data_type="molecule",
                remove_hs=False,
            )

        # ── Cross-attention layers (only instantiate what is needed) ──────
        # decoder layer A is used for:
        #   1d_2d   : 1d_query_2d  or  2d_query_1d
        #   1d_3d   : 1d_query_3d  or  3d_query_1d
        #   2d_3d   : 2d_query_3d  or  3d_query_2d
        #   1d_2d_3d: CrossAttn(Q=2D, KV=1D)   [decoder layer A]
        #             CrossAttn(Q=2D, KV=3D)   [decoder layer B]

        # dual modality cross-attention fusions
        DUAL_ATTN_FUSIONS = {
            "1d_query_2d", "2d_query_1d",
            "1d_query_3d", "3d_query_1d",
            "2d_query_3d", "3d_query_2d",
        }

        # decoder_A: needed for all dual cross-attn + all triple fusions
        needs_attn_A = (fusion in DUAL_ATTN_FUSIONS) or (modality == "1d_2d_3d")

        # decoder_B: needed for all triple fusions (both concat and plus)
        needs_attn_B = (modality == "1d_2d_3d")

        if needs_attn_A:
            self.decoder_A = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim, nhead=13, batch_first=True,
            )
        if needs_attn_B:
            self.decoder_B = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim, nhead=13, batch_first=True,
            )

        # ── MLP head (input size determined by modality + fusion) ─────────
        flat_size = FLAT_SIZE[(modality, fusion)]
        self.head = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.n_output_layers),
        )

    # ---------------------------------------------------------------------- #
    # Staged training                                                          #
    # ---------------------------------------------------------------------- #

    def set_train_stage(self, stage: str):
        def freeze(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        def unfreeze(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = True
            m.train()

        if stage == "fusion_only":
            if self.use_1d: freeze(self.chemberta)
            if self.use_3d: freeze(self.geometric_branch)
            if self.use_2d: unfreeze(self.gnn_branch)
            if hasattr(self, "decoder_A"): unfreeze(self.decoder_A)
            if hasattr(self, "decoder_B"): unfreeze(self.decoder_B)
            unfreeze(self.head)

        elif stage == "unfreeze_all":
            if self.use_1d: unfreeze(self.chemberta)
            if self.use_3d: unfreeze(self.geometric_branch)
            if self.use_2d: unfreeze(self.gnn_branch)
            if hasattr(self, "decoder_A"): unfreeze(self.decoder_A)
            if hasattr(self, "decoder_B"): unfreeze(self.decoder_B)
            unfreeze(self.head)

        else:
            raise ValueError(f"Unknown stage: '{stage}'")

    # ---------------------------------------------------------------------- #
    # Branch encoders                                                          #
    # ---------------------------------------------------------------------- #

    def _encode_1d(self, inputs):
        """
        ChemBERTa CLS token.

        Returns
        -------
        [B, 1, 767]
        """
        out = self.chemberta(**inputs)   # out[0]: [B, T, 767]
        return out[0][:, 0:1, :]        # [B, 1, 767]

    def _encode_2d(self, graph):
        """
        GAT mean + max pool.

        Returns
        -------
        [B, 2, 767]
        """
        return self.gnn_branch(graph)   # [B, 2, 767]

    def _encode_3d(self, unimol_input):
        """
        UniMol first token.

        Returns
        -------
        [B, 1, 767]
        """
        geo_out = self.geometric_branch(
            unimol_input["src_tokens"],
            unimol_input["src_distance"],
            unimol_input["src_coord"],
            unimol_input["src_edge_type"],
        )                               # [B, N, 767]

        # Case 2: [B, 767] → already pooled, just unsqueeze
        if geo_out.dim() == 2:
            return geo_out.unsqueeze(1)  # [B, 1, 767]

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, graph=None, inputs=None, unimol_input=None):

        # ── Encode active branches ────────────────────────────────────────
        seq_1d = self._encode_1d(inputs)          if self.use_1d else None  # [B, 1, 767]
        seq_2d = self._encode_2d(graph)           if self.use_2d else None  # [B, 2, 767]
        seq_3d = self._encode_3d(unimol_input)    if self.use_3d else None  # [B, 1, 767]

        # ── Single modality ───────────────────────────────────────────────
        if self.modality == "1d":
            fused = seq_1d                                  # [B, 1, 767]

        elif self.modality == "2d":
            fused = seq_2d                                  # [B, 2, 767]

        elif self.modality == "3d":
            fused = seq_3d                                  # [B, 1, 767]

        # ── Dual: 1D + 2D ─────────────────────────────────────────────────
        elif self.modality == "1d_2d":
            if self.fusion == "concat":
                fused = torch.cat([seq_1d, seq_2d], dim=1) # [B, 3, 767]
            elif self.fusion == "1d_query_2d":
                fused = self.decoder_A(seq_1d, seq_2d)     # [B, 1, 767]
            elif self.fusion == "2d_query_1d":
                fused = self.decoder_A(seq_2d, seq_1d)     # [B, 2, 767]

        # ── Dual: 1D + 3D ─────────────────────────────────────────────────
        elif self.modality == "1d_3d":
            if self.fusion == "concat":
                fused = torch.cat([seq_1d, seq_3d], dim=1) # [B, 2, 767]
            elif self.fusion == "add":
                fused = seq_1d + seq_3d                    # [B, 1, 767]
            elif self.fusion == "1d_query_3d":
                fused = self.decoder_A(seq_1d, seq_3d)     # [B, 1, 767]
            elif self.fusion == "3d_query_1d":
                fused = self.decoder_A(seq_3d, seq_1d)     # [B, 1, 767]

        # ── Dual: 2D + 3D ─────────────────────────────────────────────────
        elif self.modality == "2d_3d":
            if self.fusion == "concat":
                fused = torch.cat([seq_2d, seq_3d], dim=1) # [B, 3, 767]
            elif self.fusion == "2d_query_3d":
                fused = self.decoder_A(seq_2d, seq_3d)     # [B, 2, 767]
            elif self.fusion == "3d_query_2d":
                fused = self.decoder_A(seq_3d, seq_2d)     # [B, 1, 767]


        # ── Triple: 1D + 2D + 3D ──────────────────────────────────────────
        elif self.modality == "1d_2d_3d":
            att_2d_1d = self.decoder_A(seq_2d, seq_1d)  # [B, 2, 767]
            att_2d_3d = self.decoder_B(seq_2d, seq_3d)  # [B, 2, 767]

            if self.fusion == "concat":
                fused = torch.cat([att_2d_1d, att_2d_3d], dim=1)  # [B, 4, 767]
            elif self.fusion == "plus":
                fused = att_2d_1d + att_2d_3d  # [B, 2, 767]

        # ── Flatten + MLP ─────────────────────────────────────────────────
        fused = fused.view(fused.size(0), -1)              # [B, flat_size]
        return self.head(fused)                            # [B, n_output_layers]