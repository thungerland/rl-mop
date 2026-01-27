"""
Simple Mixture of Experts (MoE) for BabyAI.

This module provides a clean, minimal implementation of mixture-of-experts
that naturally extends the existing GRUPolicy pattern.

Key design principles:
1. Matches the GRUPolicy interface: forward(x, lang_embs, h) -> (logits, h_new)
2. Router and experts are simple GRUs (just like the original)
3. Hidden states are managed explicitly (no complex dictionaries)
4. Routing information can be extracted for analysis
5. Uses intermediate_dim for consistent layer-to-layer communication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class Expert(nn.Module):
    """
    A single expert - a GRU that processes input and projects to intermediate_dim.

    Architecture:
        Input (intermediate_dim) → GRU (hidden_size) → Linear → Output (intermediate_dim)
    """

    def __init__(self, intermediate_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim

        self.gru = nn.GRU(
            input_size=intermediate_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        # Project back to intermediate dimension
        self.projection = nn.Linear(hidden_size, intermediate_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, intermediate_dim) - single timestep
            h: (1, batch, hidden_size) - hidden state

        Returns:
            out: (batch, intermediate_dim) - expert output
            h_new: (1, batch, hidden_size) - new hidden state
        """
        out, h_new = self.gru(x, h)
        out = out.squeeze(1)  # (batch, hidden_size)
        out = self.projection(out)  # (batch, intermediate_dim)
        return out, h_new


class Router(nn.Module):
    """
    Router that decides how to weight each expert.

    Takes intermediate_dim input and outputs weights (one per expert).
    """

    def __init__(self, intermediate_dim: int, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.gru = nn.GRU(
            input_size=intermediate_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_experts)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, intermediate_dim)
            h: (1, batch, hidden_size)

        Returns:
            weights: (batch, num_experts) - softmax weights for experts
            h_new: (1, batch, hidden_size)
        """
        out, h_new = self.gru(x, h)
        out = out.squeeze(1)  # (batch, hidden_size)
        logits = self.fc(out)  # (batch, num_experts)
        weights = F.softmax(logits, dim=-1)
        return weights, h_new


class MixtureOfExpertsPolicy(nn.Module):
    """
    Mixture of Experts policy that matches the GRUPolicy interface.

    Architecture:
    1. Input layer: maps (obs + lang) to intermediate_dim
    2. Router GRU decides expert weights
    3. Multiple expert GRUs process the input (each outputs intermediate_dim)
    4. Outputs are weighted and combined → intermediate_dim
    5. Output head: maps intermediate_dim to actions

    This design allows stacking multiple MoE layers later.

    Usage (same as GRUPolicy):
        policy = MixtureOfExpertsPolicy(...)
        h = policy.init_hidden(batch_size, device)

        for t in range(steps):
            logits, h = policy(obs, lang, h)
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,  # Common dimension for layer communication
        expert_hidden_sizes: List[int],  # e.g., [32, 64, 128]
        router_hidden_size: int,
        num_actions: int,
        lang_dim: int = 128
    ):
        """
        Args:
            input_dim: Observation dimension (without language)
            intermediate_dim: Intermediate representation dimension
            expert_hidden_sizes: List of hidden sizes for each expert GRU
            router_hidden_size: Hidden size for the router GRU
            num_actions: Action space size
            lang_dim: Language embedding dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.lang_dim = lang_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = len(expert_hidden_sizes)
        self.expert_hidden_sizes = expert_hidden_sizes
        self.router_hidden_size = router_hidden_size

        # Input projection: (obs + lang) → intermediate_dim
        self.input_projection = nn.Linear(input_dim + lang_dim, intermediate_dim)

        # Router: decides which experts to use
        self.router = Router(
            intermediate_dim=intermediate_dim,
            hidden_size=router_hidden_size,
            num_experts=self.num_experts
        )

        # Experts: multiple GRUs of different sizes
        # Each expert takes intermediate_dim and outputs intermediate_dim
        self.experts = nn.ModuleList([
            Expert(intermediate_dim=intermediate_dim, hidden_size=hidden_size)
            for hidden_size in expert_hidden_sizes
        ])

        # Output head: intermediate_dim → actions
        self.head = nn.Linear(intermediate_dim, num_actions)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Initialize hidden states for router and all experts.

        Returns:
            h_router: (1, batch, router_hidden_size)
            h_experts: List of (1, batch, expert_hidden_size) tensors
        """
        h_router = torch.zeros(1, batch_size, self.router_hidden_size, device=device)
        h_experts = [
            torch.zeros(1, batch_size, hidden_size, device=device)
            for hidden_size in self.expert_hidden_sizes
        ]
        return h_router, h_experts

    def forward(
        self,
        x: torch.Tensor,
        lang_embs: torch.Tensor,
        h: Tuple[torch.Tensor, List[torch.Tensor]],
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], Optional[dict]]:
        """
        Forward pass matching GRUPolicy interface.

        Args:
            x: (batch, input_dim) - observations
            lang_embs: (batch, lang_dim) - language embeddings
            h: (h_router, h_experts) - hidden states
            return_routing_info: If True, return routing weights and expert outputs

        Returns:
            logits: (batch, num_actions) - action logits
            h_new: (h_router_new, h_experts_new) - updated hidden states
            routing_info: Optional dict with routing details (if return_routing_info=True)
        """
        batch_size = x.shape[0]
        h_router, h_experts = h

        # Combine observation and language, project to intermediate_dim
        x_combined = torch.cat([x, lang_embs], dim=1)  # (batch, input_dim + lang_dim)
        x_proj = self.input_projection(x_combined)  # (batch, intermediate_dim)
        x_proj = x_proj.unsqueeze(1)  # (batch, 1, intermediate_dim)

        # Router: compute expert weights
        router_weights, h_router_new = self.router(x_proj, h_router)
        # router_weights: (batch, num_experts)

        # Process through all experts
        expert_outputs = []
        h_experts_new = []

        for i, expert in enumerate(self.experts):
            out, h_new = expert(x_proj, h_experts[i])
            # out: (batch, intermediate_dim)
            expert_outputs.append(out)
            h_experts_new.append(h_new)

        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, intermediate_dim)

        # Weight by router
        router_weights_expanded = router_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        weighted_outputs = expert_outputs * router_weights_expanded  # (batch, num_experts, intermediate_dim)

        # Sum across experts
        combined = weighted_outputs.sum(dim=1)  # (batch, intermediate_dim)

        # Final action logits
        logits = self.head(combined)  # (batch, num_actions)

        # Prepare return values
        h_new = (h_router_new, h_experts_new)

        if return_routing_info:
            routing_info = {
                'router_weights': router_weights,  # (batch, num_experts)
                'expert_outputs': expert_outputs,  # (batch, num_experts, intermediate_dim)
                'combined_output': combined,  # (batch, intermediate_dim)
            }
            return logits, h_new, routing_info
        else:
            return logits, h_new, None


# For backwards compatibility with the original interface
def reset_hidden_on_done(
    h: Tuple[torch.Tensor, List[torch.Tensor]],
    dones: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Reset hidden states for environments that are done.

    Args:
        h: (h_router, h_experts) - hidden states
        dones: (batch,) - boolean tensor indicating which envs are done

    Returns:
        h_reset: Hidden states with done envs zeroed out
    """
    h_router, h_experts = h

    # Create mask (convert bool to float)
    done_mask = dones.float().unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)

    # Reset router hidden state
    h_router_reset = h_router * (1.0 - done_mask)

    # Reset expert hidden states
    h_experts_reset = [
        h_expert * (1.0 - done_mask)
        for h_expert in h_experts
    ]

    return h_router_reset, h_experts_reset
