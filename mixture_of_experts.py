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
        - If hidden_size > 0: Input → GRU → Linear → Output (intermediate_dim)
        - If hidden_size == 0: Identity/skip connection (returns input unchanged)
    """

    def __init__(self, intermediate_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim
        self.identity = hidden_size == 0

        if not self.identity:
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
            h: (1, batch, hidden_size) - hidden state (ignored if identity)

        Returns:
            out: (batch, intermediate_dim) - expert output
            h_new: (1, batch, hidden_size) - new hidden state (unchanged if identity)
        """
        if self.identity:
            # Skip connection: return input unchanged
            return x.squeeze(1), h

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


class MoELayer(nn.Module):
    """
    A single Mixture of Experts layer containing a router and multiple experts.

    This layer takes intermediate_dim input and produces intermediate_dim output,
    making it stackable.
    """

    def __init__(
        self,
        intermediate_dim: int,
        expert_hidden_sizes: List[int],
        router_hidden_size: int
    ):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.num_experts = len(expert_hidden_sizes)
        self.expert_hidden_sizes = expert_hidden_sizes
        self.router_hidden_size = router_hidden_size

        # Router: decides which experts to use
        self.router = Router(
            intermediate_dim=intermediate_dim,
            hidden_size=router_hidden_size,
            num_experts=self.num_experts
        )

        # Experts: multiple GRUs of different sizes
        self.experts = nn.ModuleList([
            Expert(intermediate_dim=intermediate_dim, hidden_size=hidden_size)
            for hidden_size in expert_hidden_sizes
        ])

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Initialize hidden states for this layer's router and experts.

        Returns:
            h_router: (1, batch, router_hidden_size)
            h_experts: List of (1, batch, expert_hidden_size) tensors
                       (empty tensor for identity experts with hidden_size=0)
        """
        h_router = torch.zeros(1, batch_size, self.router_hidden_size, device=device)
        h_experts = [
            torch.zeros(1, batch_size, hidden_size, device=device) if hidden_size > 0
            else torch.zeros(1, batch_size, 0, device=device)  # Placeholder for identity experts
            for hidden_size in self.expert_hidden_sizes
        ]
        return h_router, h_experts

    def forward(
        self,
        x: torch.Tensor,
        h: Tuple[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        """
        Forward pass for one MoE layer.

        Args:
            x: (batch, 1, intermediate_dim) - input
            h: (h_router, h_experts) - hidden states for this layer

        Returns:
            output: (batch, intermediate_dim) - layer output
            h_new: (h_router_new, h_experts_new) - updated hidden states
            router_weights: (batch, num_experts) - routing weights
            expert_outputs: (batch, num_experts, intermediate_dim) - individual expert outputs
        """
        h_router, h_experts = h

        # Router: compute expert weights
        router_weights, h_router_new = self.router(x, h_router)

        # Process through all experts
        expert_outputs_list = []
        h_experts_new = []

        for i, expert in enumerate(self.experts):
            out, h_new = expert(x, h_experts[i])
            expert_outputs_list.append(out)
            h_experts_new.append(h_new)

        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs_list, dim=1)  # (batch, num_experts, intermediate_dim)

        # Weight by router and sum
        router_weights_expanded = router_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        weighted_outputs = expert_outputs * router_weights_expanded
        output = weighted_outputs.sum(dim=1)  # (batch, intermediate_dim)

        h_new = (h_router_new, h_experts_new)
        return output, h_new, router_weights, expert_outputs


class MixtureOfExpertsPolicy(nn.Module):
    """
    Multi-layer Mixture of Experts policy that matches the GRUPolicy interface.

    Architecture:
    1. Input layer: maps (obs + lang) to intermediate_dim
    2. Multiple MoE layers, each with its own router and experts
    3. Output head: maps intermediate_dim to actions

    Usage (same as GRUPolicy):
        policy = MixtureOfExpertsPolicy(...)
        h = policy.init_hidden(batch_size, device)

        for t in range(steps):
            logits, h = policy(obs, lang, h)

    Configuration:
        - Single layer (original behavior):
            expert_hidden_sizes=[32, 64, 128]  # 3 experts in 1 layer

        - Multi-layer with same experts per layer:
            expert_hidden_sizes=[[32, 64], [32, 64], [32, 64]]  # 2 experts in 3 layers

        - Multi-layer with different experts per layer:
            expert_hidden_sizes=[[32, 64], [64, 128], [128, 256]]
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        expert_hidden_sizes: List,  # List[int] for single layer, List[List[int]] for multi-layer
        router_hidden_size: int,
        num_actions: int,
        lang_dim: int = 128
    ):
        """
        Args:
            input_dim: Observation dimension (without language)
            intermediate_dim: Intermediate representation dimension
            expert_hidden_sizes: Expert hidden sizes. Can be:
                - List[int]: Single layer with these expert sizes (e.g., [32, 64, 128])
                - List[List[int]]: Multiple layers, each with its own expert sizes
                  (e.g., [[32, 64], [64, 128]] for 2 layers)
            router_hidden_size: Hidden size for router GRUs (same for all layers)
            num_actions: Action space size
            lang_dim: Language embedding dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.lang_dim = lang_dim
        self.intermediate_dim = intermediate_dim
        self.router_hidden_size = router_hidden_size

        # Normalize expert_hidden_sizes to List[List[int]]
        if expert_hidden_sizes and isinstance(expert_hidden_sizes[0], int):
            # Single layer: [32, 64, 128] -> [[32, 64, 128]]
            self.layer_expert_sizes = [expert_hidden_sizes]
        else:
            # Multi-layer: [[32, 64], [64, 128], ...]
            self.layer_expert_sizes = expert_hidden_sizes

        self.num_layers = len(self.layer_expert_sizes)

        # Input projection: (obs + lang) → intermediate_dim
        self.input_projection = nn.Linear(input_dim + lang_dim, intermediate_dim)

        # MoE layers
        self.moe_layers = nn.ModuleList([
            MoELayer(
                intermediate_dim=intermediate_dim,
                expert_hidden_sizes=layer_sizes,
                router_hidden_size=router_hidden_size
            )
            for layer_sizes in self.layer_expert_sizes
        ])

        # Output head: intermediate_dim → actions
        self.head = nn.Linear(intermediate_dim, num_actions)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Initialize hidden states for all layers.

        Returns:
            List of (h_router, h_experts) tuples, one per layer
        """
        return [layer.init_hidden(batch_size, device) for layer in self.moe_layers]

    def forward(
        self,
        x: torch.Tensor,
        lang_embs: torch.Tensor,
        h: List[Tuple[torch.Tensor, List[torch.Tensor]]],
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, List[torch.Tensor]]], Optional[dict]]:
        """
        Forward pass matching GRUPolicy interface.

        Args:
            x: (batch, input_dim) - observations
            lang_embs: (batch, lang_dim) - language embeddings
            h: List of (h_router, h_experts) tuples, one per layer
            return_routing_info: If True, return routing weights and expert outputs per layer

        Returns:
            logits: (batch, num_actions) - action logits
            h_new: List of updated hidden states per layer
            routing_info: Optional dict with per-layer routing details
        """
        # Combine observation and language, project to intermediate_dim
        x_combined = torch.cat([x, lang_embs], dim=1)  # (batch, input_dim + lang_dim)
        current = self.input_projection(x_combined)  # (batch, intermediate_dim)

        h_new_all = []
        routing_info = {} if return_routing_info else None

        # Pass through each MoE layer
        for layer_idx, moe_layer in enumerate(self.moe_layers):
            # Prepare input for GRU (needs seq dim)
            current_seq = current.unsqueeze(1)  # (batch, 1, intermediate_dim)

            # Forward through layer
            current, h_layer_new, router_weights, expert_outputs = moe_layer(
                current_seq, h[layer_idx]
            )

            h_new_all.append(h_layer_new)

            if return_routing_info:
                routing_info[f'layer_{layer_idx}'] = {
                    'router_weights': router_weights,  # (batch, num_experts)
                    'expert_outputs': expert_outputs,  # (batch, num_experts, intermediate_dim)
                    'layer_output': current,  # (batch, intermediate_dim)
                }

        # Final action logits
        logits = self.head(current)  # (batch, num_actions)

        return logits, h_new_all, routing_info


def reset_hidden_on_done(
    h: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    dones: torch.Tensor
) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Reset hidden states for environments that are done.

    Args:
        h: List of (h_router, h_experts) tuples, one per layer
        dones: (batch,) - boolean tensor indicating which envs are done

    Returns:
        h_reset: Hidden states with done envs zeroed out
    """
    # Create mask (convert bool to float)
    done_mask = dones.float().unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)

    h_reset_all = []
    for h_router, h_experts in h:
        # Reset router hidden state
        h_router_reset = h_router * (1.0 - done_mask)

        # Reset expert hidden states
        h_experts_reset = [
            h_expert * (1.0 - done_mask)
            for h_expert in h_experts
        ]

        h_reset_all.append((h_router_reset, h_experts_reset))

    return h_reset_all
