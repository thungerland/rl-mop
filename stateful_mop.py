"""
Stateful wrapper around the Mixture of Pathways (MoP) model.

This wrapper makes the MoP model compatible with step-by-step RL training
by exposing and managing hidden states from all internal GRUs (routers and experts).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from mop_model import Model
from mop_config import Config


class StatefulMoPPolicy(nn.Module):
    """
    Wraps the MoP Model to provide a stateful interface compatible with
    the step-by-step training approach.

    Interface matches GRUPolicy:
    - forward(x, lang_embs, h) -> (logits, h_new)

    Where h contains all hidden states from routers and experts.
    """

    def __init__(self, mop_config: Config):
        """
        Initialize the stateful MoP policy.

        Args:
            mop_config: Configuration for the MoP model
        """
        super().__init__()

        self.mop_config = mop_config
        self.mop_model = Model(mop_config)

        # Store architecture info for hidden state management
        self.num_blocks = len(mop_config.layers)
        self.experts_per_block = [
            [int(dim) for dim in layer.split(",")]
            for layer in mop_config.layers
        ]

    def init_hidden_states(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize all hidden states for routers and experts.

        Args:
            batch_size: Number of parallel environments
            device: Device to create tensors on

        Returns:
            Dictionary containing hidden states for all GRUs
        """
        hidden_states = {}

        for block_idx in range(self.num_blocks):
            # Router hidden state (router_dim)
            hidden_states[f'router_{block_idx}'] = torch.zeros(
                1, batch_size, self.mop_config.router_dim,
                device=device
            )

            # Expert hidden states (one per expert, each with its own hidden_dim)
            expert_dims = self.experts_per_block[block_idx]
            for expert_idx, expert_dim in enumerate(expert_dims):
                if expert_dim > 0:  # Skip identity experts
                    hidden_states[f'expert_{block_idx}_{expert_idx}'] = torch.zeros(
                        1, batch_size, expert_dim,
                        device=device
                    )

        return hidden_states

    def forward(
        self,
        x: torch.Tensor,
        lang_embs: torch.Tensor,
        hidden_states: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass compatible with step-by-step training.

        Args:
            x: Observations, shape (batch, input_dim)
            lang_embs: Language embeddings, shape (batch, lang_dim)
            hidden_states: Dictionary of hidden states for all GRUs

        Returns:
            logits: Action logits, shape (batch, num_actions)
            new_hidden_states: Updated hidden states
        """
        batch_size = x.shape[0]
        device = x.device

        # Combine obs and language embeddings
        x_combined = torch.cat([x, lang_embs], dim=1)  # (batch, input_dim + lang_dim)

        # Add seq_len dimension: (batch, 1, input_dim + lang_dim)
        x_seq = x_combined.unsqueeze(1)

        # Process through the model with hidden state injection
        # We need to manually step through each block and inject/extract hidden states
        new_hidden_states = {}

        # Task IDs (all zeros for single-task BabyAI)
        task_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Input projection
        if self.mop_model.num_tasks > 1 and not self.mop_config.disable_task_embedding_layer:
            x_proj = torch.cat(
                [x_seq[:, :, :33], self.mop_model.embedding(task_ids)],
                dim=-1
            )
        else:
            x_proj = x_seq

        x_proj = self.mop_model.input_layer(x_proj)

        # Process through each block with hidden state management
        for block_idx, block in enumerate(self.mop_model.blocks):
            # Process this block (router + experts)
            x_proj, router_output, new_h_router, new_h_experts = self._forward_block_stateful(
                block,
                x_proj,
                task_ids,
                hidden_states,
                block_idx
            )

            # Store updated hidden states
            new_hidden_states[f'router_{block_idx}'] = new_h_router
            for expert_idx, h_expert in enumerate(new_h_experts):
                if h_expert is not None:
                    new_hidden_states[f'expert_{block_idx}_{expert_idx}'] = h_expert

        # Output projection
        logits = self.mop_model.output_layer(x_proj)  # (batch, 1, num_actions)
        logits = logits.squeeze(1)  # (batch, num_actions)

        return logits, new_hidden_states

    def _forward_block_stateful(
        self,
        block,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        hidden_states: Dict[str, torch.Tensor],
        block_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through a single block with explicit hidden state management.

        Args:
            block: The Block module
            x: Input tensor (batch, 1, intermediate_dim)
            task_ids: Task IDs (batch, 1)
            hidden_states: All hidden states
            block_idx: Index of this block

        Returns:
            output: Block output
            router_output: Router weights
            new_h_router: Updated router hidden state
            new_h_experts: List of updated expert hidden states
        """
        # Get router hidden state for this block
        h_router = hidden_states[f'router_{block_idx}']

        # Get expert hidden states for this block
        h_experts = []
        for expert_idx, expert_dim in enumerate(self.experts_per_block[block_idx]):
            if expert_dim > 0:
                h_experts.append(hidden_states[f'expert_{block_idx}_{expert_idx}'])
            else:
                h_experts.append(None)  # Identity expert

        # Forward through router with hidden state
        # GRU expects (seq_len, batch, features) and h of shape (num_layers, batch, hidden_dim)
        # x is currently (batch, seq_len, features), so transpose
        router = block.hmoe.router
        x_transposed = x.transpose(0, 1)  # (seq_len, batch, features)
        h_router_transposed = h_router  # Already (1, batch, router_dim)

        logits, h_router_new = router.rnn(x_transposed, h_router_transposed)
        logits = logits.transpose(0, 1)  # Back to (batch, seq_len, router_dim)
        logits = router.relu(logits)
        logits = router.output_layer(logits)

        indices = torch.zeros(
            (logits.shape[0], logits.shape[1], len(router.expert_dims)),
            dtype=torch.long,
            device=logits.device,
        )
        for i in range(len(router.expert_dims)):
            indices[:, :, i] = i

        raw_router_output = torch.nn.functional.softmax(logits, dim=-1)
        router_output = raw_router_output.clone()

        # Process through experts with hidden states
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))

        new_h_experts = []
        batch_size = x.shape[0]

        for i, expert in enumerate(block.hmoe.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            # Forward through expert with hidden state
            if self.experts_per_block[block_idx][i] == 0:
                # Identity expert - no hidden state
                if flat_mask.any():
                    expert_input = flat_x[flat_mask].unsqueeze(1)
                    expert_output = expert_input.squeeze(1)
                    gating_scores = flat_router_output[flat_mask, i]
                    weighted_output = torch.einsum("i,ij->ij", gating_scores, expert_output)
                    final_output[expert_mask] += weighted_output
                new_h_experts.append(None)
            else:
                # Get the hidden state for this expert (full batch)
                h_expert = h_experts[i]  # Shape: [1, batch_size, hidden_dim]

                # Process ALL batch elements through the expert (not just masked ones)
                # This is necessary to maintain consistent hidden state shapes
                expert_input_full = flat_x.view(batch_size, 1, -1)  # [batch, 1, dim]

                # GRU expects (seq_len, batch, features)
                expert_input_transposed = expert_input_full.transpose(0, 1)  # [1, batch, dim]
                expert_output_full, h_expert_new = expert.rnn(expert_input_transposed, h_expert)
                expert_output_full = expert_output_full.transpose(0, 1)  # Back to [batch, 1, hidden_dim]

                # Process through batch norm, relu, and output layer
                expert_output_full = expert_output_full.squeeze(1)  # [batch, hidden_dim]
                if expert_output_full.var() != 0:
                    expert_output_full = expert.batchnorm(expert_output_full)
                expert_output_full = expert.relu(expert_output_full)
                expert_output_full = expert.output_layer(expert_output_full)  # [batch, intermediate_dim]

                # Now apply the gating and masking
                if flat_mask.any():
                    expert_output_masked = expert_output_full[expert_mask.squeeze(1)]
                    gating_scores = flat_router_output[flat_mask, i]
                    weighted_output = torch.einsum("i,ij->ij", gating_scores, expert_output_masked)
                    final_output[expert_mask] += weighted_output

                new_h_experts.append(h_expert_new)

        # Apply layer norm with residual
        output = block.ln(x + final_output)

        return output, router_output, h_router_new, new_h_experts

    def get_routing_info(
        self,
        x: torch.Tensor,
        lang_embs: torch.Tensor,
        hidden_states: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass that also returns routing information for analysis.

        Args:
            x: Observations, shape (batch, input_dim)
            lang_embs: Language embeddings, shape (batch, lang_dim)
            hidden_states: Dictionary of hidden states

        Returns:
            logits: Action logits
            routing_weights: List of routing weights per layer
            new_hidden_states: Updated hidden states
        """
        # For now, just return regular forward pass
        # TODO: Extend this to capture routing weights
        logits, new_hidden_states = self.forward(x, lang_embs, hidden_states)
        routing_weights = []  # Placeholder
        return logits, routing_weights, new_hidden_states
