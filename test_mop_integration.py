"""
Test script to verify MoP integration with BabyAI.

This script tests:
1. MoP model initialization
2. Forward pass with single timestep
3. Hidden state management
4. Episode boundary handling
"""

import torch
import numpy as np
from mop_config import Config
from stateful_mop import StatefulMoPPolicy


def test_mop_config():
    """Test creating a BabyAI-compatible MoP config."""
    print("=" * 60)
    print("Test 1: Creating BabyAI MoP Config")
    print("=" * 60)

    # Create config directly without importing train_mop
    config_dict = {
        'input_dim': 280,
        'output_dim': 7,
        'intermediate_dim': 256,
        'router_dim': 64,
        'layers': ["16,32", "32,64"],
        'device': 'cpu',
        'task_id': 'babyai',
        'task_dim': 32,
        'disable_task_embedding_layer': True,
        'disable_wandb': True,
        'disable_fixation_loss': True,
        'disable_task_performance_scaling': True,
        'expert_cost_exponent': 2.0,
        'cost_based_loss_alpha': 0.0,
        'cost_based_loss_epsilon': 0.0,
        'dropout_max_prob': None,
        'dropout_router_weight_threshold': None,
        'early_stopping_threshold': None,
        'ephemeral': False,
        'learning_rate': 0.001,
        'num_epochs': 1,
        'num_steps': 1000,
        'batch_size': 16,
        'checkpoint': None,
        'run_id': 'babyai_mop',
    }

    config = Config.from_dict(config_dict, migrate=False)

    print(f"✓ Config created successfully")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Output dim: {config.output_dim}")
    print(f"  Intermediate dim: {config.intermediate_dim}")
    print(f"  Router dim: {config.router_dim}")
    print(f"  Layers: {config.layers}")
    print()

    return config


def test_mop_initialization(config):
    """Test initializing the StatefulMoPPolicy."""
    print("=" * 60)
    print("Test 2: Initializing StatefulMoPPolicy")
    print("=" * 60)

    policy = StatefulMoPPolicy(config)

    print(f"✓ Policy initialized successfully")
    print(f"  Number of blocks: {policy.num_blocks}")
    print(f"  Experts per block: {policy.experts_per_block}")

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    return policy


def test_hidden_state_init(policy):
    """Test initializing hidden states."""
    print("=" * 60)
    print("Test 3: Initializing Hidden States")
    print("=" * 60)

    batch_size = 16
    device = torch.device("cpu")

    hidden_states = policy.init_hidden_states(batch_size, device)

    print(f"✓ Hidden states initialized for batch_size={batch_size}")
    print(f"  Number of hidden state tensors: {len(hidden_states)}")

    for key, tensor in hidden_states.items():
        print(f"  {key}: {tensor.shape}")

    print()
    return hidden_states


def test_forward_pass(policy, hidden_states):
    """Test a forward pass through the policy."""
    print("=" * 60)
    print("Test 4: Forward Pass")
    print("=" * 60)

    batch_size = 16
    input_dim = 152  # 3*7*7 + 4 + 1 = 152 (without language)
    lang_dim = 128
    device = torch.device("cpu")

    # Create dummy observations and language embeddings
    obs = torch.randn(batch_size, input_dim, device=device)
    lang_embs = torch.randn(batch_size, lang_dim, device=device)

    print(f"  Input observation shape: {obs.shape}")
    print(f"  Language embedding shape: {lang_embs.shape}")

    # Forward pass
    try:
        logits, new_hidden_states = policy(obs, lang_embs, hidden_states)

        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Expected shape: ({batch_size}, 7)")

        # Verify shapes
        assert logits.shape == (batch_size, 7), f"Unexpected logits shape: {logits.shape}"
        assert len(new_hidden_states) == len(hidden_states), "Hidden state count mismatch"

        print(f"✓ All shapes correct")
        print()

        return logits, new_hidden_states

    except Exception as e:
        print(f"✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        raise


def test_episode_boundary(policy, hidden_states):
    """Test resetting hidden states on episode boundaries."""
    print("=" * 60)
    print("Test 5: Episode Boundary Handling")
    print("=" * 60)

    batch_size = 16
    device = torch.device("cpu")

    # Simulate episode done for environments 0, 5, 10
    done_mask = np.zeros(batch_size, dtype=bool)
    done_mask[[0, 5, 10]] = True
    done_tensor = torch.from_numpy(done_mask.astype(int)).to(device)

    print(f"  Resetting hidden states for envs: {np.where(done_mask)[0].tolist()}")

    # Reset hidden states
    for key in hidden_states.keys():
        mask = done_tensor.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        hidden_states[key] = hidden_states[key] * (1.0 - mask)

    # Check that the correct hidden states are zero
    for key, tensor in hidden_states.items():
        # Check envs that should be reset
        for env_idx in [0, 5, 10]:
            if tensor[0, env_idx].abs().sum() > 1e-6:
                print(f"✗ Hidden state {key} not properly reset for env {env_idx}")
                return False

        # Check envs that should NOT be reset
        for env_idx in [1, 2, 3, 4, 6, 7, 8, 9]:
            if tensor[0, env_idx].abs().sum() < 1e-6:
                print(f"✗ Hidden state {key} incorrectly reset for env {env_idx}")
                return False

    print(f"✓ Episode boundary handling works correctly")
    print()
    return True


def test_multiple_steps(policy):
    """Test multiple forward passes to simulate an unroll."""
    print("=" * 60)
    print("Test 6: Multiple Forward Passes (Simulating Unroll)")
    print("=" * 60)

    batch_size = 4  # Smaller for clarity
    input_dim = 152
    lang_dim = 128
    device = torch.device("cpu")
    unroll_len = 5

    # Initialize
    hidden_states = policy.init_hidden_states(batch_size, device)

    print(f"  Running {unroll_len} forward passes...")

    for t in range(unroll_len):
        obs = torch.randn(batch_size, input_dim, device=device)
        lang_embs = torch.randn(batch_size, lang_dim, device=device)

        logits, hidden_states = policy(obs, lang_embs, hidden_states)

        print(f"    Step {t+1}/{unroll_len}: logits shape = {logits.shape}")

    print(f"✓ Multiple forward passes successful")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("#" * 60)
    print("# MoP Integration Test Suite")
    print("#" * 60)
    print()

    try:
        # Run tests
        config = test_mop_config()
        policy = test_mop_initialization(config)
        hidden_states = test_hidden_state_init(policy)
        logits, new_hidden_states = test_forward_pass(policy, hidden_states)
        test_episode_boundary(policy, new_hidden_states)
        test_multiple_steps(policy)

        print("#" * 60)
        print("# ALL TESTS PASSED ✓")
        print("#" * 60)
        print()

    except Exception as e:
        print()
        print("#" * 60)
        print("# TESTS FAILED ✗")
        print("#" * 60)
        print()
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
