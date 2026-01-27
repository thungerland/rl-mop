"""
Simple test for the Mixture of Experts implementation.
"""

import torch
from mixture_of_experts import MixtureOfExpertsPolicy, reset_hidden_on_done


def test_basic_forward():
    """Test basic forward pass."""
    print("=" * 60)
    print("Test 1: Basic Forward Pass")
    print("=" * 60)

    # Setup
    batch_size = 4
    input_dim = 152  # 3*7*7 + 4 + 1
    lang_dim = 128
    intermediate_dim = 256
    expert_hidden_sizes = [32, 64, 128]
    router_hidden_size = 64
    num_actions = 7
    device = torch.device("cpu")

    # Create policy
    policy = MixtureOfExpertsPolicy(
        input_dim=input_dim,
        intermediate_dim=intermediate_dim,
        expert_hidden_sizes=expert_hidden_sizes,
        router_hidden_size=router_hidden_size,
        num_actions=num_actions,
        lang_dim=lang_dim
    )

    print(f"✓ Policy created")
    print(f"  Input dim: {input_dim}")
    print(f"  Intermediate dim: {intermediate_dim}")
    print(f"  Expert hidden sizes: {expert_hidden_sizes}")
    print(f"  Router hidden size: {router_hidden_size}")
    print(f"  Number of experts: {len(expert_hidden_sizes)}")

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Initialize hidden states
    h = policy.init_hidden(batch_size, device)
    h_router, h_experts = h

    print(f"\n✓ Hidden states initialized")
    print(f"  Router hidden: {h_router.shape}")
    for i, h_exp in enumerate(h_experts):
        print(f"  Expert {i} hidden: {h_exp.shape}")

    # Create dummy input
    obs = torch.randn(batch_size, input_dim, device=device)
    lang = torch.randn(batch_size, lang_dim, device=device)

    print(f"\n✓ Dummy input created")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Lang shape: {lang.shape}")

    # Forward pass
    logits, h_new, _ = policy(obs, lang, h)

    print(f"\n✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {num_actions})")

    assert logits.shape == (batch_size, num_actions), f"Unexpected logits shape: {logits.shape}"
    print(f"✓ Shape correct!")


def test_routing_info():
    """Test routing information extraction."""
    print("\n" + "=" * 60)
    print("Test 2: Routing Information Extraction")
    print("=" * 60)

    batch_size = 4
    input_dim = 152
    lang_dim = 128
    intermediate_dim = 256
    expert_hidden_sizes = [32, 64, 128]
    router_hidden_size = 64
    num_actions = 7
    device = torch.device("cpu")

    policy = MixtureOfExpertsPolicy(
        input_dim=input_dim,
        intermediate_dim=intermediate_dim,
        expert_hidden_sizes=expert_hidden_sizes,
        router_hidden_size=router_hidden_size,
        num_actions=num_actions,
        lang_dim=lang_dim
    )

    h = policy.init_hidden(batch_size, device)
    obs = torch.randn(batch_size, input_dim, device=device)
    lang = torch.randn(batch_size, lang_dim, device=device)

    # Forward with routing info
    logits, h_new, routing_info = policy(obs, lang, h, return_routing_info=True)

    print(f"✓ Routing information extracted")
    print(f"\nRouting weights shape: {routing_info['router_weights'].shape}")
    print(f"Expected: ({batch_size}, {len(expert_hidden_sizes)})")

    # Show actual routing weights
    print(f"\nExample routing weights (environment 0):")
    weights = routing_info['router_weights'][0]
    for i, w in enumerate(weights):
        print(f"  Expert {i}: {w.item():.4f}")

    print(f"\nSum of weights: {weights.sum().item():.4f} (should be 1.0)")

    print(f"\nExpert outputs shape: {routing_info['expert_outputs'].shape}")
    print(f"Expected: ({batch_size}, {len(expert_hidden_sizes)}, {intermediate_dim})")

    print(f"\nCombined output shape: {routing_info['combined_output'].shape}")
    print(f"Expected: ({batch_size}, {intermediate_dim})")

    assert routing_info['router_weights'].shape == (batch_size, len(expert_hidden_sizes))
    assert routing_info['expert_outputs'].shape == (batch_size, len(expert_hidden_sizes), intermediate_dim)
    assert routing_info['combined_output'].shape == (batch_size, intermediate_dim)

    print(f"\n✓ All routing info shapes correct!")


def test_episode_boundary():
    """Test episode boundary handling."""
    print("\n" + "=" * 60)
    print("Test 3: Episode Boundary Handling")
    print("=" * 60)

    batch_size = 4
    input_dim = 152
    lang_dim = 128
    intermediate_dim = 256
    expert_hidden_sizes = [32, 64, 128]
    router_hidden_size = 64
    num_actions = 7
    device = torch.device("cpu")

    policy = MixtureOfExpertsPolicy(
        input_dim=input_dim,
        intermediate_dim=intermediate_dim,
        expert_hidden_sizes=expert_hidden_sizes,
        router_hidden_size=router_hidden_size,
        num_actions=num_actions,
        lang_dim=lang_dim
    )

    # Initialize and do a forward pass to get non-zero hidden states
    h = policy.init_hidden(batch_size, device)
    obs = torch.randn(batch_size, input_dim, device=device)
    lang = torch.randn(batch_size, lang_dim, device=device)
    _, h, _ = policy(obs, lang, h)

    # Check hidden states are non-zero
    h_router, h_experts = h
    print(f"✓ Hidden states are non-zero")
    print(f"  Router hidden norm: {h_router.norm().item():.4f}")

    # Reset environments 1 and 3
    dones = torch.tensor([False, True, False, True], device=device)
    print(f"\n✓ Resetting environments: {dones.tolist()}")

    h_reset = reset_hidden_on_done(h, dones)
    h_router_reset, h_experts_reset = h_reset

    # Check that the correct environments were reset
    print(f"\n✓ Checking resets...")
    for env_idx in range(batch_size):
        router_norm = h_router_reset[0, env_idx].norm().item()
        if dones[env_idx]:
            assert router_norm == 0.0, f"Env {env_idx} should be reset"
            print(f"  Env {env_idx}: RESET (norm={router_norm:.4f})")
        else:
            assert router_norm > 0.0, f"Env {env_idx} should NOT be reset"
            print(f"  Env {env_idx}: ACTIVE (norm={router_norm:.4f})")

    print(f"\n✓ Episode boundary handling correct!")


def test_multiple_steps():
    """Test multiple forward passes (simulating an unroll)."""
    print("\n" + "=" * 60)
    print("Test 4: Multiple Forward Passes")
    print("=" * 60)

    batch_size = 4
    input_dim = 152
    lang_dim = 128
    intermediate_dim = 256
    expert_hidden_sizes = [32, 64, 128]
    router_hidden_size = 64
    num_actions = 7
    unroll_len = 5
    device = torch.device("cpu")

    policy = MixtureOfExpertsPolicy(
        input_dim=input_dim,
        intermediate_dim=intermediate_dim,
        expert_hidden_sizes=expert_hidden_sizes,
        router_hidden_size=router_hidden_size,
        num_actions=num_actions,
        lang_dim=lang_dim
    )

    h = policy.init_hidden(batch_size, device)

    print(f"✓ Running {unroll_len} forward passes...")

    for t in range(unroll_len):
        obs = torch.randn(batch_size, input_dim, device=device)
        lang = torch.randn(batch_size, lang_dim, device=device)

        logits, h, routing_info = policy(obs, lang, h, return_routing_info=True)

        weights = routing_info['router_weights']
        print(f"  Step {t+1}: logits {logits.shape}, avg routing: {weights.mean(dim=0).tolist()}")

    print(f"\n✓ Multiple forward passes successful!")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Mixture of Experts Test Suite")
    print("#" * 60 + "\n")

    try:
        test_basic_forward()
        test_routing_info()
        test_episode_boundary()
        test_multiple_steps()

        print("\n" + "#" * 60)
        print("# ALL TESTS PASSED ✓")
        print("#" * 60 + "\n")

    except Exception as e:
        print("\n" + "#" * 60)
        print("# TESTS FAILED ✗")
        print("#" * 60 + "\n")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
