# Mixture-of-Pathways (MoP) Integration Guide

## Overview

This guide documents the integration of the Mixture-of-Pathways (MoP) model from [thungerland/mixture-of-pathways](https://github.com/thungerland/mixture-of-pathways) into the BabyAI training setup.

## What Was Done

### 1. Files Created/Modified

#### New Files:
- `mop_model.py` - Core MoP model (Expert, Router, HeterogeneousMoE, Block, Model)
- `mop_config.py` - Configuration system for MoP
- `stateful_mop.py` - **Stateful wrapper** that adapts MoP for step-by-step RL training
- `train_mop.py` - Modified training script using MoP (cloned from train.py)
- `test_mop_integration.py` - Integration test suite

#### Preserved Files:
- `train.py` - Original GRU-based training (unchanged, preserved as baseline)

### 2. Key Innovation: Stateful Wrapper

The original MoP model was designed for **sequence-based processing** (entire episodes at once). Our BabyAI setup uses **step-by-step processing** with explicit hidden state management for online RL.

**Solution**: Created `StatefulMoPPolicy` wrapper that:
- Exposes hidden states from ALL internal GRUs (routers + experts)
- Manages hidden state updates across timesteps
- Handles episode boundary resets
- Provides interface matching the original `GRUPolicy`: `forward(obs, lang, h) -> (logits, h_new)`

### 3. Architecture Details

#### Hidden State Management:
```python
hidden_states = {
    'router_0': [1, batch, router_dim],
    'expert_0_0': [1, batch, expert_dim_0],
    'expert_0_1': [1, batch, expert_dim_1],
    ...
    'router_1': [1, batch, router_dim],
    'expert_1_0': [1, batch, expert_dim_0],
    ...
}
```

Each block (layer) has:
- 1 router GRU (routes inputs to experts)
- N expert GRUs (process inputs with different capacities)

#### Episode Boundaries:
When episodes end, all hidden states for those environments are zeroed:
```python
for key in hidden_states.keys():
    done_mask = dones_tensor.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    hidden_states[key] = hidden_states[key] * (1.0 - done_mask)
```

### 4. Configuration

#### Example BabyAI MoP Config:
```python
config = create_babyai_mop_config(
    input_dim=280,  # 3*7*7 + 4 + 1 + 128 (image + dir + carry + language)
    num_actions=7,  # BabyAI action space
    intermediate_dim=256,  # Hidden representation size
    router_dim=64,  # Router GRU hidden size
    layers=["32,64,128", "64,128,256"],  # 2 blocks, 3 experts each
    device="cuda"
)
```

#### Layer Configuration:
- `layers=["32,64,128"]` means:
  - 1 block with 3 experts
  - Expert 0: GRU hidden size 32
  - Expert 1: GRU hidden size 64
  - Expert 2: GRU hidden size 128

- `layers=["32,64,128", "64,128,256"]` means:
  - Block 0: 3 experts (32, 64, 128)
  - Block 1: 3 experts (64, 128, 256)

### 5. Training Loop Changes

#### Original (train.py):
```python
policy = GRUPolicy(...)
h = torch.zeros(1, num_envs, hidden_size)

for update in range(num_updates):
    h, loss, acc = train_unroll(policy, optimizer, vec_env, h, unroll_len, device)
```

#### With MoP (train_mop.py):
```python
mop_config = create_babyai_mop_config(...)
policy = StatefulMoPPolicy(mop_config)
hidden_states = policy.init_hidden_states(num_envs, device)

for update in range(num_updates):
    hidden_states, loss, acc = train_unroll_mop(
        policy, optimizer, vec_env, hidden_states, unroll_len, device
    )
```

### 6. Key Implementation Details

#### GRU Dimension Handling:
The original MoP GRUs use **default PyTorch ordering** (not `batch_first=True`):
- Input: `(seq_len, batch, features)`
- Hidden: `(num_layers, batch, hidden_dim)`

Our wrapper transposes appropriately:
```python
x_transposed = x.transpose(0, 1)  # batch-first -> seq-first
output, h_new = gru(x_transposed, h)
output = output.transpose(0, 1)  # seq-first -> batch-first
```

#### Expert Processing:
All experts process the full batch (even if routing weights are zero) to maintain consistent hidden state shapes:
```python
# Process full batch through expert
expert_output_full, h_new = expert.rnn(input_full, h_expert)

# Apply routing mask after processing
if mask.any():
    weighted_output = gating_scores * expert_output_full[mask]
    final_output[mask] += weighted_output
```

## How to Use

### Basic Training:
```bash
python train_mop.py --config config.yaml
```

### With Custom MoP Architecture:
```bash
python train_mop.py \
    --config config.yaml \
    --mop_layers "16,32,64;32,64,128" \
    --mop_intermediate_dim 256 \
    --mop_router_dim 64
```

### Testing Integration:
```bash
python test_mop_integration.py
```

## Extracting Routing Information

For analysis, you can extract routing maps and expert activations:

```python
logits, routing_weights, hidden_states = policy.get_routing_info(
    obs, lang_embs, hidden_states
)

# routing_weights[layer_idx] shows which experts were used at each layer
```

**Note**: Full routing extraction functionality is marked as TODO in the current implementation. The infrastructure is in place but needs to be extended.

## Configuration Parameters

### In YAML config file:
```yaml
# Original parameters (still used)
task_id: "BabyAI-GoToRedBall-v0"
num_envs: 16
unroll_len: 32
num_updates: 10000
lr: 0.0003
input_dim: 152  # without language
lang_dim: 128

# New MoP-specific parameters
mop_layers: ["32,64,128", "64,128,256"]
mop_intermediate_dim: 256
mop_router_dim: 64
```

### Command-line overrides:
```bash
--mop_layers "16,32;32,64"  # Semicolon-separated for multiple blocks
--mop_intermediate_dim 256
--mop_router_dim 64
```

## Benefits of This Integration

1. ✅ **Preserves working baseline** - Original train.py untouched
2. ✅ **Maintains step-by-step training** - No need to refactor for sequences
3. ✅ **Full hidden state management** - Proper recurrence across timesteps
4. ✅ **Episode boundary handling** - Correct resets when episodes end
5. ✅ **Routing analysis ready** - Infrastructure for extracting routing maps
6. ✅ **Flexible architecture** - Easy to experiment with different expert configurations

## Testing

The `test_mop_integration.py` script validates:
1. Config creation
2. Model initialization
3. Hidden state initialization
4. Forward passes
5. Episode boundary handling
6. Multiple timestep sequences

All tests pass ✓

## Next Steps

### Immediate:
1. Train a small model to verify learning dynamics
2. Compare performance with baseline GRU policy
3. Implement full routing map extraction for analysis

### Future:
1. Add complexity penalty losses (currently set to 0)
2. Experiment with different expert configurations
3. Visualize routing decisions over episodes
4. Add expert activation analysis tools

## Troubleshooting

### Common Issues:

**Issue**: Dimension mismatch errors
- **Cause**: MoP GRUs don't use `batch_first=True`
- **Solution**: Wrapper handles transposes automatically

**Issue**: Hidden states not resetting on episode boundaries
- **Cause**: Forgot to apply done mask
- **Solution**: Use `train_unroll_mop` which handles this correctly

**Issue**: Out of memory
- **Cause**: Too many/large experts
- **Solution**: Reduce `mop_layers` (fewer or smaller experts)

## References

- Original MoP repo: https://github.com/thungerland/mixture-of-pathways
- MoP model file: `mop_model.py`
- Stateful wrapper: `stateful_mop.py`
- Training script: `train_mop.py`
