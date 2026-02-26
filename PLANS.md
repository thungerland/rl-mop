# Future Plans & Ideas

## Routing Visualization

### Dual-target quadrant grouping for compound instructions

For sequential BabyAI tasks (e.g. `BeforeInstr`, `AfterInstr`, `AndInstr`), the current
`by_agent_and_target_quadrant` grouping uses `instr_a` (the first sub-task's target).

A useful extension would be to also offer a grouping based on `instr_b` (the second
sub-task's target), and potentially a combined view showing both targets simultaneously.

**What to add:**

- A second helper `_second_target_pos(instr)` in `eval_mop.py` that recurses into
  `instr_b` instead of `instr_a`, storing the result as `target_pos_b` in `env_context`.
- A new analysis type `'by_agent_and_second_target_quadrant'` using `target_pos_b` for
  the quadrant calculation — all other logic (label functions, grouping, plotting) can
  reuse the existing quadrant infrastructure.
- Or alternatively, parameterise `_first_target_pos` to accept a `which='a'|'b'` argument.

**Relevant files:**
- `eval_mop.py` — `_first_target_pos()` and `_extract_env_context()`
- `plotting_utils.py` — `get_available_analyses()`, `group_routing_data()`, label dispatch
- `routing_viz.py` — `analysis_options` dict, plotting cell, save cell

**Instruction semantics to keep in mind:**

| Instruction | `instr_a` | `instr_b` | Execution order |
|---|---|---|---|
| `BeforeInstr` | first task | second task | `a` → `b` |
| `AfterInstr` | final goal (second) | prerequisite (first) | `b` → `a` |
| `AndInstr` | one task | other task | either order |

Current `_first_target_pos` always recurses `instr_a` before `instr_b`, which is correct
for `BeforeInstr` but wrong for `AfterInstr` (where `instr_b` happens first). When
implementing the second-target extension, `AfterInstr` will need special handling to
return targets in the correct execution order rather than the grammatical order.
