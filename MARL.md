# MARL interface

## Policy

```python
action = policy.act(observation, agent_id, state=None)  # discrete-9 int
```

Algorithms (MAPPO, MADDPG, Independent DQN, IPPO, …) wrap this interface. Nothing in the simulator is hardcoded to one algorithm.

`app/marl/inference.py` loads a PyTorch actor checkpoint (`state_dict` / `actor` / `model`).

Named policies and default files under `backend/data/checkpoints/`:

| Policy | File | Algorithm |
| --- | --- | --- |
| `dqn` | `dqn.pt` | Independent DQN |
| `mappo` | `mappo.pt` | MAPPO (central critic, decentralized actor) |
| `maddpg` | `maddpg.pt` | MADDPG + Gumbel-Softmax (discrete-9) |

MAPPO and MADDPG are trained by `scripts/train_mappo_maddpg.py` to **overfly and extinguish**, not only detect:

1. Expert rollouts from **greedy fire-seeking** (nearest burning cell; HOVER on the front). Lawnmower is not cloned.
2. Near-fire HOVER is oversampled. Behavioral cloning repeats until holdout extinguished-fire count is measured ≥ 1.5 / 2 when possible.
3. MAPPO and MADDPG fine-tune with a BC regularizer and decaying expert mixing so they do not forget suppression.
4. The checkpoint kept is the one with the highest measured holdout extinguish count. Numbers are not invented.

If a file is missing or incompatible, `UntrainedMARLPolicy` acts randomly and is labeled `trained=False`. Do not read that as a MARL result.

## Observation (execution, default)

Length 57, `float32`, partial observability:

| Index | Content |
| ---: | --- |
| 0–10 | Own pose, battery, wind, normalized time |
| 11–19 | Up to 3 **known active** fires (dx, dy, intensity) |
| 20–31 | Up to 3 UAVs **inside communication range** |
| 32–56 | 5×5 local coverage patch |

Known fires at execution time are own true detections plus successfully delivered messages — **not** the global fire list — unless `policy.allow_global_state=True` (training privilege; off by default). Extinguished or burned-out sources are dropped so agents seek remaining fires.

Normalization: positions / AOI extent; heading as sin/cos of navigation heading; battery in [0, 1]; wind / 40 m/s; time / duration.

## Action

Discrete-9: N, NE, E, SE, S, SW, W, NW, HOVER. Mapped to target heading and cruise speed, then constrained by `uav_dynamics.apply_action`.

HOVER at a fire only extinguishes if the UAV is inside `suppression_range_m`.

## Reward (matches the UI equation)

```
R = α N_detected + ζ N_cells_out + η N_fires_out + β Δcoverage − γ Δenergy − δ N_collisions − ε Δoverlap
```

Default: α=8, ζ=0.35, η=16, β=0.4, γ=0.04, δ=8, ε=0.3.

Detection is not suppression. Cells become `EXTINGUISHED` only when a UAV is inside suppression range of a `BURNING` cell.

Reward is a **team** optimisation signal shared by all agents in the step. It is **not** classification accuracy.

## Training vs execution

| | Training | Execution |
| --- | --- | --- |
| Code path | `WildfireEnv.reset/step` | same |
| Visualization | none | MapLibre / optional Cesium |
| Global state | only if `allow_global_state` | same flag, default false |
| Policy | learner writes checkpoint | `Policy.act` reads observation only |
