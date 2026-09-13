# Metrics

Every dashboard value is computed in `app/simulation/metrics.py` from live state. `null` → UI **N/A**.

The HTTP catalog `GET /api/metrics/catalog` is the machine-readable list (name, definition, unit, source).

## Detection

| Metric | Definition |
| --- | --- |
| Detection time (per fire) | `first_true_detection_time − ignition_time` |
| Average detection time | Mean of the above over **detected** fires |
| Time to first detection | Time of the first true detection from `t = 0` |
| Detection rate | `detected_fires / total_fires` |
| Miss rate | `undetected_fires / total_fires` |
| False positive rate | `false_positive_reports / all_reports` |

## Coverage and fire area

| Metric | Definition |
| --- | --- |
| Coverage | `explored_eligible / eligible` |
| Explored area | explored eligible cells × cell area (m²) |
| Eligible area | AOI minus inaccessible |
| Burned area | BURNING ∪ BURNED cells × fire cell area |
| Extinguished fires | Sources whose last BURNING cells were put out by a UAV inside suppression range |
| Extinguished area | EXTINGUISHED cells × fire cell area (natural BURNED excluded) |
| Suppression rate | extinguished_fires / total_fires |

A green detection is not an extinguish. Those counts are separate.

## UAV

Distance is integrated from the kinematic update. Mean speed = total distance / airborne time. Battery remaining = `battery_j / capacity`. Energy used = `1 − mean battery`.

## Multi-agent

| Metric | Definition |
| --- | --- |
| Communication connectivity | active undirected links / possible pairs |
| Messages sent/received/failures | counts from the communication model |
| Redundant exploration | eligible cells seen by ≥2 UAVs / eligible |
| Mean pairwise Jaccard | mean Jaccard of per-UAV explored sets |
| Min inter-agent distance | minimum planimetric UAV–UAV distance so far |
| Workload CV | std/mean of per-UAV covered cell counts |

Coordination-efficiency slogans without a formula are **not** implemented.

## RL (deployment panel)

Episode reward, mean step reward, reward variance, episode length. `policy_entropy`, `value_loss`, `policy_loss`, `success_rate` are **N/A** unless a trainer supplies them. They are training diagnostics, not detection accuracy.
