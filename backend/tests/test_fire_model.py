from __future__ import annotations

import numpy as np

from app.simulation.fire_model import BURNED, BURNING, FireModel, FireModelConfig


def _model(seed: int, wind_speed: float = 8.0, stochastic: bool = True) -> FireModel:
    rng = np.random.default_rng(seed)
    cfg = FireModelConfig(cell_size_m=50.0, stochastic=stochastic, base_ros_mps=0.6)
    m = FireModel.create(
        origin_x=0.0,
        origin_y=0.0,
        extent_x=4000.0,
        extent_y=4000.0,
        config=cfg,
        rng=rng,
    )
    m.ignite("fire_01", 2000.0, 2000.0, -37.5, 145.3, 0.9, 0.0, 80.0)
    return m, wind_speed


def test_ignition_sets_burning_cells() -> None:
    m, _ = _model(42)
    assert np.any(m.state == BURNING)
    assert m.sources[0].active
    assert m.sources[0].fire_id == "fire_01"


def test_spread_increases_affected_area() -> None:
    m, wind = _model(42)
    initial = int(np.sum((m.state == BURNING) | (m.state == BURNED)))
    for t in range(40):
        m.step(1.0, wind, 315.0, float(t + 1))
    later = int(np.sum((m.state == BURNING) | (m.state == BURNED)))
    assert later > initial


def test_deterministic_seed() -> None:
    def run(seed: int) -> np.ndarray:
        m, wind = _model(seed)
        for t in range(25):
            m.step(1.0, wind, 315.0, float(t + 1))
        return m.state.copy()

    a = run(42)
    b = run(42)
    c = run(7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_wind_biases_spread_downwind() -> None:
    """Strong westerly wind (from 270°) should spread more to the east."""

    def burned_east_west(wind_from: float) -> tuple[int, int]:
        rng = np.random.default_rng(0)
        cfg = FireModelConfig(cell_size_m=50.0, stochastic=False, base_ros_mps=0.8, lb_wind_coeff=0.4)
        m = FireModel.create(0.0, 0.0, 4000.0, 4000.0, cfg, rng)
        m.ignite("f", 2000.0, 2000.0, -37.5, 145.3, 1.0, 0.0, 50.0)
        for t in range(50):
            m.step(1.0, 12.0, wind_from, float(t + 1))
        mid = m.width // 2
        east = int(np.sum(((m.state == BURNING) | (m.state == BURNED))[:, mid:]))
        west = int(np.sum(((m.state == BURNING) | (m.state == BURNED))[:, :mid]))
        return east, west

    east, west = burned_east_west(270.0)  # from west → downwind east
    assert east > west
