# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np
import pytest

from scalecov import block_bootstrap_r2_within, r2_unit_means, r2_within, rho_within
from scalecov.simulate import nested_field


@pytest.fixture(scope="module")
def sim():
    return nested_field(n_groups=12, units_per_group=3, unit_shape=(10, 10), random_state=5)


def test_r2_within_perfect_and_flat(sim):
    assert r2_within(sim.truth, sim.truth, sim.unit) == pytest.approx(1.0)
    flat = sim.groupby("unit").truth.transform("mean")
    assert r2_within(sim.truth, flat, sim.unit) == pytest.approx(0.0)


def test_r2_within_invariant_to_unit_shift(sim):
    shift = sim.unit.map({u: float(u) * 3.0 for u in sim.unit.unique()})
    assert r2_within(sim.truth, sim.pred + shift, sim.unit) == pytest.approx(r2_within(sim.truth, sim.pred, sim.unit))


def test_unit_mean_r2_perfect():
    y = np.r_[np.ones(5), 3 * np.ones(5)]
    assert r2_unit_means(y, y + np.random.default_rng(0).normal(0, 0.1, 10) * 0, np.repeat([0, 1], 5)) == 1.0


def test_rho_within_perfect(sim):
    assert rho_within(sim.truth, sim.truth, sim.unit) == pytest.approx(1.0)


def test_block_bootstrap_brackets_estimate(sim):
    r = block_bootstrap_r2_within(sim.truth, sim.pred, sim.unit, sim.group, n_boot=500)
    assert r["lo"] <= r["estimate"] <= r["hi"]
    assert r["estimate"] == pytest.approx(r2_within(sim.truth, sim.pred, sim.unit))
