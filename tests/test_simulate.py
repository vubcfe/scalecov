# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np

from scalecov.simulate import nested_field


def test_shape_and_identity():
    d = nested_field(n_groups=3, units_per_group=2, unit_shape=(5, 4), random_state=0)
    assert len(d) == 3 * 2 * 20
    assert np.allclose(d.truth - d.pred, d.error)
    assert d.unit.nunique() == 6 and d.group.nunique() == 3


def test_unit_mean_error_variance_matches_components():
    # unit-mean error variance = sg^2 + su^2 + Var(mean of spatial field) + nugget^2 / N (no spatial part here)
    d = nested_field(n_groups=400, units_per_group=2, unit_shape=(5, 5), sigma_group=0.3, sigma_unit=0.6,
                     sigma_spatial=0.0, nugget=0.5, random_state=1)
    v = d.groupby("unit").error.mean().var()
    assert abs(v - (0.3 ** 2 + 0.6 ** 2 + 0.5 ** 2 / 25)) < 0.06
