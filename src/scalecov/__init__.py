# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""ScaleCov (coverage across scales): calibrate and check prediction intervals of spatial maps across nested scales."""
from . import simulate
from ._version import __version__
from .aggregate_only import coverage_vs_kappa, k_dense_coverage, kappa, unit_quantile
from .conformal import PixelConformal, conformal_quantile, unit_weights
from .diagnostics import (
                          check_folds,
                          coverage_interval,
                          cross_calibrate,
                          nominal_range,
                          pixel_coverage_by_unit,
                          summarise,
)
from .metrics import block_bootstrap_r2_within, r2_unit_means, r2_within, rho_within, within_sums
from .units import METHODS, UnitIntervals
from .variogram import ExpCovariance, fit_within_unit_covariance, mean_error_variance

__all__ = [
    "METHODS",
    "ExpCovariance",
    "PixelConformal",
    "UnitIntervals",
    "__version__",
    "block_bootstrap_r2_within",
    "check_folds",
    "conformal_quantile",
    "coverage_interval",
    "coverage_vs_kappa",
    "cross_calibrate",
    "fit_within_unit_covariance",
    "k_dense_coverage",
    "kappa",
    "mean_error_variance",
    "nominal_range",
    "pixel_coverage_by_unit",
    "r2_unit_means",
    "r2_within",
    "rho_within",
    "simulate",
    "summarise",
    "unit_quantile",
    "unit_weights",
    "within_sums",
]
