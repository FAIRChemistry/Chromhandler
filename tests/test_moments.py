import numpy as np

from chromhandler.fitting.moments import (
    PeakMomentMetrics,
    compute_peak_moment_metrics,
    compute_peak_moment_metrics_from_peak_masks,
    estimate_skew_normal_prior_hints,
)


def _metric(
    *,
    centroid: float,
    sigma: float,
    skewness: float,
    area: float,
) -> PeakMomentMetrics:
    return PeakMomentMetrics(
        window_low=0.0,
        window_high=1.0,
        area=area,
        apex_time=centroid,
        apex_height=1.0,
        centroid=centroid,
        sigma=sigma,
        skewness=skewness,
        centroid_apex_z=0.0,
        tail_ratio=1.0,
        log_tail_ratio=0.0,
        left_sigma=sigma,
        right_sigma=sigma,
        start_time=0.0,
        end_time=1.0,
        baseline_slope=0.0,
        baseline_intercept=0.0,
    )


def test_compute_peak_moment_metrics_uses_external_baseline_only() -> None:
    x = np.linspace(0.0, 10.0, 200)
    y = 2.0 * x + 1.0

    no_baseline = compute_peak_moment_metrics(
        x_values=x,
        y_values=y,
        window_low=0.0,
        window_high=10.0,
        baseline_slope=0.0,
        baseline_intercept=0.0,
    )
    assert np.isfinite(no_baseline.area)
    assert no_baseline.area > 1.0

    exact_baseline = compute_peak_moment_metrics(
        x_values=x,
        y_values=y,
        window_low=0.0,
        window_high=10.0,
        baseline_slope=2.0,
        baseline_intercept=1.0,
    )
    assert np.isnan(exact_baseline.area)


def test_compute_peak_moment_metrics_from_peak_masks_uses_trace_baseline() -> None:
    x = np.linspace(0.0, 10.0, 300)
    x_matrix = np.stack([x, x], axis=0)

    peak_shape = np.exp(-0.5 * ((x - 5.0) / 0.25) ** 2)
    y_trace_0 = 0.4 * x + 1.0 + 6.0 * peak_shape
    y_trace_1 = -0.2 * x + 3.0 + 5.0 * peak_shape
    y_matrix = np.stack([y_trace_0, y_trace_1], axis=0)

    peak_mask = (x >= 4.2) & (x <= 5.8)
    peak_masks = np.zeros((1, 2, x.size), dtype=bool)
    peak_masks[0, 0, :] = peak_mask
    peak_masks[0, 1, :] = peak_mask

    slopes = np.array([0.4, -0.2], dtype=float)
    intercepts = np.array([1.0, 3.0], dtype=float)

    metrics_by_peak = compute_peak_moment_metrics_from_peak_masks(
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        peak_masks=peak_masks,
        baseline_slopes=slopes,
        baseline_intercepts=intercepts,
    )

    assert len(metrics_by_peak) == 1
    assert len(metrics_by_peak[0]) == 2
    for trace_index, metrics in enumerate(metrics_by_peak[0]):
        assert np.isfinite(metrics.area)
        assert 4.8 <= metrics.centroid <= 5.2
        assert np.isclose(metrics.baseline_slope, slopes[trace_index])
        assert np.isclose(metrics.baseline_intercept, intercepts[trace_index])


def test_estimate_skew_normal_prior_hints_uses_skewness_direction() -> None:
    pos_peak_metrics = [
        _metric(centroid=3.0, sigma=0.15, skewness=0.60, area=5.0),
        _metric(centroid=3.1, sigma=0.18, skewness=0.50, area=5.5),
    ]
    neg_peak_metrics = [
        _metric(centroid=7.0, sigma=0.20, skewness=-0.55, area=9.0),
        _metric(centroid=6.9, sigma=0.22, skewness=-0.45, area=8.5),
    ]

    hints = estimate_skew_normal_prior_hints([pos_peak_metrics, neg_peak_metrics])
    assert len(hints) == 2

    assert np.isfinite(hints[0].alpha_loc)
    assert np.isfinite(hints[1].alpha_loc)
    assert hints[0].alpha_loc > 0.0
    assert hints[1].alpha_loc < 0.0
    assert hints[0].sigma_loc > 0.0
    assert hints[1].area_loc > 0.0
