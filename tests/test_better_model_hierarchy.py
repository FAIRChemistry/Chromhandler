import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers
import pytest
from pydantic import ValidationError

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.better_fitter import _DEFAULT_SUBSET_NAME, BetterFitter
from chromhandler.fitting.better_model import TRACE_PARAMETER_NAMES, model


def _model_kwargs() -> dict[str, jnp.ndarray | None]:
    return {
        "x": jnp.asarray(
            np.broadcast_to(np.linspace(0.0, 1.0, 5, dtype=float)[None, :], (2, 5))
        ),
        "y": None,
        "peak_mode_code": jnp.asarray([0, 1, 2], dtype=jnp.int32),
        "artefact_side": jnp.asarray([0, 1, 0], dtype=jnp.int32),
        "artefact_peak_index": jnp.asarray([1], dtype=jnp.int32),
        "free_peak_index": jnp.asarray([2], dtype=jnp.int32),
        "nonfree_peak_index": jnp.asarray([0, 1], dtype=jnp.int32),
        "apex_loc": jnp.asarray([0.22, 0.48, 0.76], dtype=jnp.float32),
        "apex_scale": jnp.asarray([0.01, 0.02, 0.03], dtype=jnp.float32),
        "trace_shift_scale": jnp.asarray(0.02, dtype=jnp.float32),
        "sigma_loc": jnp.asarray([0.03, 0.05, 0.06], dtype=jnp.float32),
        "sigma_scale": jnp.asarray([0.005, 0.006, 0.008], dtype=jnp.float32),
        "alpha_loc": jnp.asarray([0.15, 0.20, 0.25], dtype=jnp.float32),
        "alpha_scale": jnp.asarray([0.03, 0.03, 0.04], dtype=jnp.float32),
        "dominant_area_loc_per_trace": jnp.asarray(
            [[8.0, 7.0, 6.0], [8.5, 7.5, 6.5]],
            dtype=jnp.float32,
        ),
        "area_total_loc_per_trace": jnp.asarray(
            [[10.0, 9.0, 8.0], [10.5, 9.5, 8.5]],
            dtype=jnp.float32,
        ),
        "artefact_area_loc_shared": jnp.asarray([1.2], dtype=jnp.float32),
        # free peak (local pos 0) is fixed by default
        "free_fixed_local_index": jnp.asarray([0], dtype=jnp.int32),
        "free_vary_local_index": jnp.asarray([], dtype=jnp.int32),
        "window_lo": jnp.asarray([0.10, 0.30, 0.60], dtype=jnp.float32),
        "window_hi": jnp.asarray([0.35, 0.65, 0.92], dtype=jnp.float32),
        "baseline_intercept_loc": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "baseline_intercept_scale": jnp.asarray([0.1, 0.1], dtype=jnp.float32),
        "baseline_slope_loc": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "baseline_slope_scale": jnp.asarray([0.01, 0.01], dtype=jnp.float32),
        "sigma_y_prior_loc": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    }


def _build_default_view() -> BetterFitter:
    x = np.linspace(0.0, 6.0, 400, dtype=float)
    shifts = np.asarray([-0.030, 0.000, 0.018, 0.045], dtype=float)
    scales = np.asarray([1.00, 0.92, 1.08, 0.97], dtype=float)
    peak_centers = np.asarray([1.15, 2.55, 4.20], dtype=float)
    peak_widths = np.asarray([0.055, 0.070, 0.080], dtype=float)
    peak_heights = np.asarray([1.30, 0.95, 0.75], dtype=float)

    signal_rows: list[np.ndarray] = []
    for shift, scale in zip(shifts, scales, strict=True):
        y = np.full_like(x, 0.05)
        for center, width, height in zip(
            peak_centers,
            peak_widths,
            peak_heights,
            strict=True,
        ):
            z = (x - (center + shift)) / width
            y = y + scale * height * np.exp(-0.5 * z**2)
        signal_rows.append(y)

    peaks = [
        PeakAnnotation(molecule_id="p_single", rt_min=0.95, rt_max=1.35, mode="single"),
        PeakAnnotation(
            molecule_id="p_art",
            rt_min=2.30,
            rt_max=2.85,
            mode="artefact_doublet",
            artefact_side="right",
        ),
        PeakAnnotation(
            molecule_id="p_free",
            rt_min=3.90,
            rt_max=4.50,
            mode="free_doublet",
        ),
    ]
    baselines = [
        BaselineAnnotation(rt_min=0.10, rt_max=0.45),
        BaselineAnnotation(rt_min=5.30, rt_max=5.80),
    ]

    parent = BetterFitter(
        np.broadcast_to(x[None, :], (len(signal_rows), x.size)).copy(),
        np.asarray(signal_rows, dtype=float),
        peaks=peaks,
        baselines=baselines,
    )
    return parent._make_subset_view(_DEFAULT_SUBSET_NAME)


def test_compute_model_inputs_exposes_hierarchical_apex_priors() -> None:
    fitter = _build_default_view()

    inputs = fitter.compute_model_inputs()

    assert inputs["apex_loc"].shape == (3,)
    assert inputs["apex_scale"].shape == (3,)
    assert np.all(np.isfinite(inputs["apex_loc"]))
    assert np.all(np.isfinite(inputs["apex_scale"]))
    assert np.all(inputs["apex_scale"] > 0.0)
    assert np.isfinite(float(inputs["trace_shift_scale"]))
    assert float(inputs["trace_shift_scale"]) > 0.0


def test_compute_model_inputs_includes_window_bounds() -> None:
    fitter = _build_default_view()

    inputs = fitter.compute_model_inputs()

    assert "window_lo" in inputs
    assert "window_hi" in inputs
    assert inputs["window_lo"].shape == (3,)
    assert inputs["window_hi"].shape == (3,)
    assert np.all(inputs["window_hi"] > inputs["window_lo"])


def test_model_assembles_apex_and_separation_consistently_across_modes() -> None:
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(
        **_model_kwargs()
    )

    apex = np.asarray(trace["apex"]["value"])
    apex_l = np.asarray(trace["apex_l"]["value"])
    apex_r = np.asarray(trace["apex_r"]["value"])
    separation = np.asarray(trace["separation"]["value"])
    separation_free = np.asarray(trace["separation_free"]["value"])
    trace_shift = np.asarray(trace["trace_shift"]["value"])
    apex_residual = np.asarray(trace["apex_residual"]["value"])

    assert np.allclose(np.mean(trace_shift), 0.0, atol=1e-6)
    assert apex.shape == apex_l.shape == apex_r.shape == apex_residual.shape == (2, 3)
    assert np.all(separation >= 0.0)

    assert np.allclose(apex_l[:, 0], apex[:, 0])
    assert np.allclose(apex_r[:, 0], apex[:, 0])
    assert np.allclose(separation[:, 0], 0.0)

    assert np.allclose(apex_l[:, 1], apex[:, 1])
    assert np.allclose(apex_r[:, 1] - apex_l[:, 1], separation[:, 1])

    assert np.allclose(apex[:, 2], 0.5 * (apex_l[:, 2] + apex_r[:, 2]))
    assert np.allclose(apex_r[:, 2] - apex_l[:, 2], separation[:, 2])
    assert np.allclose(separation[:, [2]], separation_free)


def _model_kwargs_vary() -> dict[str, jnp.ndarray | None]:
    """Fixture with the free peak set to vary_separation=True."""
    kwargs = _model_kwargs()
    kwargs["free_fixed_local_index"] = jnp.asarray([], dtype=jnp.int32)
    kwargs["free_vary_local_index"] = jnp.asarray([0], dtype=jnp.int32)
    return kwargs


def test_free_doublet_separation_fixed_shapes_and_bounds() -> None:
    """Fixed separation: all traces share the typical; no trace-scale sampled."""
    trace = handlers.trace(handlers.seed(model, rng_seed=1)).get_trace(
        **_model_kwargs()  # default: free_fixed_local_index=[0], free_vary=[]
    )

    separation_free_min = np.asarray(trace["separation_free_min"]["value"])
    separation_free_max = np.asarray(trace["separation_free_max"]["value"])
    separation_free_typical = np.asarray(trace["separation_free_typical"]["value"])
    separation_free = np.asarray(trace["separation_free"]["value"])

    assert separation_free_min.shape == (1,)
    assert separation_free_max.shape == (1,)
    assert separation_free_typical.shape == (1,)
    assert separation_free.shape == (2, 1)

    assert np.all(separation_free_min > 0.0)
    assert np.all(separation_free_max > separation_free_min)
    assert np.all(separation_free_typical >= separation_free_min)
    assert np.all(separation_free_typical <= separation_free_max)

    # All traces share the exact same separation
    assert np.allclose(separation_free[0], separation_free[1])
    assert np.allclose(separation_free, separation_free_typical[None, :])

    # sep_trace_scale is NOT in the trace when no varying peaks
    assert "sep_trace_scale" not in trace


def test_free_doublet_separation_vary_shapes_and_bounds() -> None:
    """Varying separation: per-peak trace-scale sampled; traces differ."""
    trace = handlers.trace(handlers.seed(model, rng_seed=1)).get_trace(
        **_model_kwargs_vary()
    )

    separation_free_min = np.asarray(trace["separation_free_min"]["value"])
    separation_free_max = np.asarray(trace["separation_free_max"]["value"])
    sep_trace_scale = np.asarray(trace["sep_trace_scale"]["value"])
    separation_free_trace_offset = np.asarray(
        trace["separation_free_trace_offset"]["value"]
    )
    separation_free = np.asarray(trace["separation_free"]["value"])

    assert sep_trace_scale.shape == (1,)  # per-peak, NOT scalar
    assert separation_free_trace_offset.shape == (2, 1)
    assert separation_free.shape == (2, 1)

    assert float(sep_trace_scale[0]) > 0.0
    assert np.all(separation_free >= separation_free_min[None, :])
    assert np.all(separation_free <= separation_free_max[None, :])


def test_free_doublet_separation_broadcasts_typical_when_trace_offset_is_zero() -> None:
    trace = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=2),
            data={
                "separation_free_trace_offset": jnp.zeros((2, 1), dtype=jnp.float32),
            },
        )
    ).get_trace(**_model_kwargs_vary())

    separation_free_typical = np.asarray(trace["separation_free_typical"]["value"])
    separation_free = np.asarray(trace["separation_free"]["value"])

    assert np.allclose(
        separation_free,
        np.broadcast_to(separation_free_typical[None, :], separation_free.shape),
    )


def test_free_doublet_separation_stays_within_bounds_for_extreme_offsets() -> None:
    kwargs = _model_kwargs_vary()

    # Large negative offsets → separation approaches sep_min
    trace_neg = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=3),
            data={
                "separation_free_trace_offset": jnp.full(
                    (2, 1), -20.0, dtype=jnp.float32
                ),
            },
        )
    ).get_trace(**kwargs)

    sep_min = np.asarray(trace_neg["separation_free_min"]["value"])
    sep_max = np.asarray(trace_neg["separation_free_max"]["value"])
    sep_neg = np.asarray(trace_neg["separation_free"]["value"])

    assert np.all(sep_neg >= sep_min[None, :])
    assert np.all(sep_neg > 0.0)

    # Large positive offsets → separation approaches sep_max
    trace_pos = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=3),
            data={
                "separation_free_trace_offset": jnp.full(
                    (2, 1), 20.0, dtype=jnp.float32
                ),
            },
        )
    ).get_trace(**kwargs)

    sep_pos = np.asarray(trace_pos["separation_free"]["value"])
    assert np.all(sep_pos <= sep_max[None, :])


def test_separation_bounded_by_window_geometry() -> None:
    kwargs = _model_kwargs()
    # Narrow window for the free peak (index 2)
    kwargs["window_lo"] = jnp.asarray([0.10, 0.30, 0.72], dtype=jnp.float32)
    kwargs["window_hi"] = jnp.asarray([0.35, 0.65, 0.80], dtype=jnp.float32)

    trace = handlers.trace(handlers.seed(model, rng_seed=42)).get_trace(**kwargs)

    sep_max = np.asarray(trace["separation_free_max"]["value"])
    separation_free = np.asarray(trace["separation_free"]["value"])

    # sep_max should be 0.5 * (0.80 - 0.72) = 0.04
    assert np.allclose(sep_max, 0.04, atol=1e-6)
    assert np.all(separation_free <= sep_max[None, :])
    assert np.all(separation_free > 0.0)


def test_artefact_area_hierarchy_shapes_and_positivity() -> None:
    trace = handlers.trace(handlers.seed(model, rng_seed=5)).get_trace(
        **_model_kwargs()
    )

    area_artefact_typical = np.asarray(trace["area_artefact_typical"]["value"])
    area_artefact_trace_offset = np.asarray(
        trace["area_artefact_trace_offset"]["value"]
    )
    area_artefact = np.asarray(trace["area_artefact"]["value"])

    assert area_artefact_typical.shape == (1,)  # n_artefact=1 in fixture
    assert area_artefact_trace_offset.shape == (2, 1)  # [n_trace, n_artefact]
    assert area_artefact.shape == (2, 1)  # per-trace
    assert np.all(area_artefact_typical > 0.0)
    assert np.all(area_artefact > 0.0)


def test_artefact_area_equals_typical_when_offset_is_zero() -> None:
    trace = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=6),
            data={
                "area_artefact_trace_offset": jnp.zeros((2, 1), dtype=jnp.float32),
            },
        )
    ).get_trace(**_model_kwargs())

    area_artefact_typical = np.asarray(trace["area_artefact_typical"]["value"])
    area_artefact = np.asarray(trace["area_artefact"]["value"])

    assert np.allclose(
        area_artefact,
        np.broadcast_to(area_artefact_typical[None, :], area_artefact.shape),
    )


def test_artefact_area_varies_across_traces() -> None:
    trace = handlers.trace(handlers.seed(model, rng_seed=7)).get_trace(
        **_model_kwargs()
    )

    area_artefact = np.asarray(trace["area_artefact"]["value"])
    # With natural random offsets, traces should differ
    assert not np.allclose(area_artefact[0, 0], area_artefact[1, 0])


def test_trace_parameter_names_expose_pooled_separation_diagnostics() -> None:
    assert "separation_free_typical" in TRACE_PARAMETER_NAMES
    assert "sep_trace_scale" in TRACE_PARAMETER_NAMES
    assert "separation" in TRACE_PARAMETER_NAMES
    assert "area_artefact_typical" in TRACE_PARAMETER_NAMES
    assert "separation_free_trace_offset" not in TRACE_PARAMETER_NAMES
    assert "area_artefact_trace_offset" not in TRACE_PARAMETER_NAMES


def test_peak_annotation_vary_separation_defaults_false() -> None:
    ann = PeakAnnotation(molecule_id="x", rt_min=1.0, rt_max=2.0, mode="free_doublet")
    assert ann.vary_separation is False


def test_peak_annotation_vary_separation_true_only_on_free_doublet() -> None:
    import pytest

    with pytest.raises(ValueError, match="vary_separation=True is only valid"):
        PeakAnnotation(
            molecule_id="x", rt_min=1.0, rt_max=2.0, mode="single", vary_separation=True
        )


def test_peak_structure_splits_fixed_and_vary_local_indices() -> None:
    fitter = _build_default_view()

    structure = fitter.peak_structure()

    # p_free is at peak index 2, local pos 0 in free_peak_index, vary_separation=False
    assert list(structure["free_fixed_local_index"]) == [0]
    assert list(structure["free_vary_local_index"]) == []


def test_peak_structure_vary_separation_true_goes_to_vary_local() -> None:
    x = np.linspace(0.0, 6.0, 200, dtype=float)
    peaks = [
        PeakAnnotation(
            molecule_id="p0",
            rt_min=1.0,
            rt_max=2.0,
            mode="free_doublet",
            vary_separation=False,
        ),
        PeakAnnotation(
            molecule_id="p1",
            rt_min=3.0,
            rt_max=4.0,
            mode="free_doublet",
            vary_separation=True,
        ),
        PeakAnnotation(
            molecule_id="p2",
            rt_min=4.5,
            rt_max=5.5,
            mode="free_doublet",
            vary_separation=False,
        ),
    ]
    baselines = [BaselineAnnotation(rt_min=0.1, rt_max=0.4)]
    signal = np.random.default_rng(0).normal(1.0, 0.1, (3, x.size))
    parent = BetterFitter(
        np.broadcast_to(x[None, :], (3, x.size)).copy(),
        signal,
        peaks=peaks,
        baselines=baselines,
    )
    view = parent._make_subset_view(_DEFAULT_SUBSET_NAME)
    structure = view.peak_structure()

    # local positions: p0→0, p1→1, p2→2
    assert list(structure["free_fixed_local_index"]) == [0, 2]
    assert list(structure["free_vary_local_index"]) == [1]


def test_model_fixed_separation_equal_across_traces() -> None:
    """With all fixed peaks, every trace must have identical separation."""
    trace = handlers.trace(handlers.seed(model, rng_seed=10)).get_trace(
        **_model_kwargs()  # free_fixed=[0], free_vary=[]
    )
    separation_free = np.asarray(trace["separation_free"]["value"])
    assert np.allclose(separation_free[0], separation_free[1])


def test_model_vary_separation_differs_across_traces() -> None:
    """With varying peaks, different traces should (almost surely) differ."""
    trace = handlers.trace(handlers.seed(model, rng_seed=10)).get_trace(
        **_model_kwargs_vary()  # free_vary=[0]
    )
    separation_free = np.asarray(trace["separation_free"]["value"])
    assert not np.allclose(separation_free[0], separation_free[1])


# ---------------------------------------------------------------------------
# include_artefact_in_area flag — annotation validation
# ---------------------------------------------------------------------------


def test_include_artefact_in_area_default_is_false() -> None:
    p = PeakAnnotation(
        molecule_id="X", rt_min=1.0, rt_max=2.0,
        mode="artefact_doublet", artefact_side="right",
    )
    assert p.include_artefact_in_area is False


def test_include_artefact_in_area_accepted_on_artefact_doublet() -> None:
    p = PeakAnnotation(
        molecule_id="X", rt_min=1.0, rt_max=2.0,
        mode="artefact_doublet", artefact_side="right",
        include_artefact_in_area=True,
    )
    assert p.include_artefact_in_area is True


def test_include_artefact_in_area_rejected_on_single() -> None:
    with pytest.raises(ValidationError, match="include_artefact_in_area"):
        PeakAnnotation(
            molecule_id="X", rt_min=1.0, rt_max=2.0,
            mode="single",
            include_artefact_in_area=True,
        )


def test_include_artefact_in_area_rejected_on_free_doublet() -> None:
    with pytest.raises(ValidationError, match="include_artefact_in_area"):
        PeakAnnotation(
            molecule_id="X", rt_min=1.0, rt_max=2.0,
            mode="free_doublet",
            include_artefact_in_area=True,
        )


# ---------------------------------------------------------------------------
# include_artefact_in_area flag — _molecule_area_slice helper
# ---------------------------------------------------------------------------


def _art_peak(side: str, *, include: bool = False) -> PeakAnnotation:
    return PeakAnnotation(
        molecule_id="X", rt_min=1.0, rt_max=2.0,
        mode="artefact_doublet", artefact_side=side,
        include_artefact_in_area=include,
    )


def test_molecule_area_slice_artefact_right_excluded() -> None:
    """Artefact on right → dominant on left → return area_l."""
    peak = _art_peak("right", include=False)
    a_l = np.array([100.0, 200.0])
    a_r = np.array([10.0, 20.0])
    np.testing.assert_array_equal(BetterFitter._molecule_area_slice(peak, a_l, a_r), a_l)


def test_molecule_area_slice_artefact_left_excluded() -> None:
    """Artefact on left → dominant on right → return area_r."""
    peak = _art_peak("left", include=False)
    a_l = np.array([10.0, 20.0])
    a_r = np.array([100.0, 200.0])
    np.testing.assert_array_equal(BetterFitter._molecule_area_slice(peak, a_l, a_r), a_r)


def test_molecule_area_slice_artefact_included_sums_both() -> None:
    """With include_artefact_in_area=True → area_l + area_r regardless of side."""
    for side in ("left", "right"):
        peak = _art_peak(side, include=True)
        a_l = np.array([100.0, 200.0])
        a_r = np.array([10.0, 20.0])
        np.testing.assert_array_equal(
            BetterFitter._molecule_area_slice(peak, a_l, a_r), a_l + a_r
        )


def test_molecule_area_slice_single_returns_area_l() -> None:
    peak = PeakAnnotation(molecule_id="X", rt_min=1.0, rt_max=2.0, mode="single")
    a_l = np.array([50.0])
    a_r = np.array([0.0])
    np.testing.assert_array_equal(BetterFitter._molecule_area_slice(peak, a_l, a_r), a_l)


def test_molecule_area_slice_free_doublet_sums_both() -> None:
    peak = PeakAnnotation(molecule_id="X", rt_min=1.0, rt_max=2.0, mode="free_doublet")
    a_l = np.array([60.0, 70.0])
    a_r = np.array([40.0, 30.0])
    np.testing.assert_array_equal(
        BetterFitter._molecule_area_slice(peak, a_l, a_r), a_l + a_r
    )


def test_sigma_r_artefact_respects_loguniform_bounds() -> None:
    """sigma_r_artefact must stay within [0.5, 2.0] × sigma_prior_loc[artefact_idx]."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    kwargs = _model_kwargs()
    sigma_loc = np.asarray(kwargs["sigma_loc"])  # [0.03, 0.05, 0.06]
    artefact_idx = np.asarray(kwargs["artefact_peak_index"])  # [1]
    # artefact peaks are always nonfree → sigma_prior_loc = sigma_loc
    ref = sigma_loc[artefact_idx]  # [0.05]
    lower = 0.5 * ref
    upper = 2.0 * ref

    for seed in range(30):
        trace = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace(**kwargs)
        sigma_art = np.asarray(trace["sigma_r_artefact"]["value"])
        assert np.all(sigma_art >= lower - 1e-5), (
            f"seed={seed}: sigma_r_artefact={sigma_art} below lower={lower}"
        )
        assert np.all(sigma_art <= upper + 1e-5), (
            f"seed={seed}: sigma_r_artefact={sigma_art} above upper={upper}"
        )


def test_baseline_centring_flat_when_slope_is_zero() -> None:
    """When baseline_slope=0 for all traces, baseline_curve must equal baseline_intercept."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    kwargs = _model_kwargs()
    # Force all per-trace slope components to zero:
    # baseline_slope = pop_mean + pop_scale * raw
    # → set pop_mean=0 and raw=zeros  (pop_scale can be anything)
    trace = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=0),
            data={
                "baseline_slope_pop_mean": jnp.zeros((), dtype=jnp.float32),
                "baseline_slope_raw": jnp.zeros((2,), dtype=jnp.float32),
            },
        )
    ).get_trace(**kwargs)

    baseline_intercept = np.asarray(trace["baseline_intercept"]["value"])  # [n_trace]
    baseline_curve = np.asarray(trace["baseline_curve"]["value"])  # [n_trace, n_time]

    # When slope=0, baseline = intercept everywhere
    for t in range(baseline_intercept.shape[0]):
        np.testing.assert_allclose(
            baseline_curve[t],
            np.full(baseline_curve.shape[1], baseline_intercept[t]),
            rtol=1e-5,
            err_msg=f"trace {t}: baseline_curve not flat when slope=0",
        )


def test_sigma_r_free_respects_loguniform_bounds() -> None:
    """sigma_r_free must stay within [0.5, 2.0] × sigma_prior_loc[free_idx]."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    kwargs = _model_kwargs()
    sigma_loc = np.asarray(kwargs["sigma_loc"])  # [0.03, 0.05, 0.06]
    free_idx = np.asarray(kwargs["free_peak_index"])  # [2]
    peak_mode_code = np.asarray(kwargs["peak_mode_code"])
    free_mask = peak_mode_code == 2
    sigma_prior_loc = np.where(free_mask, 0.5 * sigma_loc, sigma_loc)
    ref = sigma_prior_loc[free_idx]  # 0.5 * 0.06 = 0.03
    lower = 0.5 * ref
    upper = 2.0 * ref

    for seed in range(30):
        trace = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace(**kwargs)
        sigma_free = np.asarray(trace["sigma_r_free"]["value"])
        assert np.all(sigma_free >= lower - 1e-5), (
            f"seed={seed}: sigma_r_free={sigma_free} below lower={lower}"
        )
        assert np.all(sigma_free <= upper + 1e-5), (
            f"seed={seed}: sigma_r_free={sigma_free} above upper={upper}"
        )


def test_hierarchical_slope_sites_exist_in_trace() -> None:
    """Model must produce baseline_slope_pop_mean, baseline_slope_pop_scale, baseline_slope_raw."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(**_model_kwargs())
    assert "baseline_slope_pop_mean" in trace, "missing baseline_slope_pop_mean"
    assert "baseline_slope_pop_scale" in trace, "missing baseline_slope_pop_scale"
    assert "baseline_slope_raw" in trace, "missing baseline_slope_raw"
    assert "baseline_slope" in trace, "baseline_slope deterministic missing"


def test_hierarchical_slope_per_trace_shapes() -> None:
    """baseline_slope_raw [n_trace], baseline_slope [n_trace], pop scalars."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    kwargs = _model_kwargs()  # n_trace=2
    trace = handlers.trace(handlers.seed(model, rng_seed=1)).get_trace(**kwargs)

    assert np.asarray(trace["baseline_slope_pop_mean"]["value"]).shape == ()
    assert np.asarray(trace["baseline_slope_pop_scale"]["value"]).shape == ()
    assert np.asarray(trace["baseline_slope_raw"]["value"]).shape == (2,)
    assert np.asarray(trace["baseline_slope"]["value"]).shape == (2,)


def test_hierarchical_slope_pop_scale_positive() -> None:
    """Population scale is always positive (HalfNormal)."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    for seed in range(20):
        trace = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace(**_model_kwargs())
        pop_scale = float(np.asarray(trace["baseline_slope_pop_scale"]["value"]))
        assert pop_scale > 0.0, f"seed={seed}: pop_scale={pop_scale} not positive"


def test_hierarchical_slope_in_summary_parameter_names() -> None:
    from chromhandler.fitting.better_model import SUMMARY_PARAMETER_NAMES
    assert "baseline_slope_pop_mean" in SUMMARY_PARAMETER_NAMES
    assert "baseline_slope_pop_scale" in SUMMARY_PARAMETER_NAMES


def test_sigma_base_respects_loguniform_bounds() -> None:
    """sigma_base must stay within [0.5, 2.0] × sigma_prior_loc for every prior sample."""
    import numpyro.handlers as handlers
    from chromhandler.fitting.better_model import model

    kwargs = _model_kwargs()
    sigma_loc = np.asarray(kwargs["sigma_loc"])  # [0.03, 0.05, 0.06]
    peak_mode_code = np.asarray(kwargs["peak_mode_code"])  # [0, 1, 2]
    free_mask = peak_mode_code == 2  # [False, False, True]
    sigma_prior_loc = np.where(free_mask, 0.5 * sigma_loc, sigma_loc)
    lower = 0.5 * sigma_prior_loc
    upper = 2.0 * sigma_prior_loc

    for seed in range(30):
        trace = handlers.trace(handlers.seed(model, rng_seed=seed)).get_trace(**kwargs)
        sigma_base = np.asarray(trace["sigma_base"]["value"])
        assert np.all(sigma_base >= lower - 1e-5), (
            f"seed={seed}: sigma_base={sigma_base} below lower={lower}"
        )
        assert np.all(sigma_base <= upper + 1e-5), (
            f"seed={seed}: sigma_base={sigma_base} above upper={upper}"
        )
