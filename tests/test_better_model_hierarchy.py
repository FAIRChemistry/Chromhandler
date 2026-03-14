import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers

from chromhandler.annotations import BaselineAnnotation, PeakAnnotation
from chromhandler.fitting.better_fitter import BetterFitter, _DEFAULT_SUBSET_NAME
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


def test_model_assembles_apex_and_separation_consistently_across_modes() -> None:
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(**_model_kwargs())

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


def test_free_doublet_separation_hierarchy_shapes_and_positivity() -> None:
    trace = handlers.trace(handlers.seed(model, rng_seed=1)).get_trace(**_model_kwargs())

    separation_free_min = np.asarray(trace["separation_free_min"]["value"])
    separation_free_typical = np.asarray(trace["separation_free_typical"]["value"])
    separation_free_trace_scale = np.asarray(
        trace["separation_free_trace_scale"]["value"]
    )
    separation_free_trace_offset = np.asarray(
        trace["separation_free_trace_offset"]["value"]
    )
    separation_free = np.asarray(trace["separation_free"]["value"])

    assert separation_free_min.shape == (1,)
    assert separation_free_typical.shape == (1,)
    assert separation_free_trace_offset.shape == (2, 1)
    assert separation_free.shape == (2, 1)
    assert separation_free_trace_scale.shape == ()
    assert float(separation_free_trace_scale) > 0.0
    assert np.all(separation_free_min > 0.0)
    assert np.all(separation_free_typical >= separation_free_min)
    assert np.all(separation_free >= separation_free_min[None, :])


def test_free_doublet_separation_broadcasts_typical_when_trace_offset_is_zero() -> None:
    trace = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=2),
            data={
                "separation_free_trace_offset": jnp.zeros((2, 1), dtype=jnp.float32),
            },
        )
    ).get_trace(**_model_kwargs())

    separation_free_typical = np.asarray(trace["separation_free_typical"]["value"])
    separation_free = np.asarray(trace["separation_free"]["value"])

    assert np.allclose(
        separation_free,
        np.broadcast_to(separation_free_typical[None, :], separation_free.shape),
    )


def test_free_doublet_separation_stays_above_min_for_large_negative_offsets() -> None:
    trace = handlers.trace(
        handlers.substitute(
            handlers.seed(model, rng_seed=3),
            data={
                "separation_free_trace_offset": jnp.full((2, 1), -20.0, dtype=jnp.float32),
            },
        )
    ).get_trace(**_model_kwargs())

    separation_free_min = np.asarray(trace["separation_free_min"]["value"])
    separation_free = np.asarray(trace["separation_free"]["value"])

    assert np.all(separation_free >= separation_free_min[None, :])
    assert np.all(separation_free > 0.0)


def test_trace_parameter_names_expose_pooled_separation_diagnostics() -> None:
    assert "separation_free_typical" in TRACE_PARAMETER_NAMES
    assert "separation_free_trace_scale" in TRACE_PARAMETER_NAMES
    assert "separation" in TRACE_PARAMETER_NAMES
    assert "separation_free_trace_offset" not in TRACE_PARAMETER_NAMES
