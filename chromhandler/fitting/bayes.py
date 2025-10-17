from functools import partial
from typing import Any, Optional

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr
from numpyro.infer import MCMC, NUTS
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

numpyro.set_host_device_count(8)

num_devices = jax.local_device_count()
console = Console()

print(f"Number of devices: {num_devices}")

# =====================================================================
# EMG PDF functions
# =====================================================================


def emg_logpdf(
    x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, tau: jnp.ndarray
) -> jnp.ndarray:
    """Stable log-pdf of Exponentially Modified Gaussian (area=1).

    log f(x; μ, σ, τ) = -log(τ) + σ²/(2τ²) - (x-μ)/τ + log Φ((x-μ)/σ - σ/τ)

    Args:
        x: Data points, shape [N]
        mu: Centers, shape [K]
        sigma: Gaussian widths, shape [K]
        tau: Exponential tails, shape [K]

    Returns:
        Log-pdf matrix, shape [K, N]
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    mu = jnp.asarray(mu, dtype=x.dtype)
    sigma = jnp.asarray(sigma, dtype=x.dtype)
    tau = jnp.asarray(tau, dtype=x.dtype)

    s = (x[None, :] - mu[:, None]) / sigma[:, None]  # [K, N]
    k = (sigma / tau)[:, None]  # [K, 1]

    return (
        -jnp.log(tau)[:, None]
        + (sigma**2)[:, None] / (2.0 * tau**2)[:, None]
        - (x[None, :] - mu[:, None]) / tau[:, None]
        + log_ndtr(s - k)
    )


def emg_pdf_matrix(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    """EMG pdf matrix, shape [K, N]."""
    return jnp.exp(emg_logpdf(x, mu, sigma, tau))


def emg_mixture_area(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    """Sum of area-weighted EMG components, shape [N]."""
    pdfs = emg_pdf_matrix(x, mu, sigma, tau)
    return jnp.sum(pdfs * A[:, None], axis=0)


def emg_components_area(
    x: jnp.ndarray,
    A: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    """Per-component curves for plotting, shape [K, N]."""
    return emg_pdf_matrix(x, mu, sigma, tau) * A[:, None]


# =====================================================================
# NumPyro model
# =====================================================================


def emg_mixture_model(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray],
    mu_lo: jnp.ndarray,
    mu_hi: jnp.ndarray,
    sigma_min: float,
    sigma_max: float,
    r_min: float,
    r_max: jnp.ndarray,
    auc_hat: float,
    h_anchor: jnp.ndarray,
    alpha_dirichlet: float = 0.3,
    use_baseline: bool = False,
) -> None:
    """NumPyro model for EMG-mixture chromatographic peaks.

    Uses height-based parameterization with improved numerical stability.

    Key improvements over previous version:
    - Tighter priors on log_h using observed data
    - More informative priors on tau/ratio
    - Removed numpyro.factor (can cause gradient issues)
    - Better centering of transformations
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    mu_lo = jnp.asarray(mu_lo, dtype=jnp.float32)
    mu_hi = jnp.asarray(mu_hi, dtype=jnp.float32)
    r_max = jnp.asarray(r_max, dtype=jnp.float32)
    h_anchor = jnp.asarray(h_anchor, dtype=jnp.float32)

    K = mu_lo.shape[0]
    x_mean = jnp.mean(x)
    x_span = jnp.maximum(jnp.max(x) - jnp.min(x), jnp.array(1e-6, dtype=jnp.float32))

    # Handle prior predictive: if y is None, use h_anchor for scale
    if y is None:
        y_scale = jnp.maximum(jnp.max(h_anchor), jnp.array(1.0, dtype=jnp.float32))
    else:
        y = jnp.asarray(y, dtype=jnp.float32)
        y_scale = jnp.maximum(jnp.max(jnp.abs(y)), jnp.array(1e-6, dtype=jnp.float32))

    # ----------------------------------------------------------------
    # Peak centers: Uniform in user-specified windows
    # ----------------------------------------------------------------
    mu = numpyro.sample("mu", dist.Uniform(mu_lo, mu_hi))

    # ----------------------------------------------------------------
    # Gaussian widths: Better centered LogitNormal
    # ----------------------------------------------------------------
    # Center the sigmoid transformation at the midpoint
    sigma_range = sigma_max - sigma_min

    # Sample with bias toward middle
    sigma_raw = numpyro.sample("sigma_raw", dist.Normal(0.0, 1.5).expand([K]))
    sigma_01 = jax.nn.sigmoid(sigma_raw)
    sigma = sigma_min + sigma_range * sigma_01
    numpyro.deterministic("sigma", sigma)

    # ----------------------------------------------------------------
    # Tail ratios: Tighter prior centered on reasonable value
    # Most chromatographic peaks have ratio ~ 0.5-2.0
    # ----------------------------------------------------------------
    # Log-space prior on ratio (more natural for multiplicative scale)
    alpha_u, beta_u = 2.0, 5.0  # ← tweak if you want a different bias
    u = numpyro.sample("u_ratio", dist.Beta(alpha_u, beta_u).expand([K]))

    r = r_min + (r_max - r_min) * u  # [K]   now in physical units
    tau = sigma * r  # τ = ρ·σ
    numpyro.deterministic("tau", tau)
    numpyro.deterministic("ratio", r)

    # ----------------------------------------------------------------
    # HEIGHT-BASED PARAMETERIZATION
    # TIGHTER prior using observed data (reduces identifiability issues)
    # ----------------------------------------------------------------
    # Use TIGHTER prior on log-height (was 0.8, now 0.5)
    # This is key: we trust our height anchors more
    log_h = numpyro.sample(
        "log_h",
        dist.Normal(
            jnp.log(h_anchor),
            jnp.array(0.5, dtype=jnp.float32),  # TIGHTER (was 0.8)
        ).expand([K]),
    )
    h = jnp.exp(log_h)
    numpyro.deterministic("h", h)

    # Derive areas from heights
    A = h * tau
    numpyro.deterministic("A", A)

    # REMOVED numpyro.factor - use proper prior instead
    # Total area prior (weakly informative)
    total_A = A.sum()
    numpyro.deterministic("total_A", total_A)

    # Optional: add as observable (not factor) for diagnostics
    # This doesn't constrain, just records
    numpyro.deterministic(
        "log_total_A_deviation", (jnp.log(total_A) - jnp.log(auc_hat))
    )

    # ----------------------------------------------------------------
    # Mean signal
    # ----------------------------------------------------------------
    mu_y = emg_mixture_area(x, A, mu, sigma, tau)

    # ----------------------------------------------------------------
    # Optional baseline
    # ----------------------------------------------------------------
    if use_baseline:
        b0 = numpyro.sample("b0", dist.Normal(0.0, 0.1 * y_scale))
        b1 = numpyro.sample("b1", dist.Normal(0.0, 0.05 * y_scale / x_span))
        mu_y = mu_y + b0 + b1 * (x - x_mean)
    else:
        numpyro.sample("b0", dist.Normal(0.0, 0.1 * y_scale))
        numpyro.sample("b1", dist.Normal(0.0, 0.05 * y_scale / x_span))

    numpyro.deterministic("mu_y", mu_y)

    # ----------------------------------------------------------------
    # Noise model: Tighter prior on noise scale
    # ----------------------------------------------------------------
    # Estimate noise from high-frequency variation
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(0.05 * y_scale))

    # ----------------------------------------------------------------
    # Likelihood
    # ----------------------------------------------------------------
    numpyro.sample("y", dist.Normal(mu_y, sigma_y).to_event(1), obs=y)


###  gooooooo


def relabel_by_sort(idata: az.InferenceData, key: str = "mu") -> az.InferenceData:
    """
    Sort each draw by ascending μ and permute all component-indexed variables.
    Works for arrays of shape (chain, draw, K) or (chain, draw, K, …).
    """
    post = idata.posterior
    mu = post[key].values  # (chain, draw, K)
    order = jnp.argsort(mu, axis=-1)  # same shape

    comp_vars = [
        v
        for v in post.data_vars
        if post[v].shape[-1] == mu.shape[-1] and post[v].ndim >= 3
    ]

    reordered = {}
    for v in comp_vars:
        arr = post[v].values
        # broadcast order to arr's shape: add as many trailing None as needed
        idx = order[(...,) + (None,) * (arr.ndim - order.ndim)]
        new = jnp.take_along_axis(arr, idx, axis=-1)
        reordered[v] = (post[v].dims, new)

    post_rl = post.assign(**reordered)

    other_groups = {g: getattr(idata, g) for g in idata.groups() if g != "posterior"}
    return az.InferenceData(posterior=post_rl, **other_groups)


# =====================================================================
# Prediction helper
# =====================================================================


def _predict_single_sample(
    x: jnp.ndarray,
    A_s: jnp.ndarray,
    mu_s: jnp.ndarray,
    sigma_s: jnp.ndarray,
    tau_s: jnp.ndarray,
) -> jnp.ndarray:
    """Helper to predict mean from single posterior sample."""
    return emg_mixture_area(x, A_s, mu_s, sigma_s, tau_s)


def predict_mean(
    x: jnp.ndarray,
    samples: dict[str, Any],
    use_baseline: bool = False,
) -> jnp.ndarray:
    """Reconstruct mean signal μ_y from posterior samples.

    Args:
        x: Time/retention points, shape [N]
        samples: Posterior samples dict
        use_baseline: If True, include baseline terms

    Returns:
        Mean predictions, shape [num_samples, N]
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    A = jnp.asarray(samples["A"], dtype=jnp.float32)
    mu = jnp.asarray(samples["mu"], dtype=jnp.float32)
    sigma = jnp.asarray(samples["sigma"], dtype=jnp.float32)
    tau = jnp.asarray(samples["tau"], dtype=jnp.float32)

    predict_fn = partial(_predict_single_sample, x)
    mu_y = jax.vmap(predict_fn)(A, mu, sigma, tau)

    if use_baseline:
        b0 = jnp.asarray(samples["b0"], dtype=jnp.float32)
        b1 = jnp.asarray(samples["b1"], dtype=jnp.float32)
        x_mean = jnp.mean(x)
        baseline = b0[:, None] + b1[:, None] * (x[None, :] - x_mean)
        mu_y = mu_y + baseline

    return mu_y


# =====================================================================
# Helper: Peak height anchors
# =====================================================================


def peak_height_anchors(
    x: jnp.ndarray,
    y: jnp.ndarray,
    mu_lo: jnp.ndarray,
    mu_hi: jnp.ndarray,
    eps: float = 1e-3,
) -> jnp.ndarray:
    """Compute observed peak maximum in each μ-window as height prior anchor.

    For each component k, finds the maximum observed signal in [mu_lo[k], mu_hi[k]].
    This provides a weakly-informative prior center that helps sampling when peaks overlap.

    Args:
        x: Retention time points, shape [N]
        y: Observed signal, shape [N]
        mu_lo: Lower bounds for peak centers, shape [K]
        mu_hi: Upper bounds for peak centers, shape [K]
        eps: Small offset to avoid log(0), default 1e-3

    Returns:
        Peak height anchors, shape [K]
    """
    anchors = []
    for lo, hi in zip(mu_lo, mu_hi):
        mask = (x >= lo) & (x <= hi)
        max_val = float(y[mask].max()) if mask.any() else eps
        anchors.append(max_val + eps)  # +eps to avoid log(0)
    return jnp.array(anchors, dtype=jnp.float32)


# =====================================================================
# ChromFitter class
# =====================================================================


class ChromFitter:
    """Bayesian EMG-mixture fitter for chromatographic peaks.

    Attributes:
        x: Retention time/data points
        y: Observed signal
        mcmc: NumPyro MCMC object (after fit)
        idata: ArviZ InferenceData object (after fit)
        samples: Posterior samples dictionary (after fit)
    """

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mu_lo: jnp.ndarray,
        mu_hi: jnp.ndarray,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        r_min: float = 0.05,
        r_max: Optional[jnp.ndarray] = None,
        auc_hat: Optional[float] = None,
        alpha_dirichlet: float = 0.3,
        use_baseline: bool = False,
    ):
        """Initialize ChromFitter with data and constraints.

        Args:
            x: Retention time/data points, shape [N]
            y: Observed signal, shape [N]
            mu_lo: Lower bounds for peak centers, shape [K]
            mu_hi: Upper bounds for peak centers, shape [K]
            sigma_min: Minimum Gaussian width (auto-computed if None)
            sigma_max: Maximum Gaussian width (auto-computed if None)
            r_min: Minimum tail ratio τ/σ
            r_max: Maximum tail ratio per component, shape [K] (auto-computed if None)
            auc_hat: AUC prior anchor (auto-computed if None)
            alpha_dirichlet: Dirichlet concentration for sparsity
            use_baseline: Whether to include linear baseline
        """
        self.x = jnp.asarray(x, dtype=jnp.float32)
        self.y = jnp.asarray(y, dtype=jnp.float32)
        self.mu_lo = jnp.asarray(mu_lo, dtype=jnp.float32)
        self.mu_hi = jnp.asarray(mu_hi, dtype=jnp.float32)
        self.K = len(mu_lo)
        self.alpha_dirichlet = alpha_dirichlet
        self.use_baseline = use_baseline

        # Auto-compute constraints if not provided
        dx = float(jnp.median(jnp.diff(x)))
        self.sigma_min = sigma_min if sigma_min is not None else 12 * dx / 2.355
        self.sigma_max = (
            sigma_max if sigma_max is not None else (x.max() - x.min()) / 8.0
        )
        self.r_min = r_min

        if r_max is None:
            sigma_ref = 0.5 * (self.sigma_min + self.sigma_max)
            eps = 0.01
            r_hard = 5
            d_right = float(x.max() - float(mu_lo.min()))
            r_edge = d_right / (sigma_ref * jnp.log(1.0 / eps))
            r_max_val = jnp.maximum(r_edge, r_hard)
            self.r_max = jnp.full(self.K, r_max_val, dtype=jnp.float32)
        else:
            self.r_max = jnp.asarray(r_max, dtype=jnp.float32)

        self.auc_hat = (
            auc_hat
            if auc_hat is not None
            else float(jnp.trapezoid(jnp.clip(y, a_min=0.0, a_max=None), x))
        )

        # Compute peak height anchors from observed maxima in each window
        self.h_anchor = peak_height_anchors(self.x, self.y, self.mu_lo, self.mu_hi)

        # Placeholders for results
        self.mcmc: Optional[MCMC] = None
        self.idata: Optional[az.InferenceData] = None
        self.samples: Optional[dict[str, Any]] = None

        # Display initialization summary with rich styling
        self._print_initialization_summary()

    def _print_initialization_summary(self) -> None:
        """Print a rich-styled summary of initialization parameters."""
        # Create main panel
        title = Text("🧬 ChromFitter Initialization Summary", style="bold cyan")

        # Data summary table
        data_table = Table(
            title="📊 Data Summary", show_header=True, header_style="bold magenta"
        )
        data_table.add_column("Parameter", style="cyan", no_wrap=True)
        data_table.add_column("Value", style="green")
        data_table.add_column("Description", style="dim")

        data_table.add_row(
            "Data points (N)", f"{len(self.x):,}", "Number of time points"
        )
        data_table.add_row("Components (K)", f"{self.K}", "Number of EMG components")
        data_table.add_row(
            "Time range",
            f"{float(self.x.min()):.3f} - {float(self.x.max()):.3f}",
            "Retention time span",
        )
        data_table.add_row(
            "Signal range",
            f"{float(self.y.min()):.3f} - {float(self.y.max()):.3f}",
            "Observed signal span",
        )
        data_table.add_row(
            "Sampling Δx",
            f"{float(jnp.median(jnp.diff(self.x))):.6f}",
            "Median time step",
        )

        # Constraints table
        constraints_table = Table(
            title="⚙️ Model Constraints", show_header=True, header_style="bold yellow"
        )
        constraints_table.add_column("Parameter", style="cyan", no_wrap=True)
        constraints_table.add_column("Value", style="green")
        constraints_table.add_column("Description", style="dim")

        constraints_table.add_row(
            "σ_min", f"{self.sigma_min:.6f}", "Minimum Gaussian width"
        )
        constraints_table.add_row(
            "σ_max", f"{self.sigma_max:.6f}", "Maximum Gaussian width"
        )
        constraints_table.add_row(
            "r_min", f"{self.r_min:.3f}", "Minimum tail ratio τ/σ"
        )
        constraints_table.add_row(
            "r_max", f"{float(self.r_max[0]):.3f}", "Maximum tail ratio (per component)"
        )
        constraints_table.add_row(
            "AUC_hat", f"{self.auc_hat:.3f}", "Prior anchor for total area"
        )

        # Peak bounds table
        peak_table = Table(
            title="🎯 Peak Center Bounds & Height Anchors",
            show_header=True,
            header_style="bold blue",
        )
        peak_table.add_column("Component", style="cyan", justify="center")
        peak_table.add_column("μ_lo", style="green", justify="right")
        peak_table.add_column("μ_hi", style="green", justify="right")
        peak_table.add_column("Range", style="yellow", justify="right")
        peak_table.add_column("h_anchor", style="magenta", justify="right")

        for k in range(self.K):
            mu_lo_val = float(self.mu_lo[k])
            mu_hi_val = float(self.mu_hi[k])
            range_val = mu_hi_val - mu_lo_val
            h_anchor_val = float(self.h_anchor[k])
            peak_table.add_row(
                f"k={k}",
                f"{mu_lo_val:.3f}",
                f"{mu_hi_val:.3f}",
                f"{range_val:.3f}",
                f"{h_anchor_val:.3f}",
            )

        # Model settings table
        settings_table = Table(
            title="🔧 Model Settings", show_header=True, header_style="bold red"
        )
        settings_table.add_column("Setting", style="cyan", no_wrap=True)
        settings_table.add_column("Value", style="green")
        settings_table.add_column("Description", style="dim")

        settings_table.add_row(
            "α_dirichlet",
            f"{self.alpha_dirichlet:.1f}",
            "Dirichlet concentration (sparsity)",
        )
        settings_table.add_row(
            "use_baseline", f"{self.use_baseline}", "Include linear baseline"
        )
        settings_table.add_row("Devices", f"{num_devices}", "Available JAX devices")

        # Print all tables
        console.print()
        console.print(Panel.fit(title, border_style="cyan"))
        console.print()
        console.print(data_table)
        console.print()
        console.print(constraints_table)
        console.print()
        console.print(peak_table)
        console.print()
        console.print(settings_table)
        console.print()

    def fit(
        self,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = num_devices,
        seed: int = 42,
        progress_bar: bool = True,
        dense_mass: bool = True,
    ) -> "ChromFitter":
        """Run MCMC sampling with parallel chains.

        Args:
            num_warmup: Number of warmup iterations
            num_samples: Number of samples per chain
            num_chains: Number of parallel chains
            seed: Random seed
            progress_bar: Whether to show progress bar
            dense_mass: Whether to use dense mass matrix

        Returns:
            self for method chaining
        """
        kernel = NUTS(emg_mixture_model, dense_mass=dense_mass)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            chain_method="parallel" if num_chains > 1 else "sequential",
        )

        rng_key = jax.random.PRNGKey(seed)
        self.mcmc.run(
            rng_key,
            x=self.x,
            y=self.y,
            mu_lo=self.mu_lo,
            mu_hi=self.mu_hi,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            r_min=self.r_min,
            r_max=self.r_max,
            auc_hat=self.auc_hat,
            h_anchor=self.h_anchor,
            alpha_dirichlet=self.alpha_dirichlet,
            use_baseline=self.use_baseline,
        )

        self.samples = self.mcmc.get_samples()
        idata_raw = az.from_numpyro(self.mcmc)  # unchanged draws

        # --- new: relabel so every chain uses the same component order
        self.idata = relabel_by_sort(idata_raw, key="mu")

        self.samples = {
            v: self.idata.posterior[v].values.reshape(
                -1, *self.idata.posterior[v].shape[2:]
            )
            for v in self.idata.posterior.data_vars
        }

        return self

    def predict(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Predict mean signal from posterior samples.

        Args:
            x: Prediction points (uses self.x if None)

        Returns:
            Posterior mean predictions, shape [num_samples, N]
        """
        if self.samples is None:
            raise RuntimeError("Must call fit() before predict()")

        x_pred = self.x if x is None else jnp.asarray(x, dtype=jnp.float32)
        return predict_mean(x_pred, self.samples, use_baseline=self.use_baseline)

    def summary(self, var_names: Optional[list[str]] = None, round_to: int = 3) -> Any:
        """Generate summary statistics for posterior.

        Args:
            var_names: Variables to include (None for all)
            round_to: Decimal places to round to

        Returns:
            Styled ArviZ summary DataFrame
        """
        if self.idata is None:
            raise RuntimeError("Must call fit() before summary()")

        df = az.summary(
            self.idata,
            var_names=[
                v for v in self.idata.posterior.data_vars if not v.startswith("mu_y")
            ],
            round_to=round_to,
        )

        def color_rhat(val: float) -> str:
            color = "red" if val > 1.05 else "green"
            return f"color: {color}"

        styled = (
            df.style.applymap(color_rhat, subset=["r_hat"]).format(
                precision=round_to
            )  # ✅ apply same rounding
        )
        return styled

    def plot_trace(
        self, var_names: Optional[list[str]] = None, compact: bool = True
    ) -> None:
        """Plot trace and posterior distributions."""
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_trace()")
        az.plot_trace(self.idata, var_names=var_names, compact=compact)
        plt.tight_layout()
        plt.show()

    def plot_rank(self, var_names: Optional[list[str]] = None) -> None:
        """Plot rank statistics for convergence diagnostics."""
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_rank()")
        az.plot_rank(self.idata, var_names=var_names)  # type: ignore
        plt.tight_layout()
        plt.show()

    def plot_autocorr(self, var_names: Optional[list[str]] = None) -> None:
        """Plot autocorrelation for selected variables."""
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_autocorr()")
        az.plot_autocorr(self.idata, var_names=var_names)  # type: ignore
        plt.tight_layout()
        plt.show()

    def plot_pair(
        self,
        var_names: Optional[list[str]] = None,
        kind: str = "kde",
        save_path: Optional[str] = None,  # ⇦ new
        dpi: int = 100,
    ) -> None:
        """
        Corner plot of the requested variables.

        Args
        ----
        var_names   : list of variable names to plot (None = all)
        kind        : "scatter", "kde", "hexbin", …
        save_path   : filesystem path — supports .png, .pdf, .svg, etc.
                    • .png/.tiff use `dpi`
                    • .pdf/.svg are vector-based (best for unlimited zoom)
        dpi         : resolution for raster formats
        """
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_pair()")

        # lift ArviZ subplot cap
        with az.rc_context(rc={"plot.max_subplots": None}):
            az.plot_pair(
                self.idata,
                var_names=var_names,
                kind=kind,
                marginals=True,
                divergences=True,
            )

            plt.tight_layout()

            if save_path is not None:
                # vector formats (.pdf, .svg) ignore dpi → infinite zoom
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
                print(f"Pair plot saved to {save_path!s}")

            plt.show()

    def plot_fit(self, show_components: bool = False) -> None:
        """Plot data vs posterior mean fit.

        Args:
            show_components: Whether to show individual EMG components
        """
        if self.samples is None:
            raise RuntimeError("Must call fit() before plot_fit()")

        mu_y_samps = self.predict()
        y_hat = mu_y_samps.mean(axis=0)

        plt.figure(figsize=(10, 4))
        plt.plot(self.x, self.y, ".", ms=3, alpha=0.5, label="data", color="C0")
        plt.plot(self.x, y_hat, "-", lw=2, label="predicted mean", color="C1")

        if show_components:
            # Plot median components
            A_med = jnp.median(self.samples["A"], axis=0)
            mu_med = jnp.median(self.samples["mu"], axis=0)
            sigma_med = jnp.median(self.samples["sigma"], axis=0)
            tau_med = jnp.median(self.samples["tau"], axis=0)

            comps = emg_components_area(self.x, A_med, mu_med, sigma_med, tau_med)
            for k in range(self.K):
                plt.fill_between(self.x, 0, comps[k], alpha=0.4, label=f"peak {k + 1}")

        plt.xlabel("Retention time")
        plt.ylabel("Signal")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_corr(
        self,
        var_names: Optional[list[str]] = None,
        figsize: tuple[int, int] = (12, 10),
        cmap: str = "coolwarm",
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> None:
        """
        Show a parameter–parameter correlation matrix.

        Parameters
        ----------
        var_names : list[str] | None
            Variables to include.  None = all component-indexed variables
            (A, mu, sigma, tau, …).
        figsize   : (w, h)
            Figure size in inches.
        cmap      : str
            Matplotlib colormap.
        vmin/vmax : float
            Color scale limits.
        """
        if self.idata is None:
            raise RuntimeError("Must call fit() before plot_corr()")

        # -------- pick variables --------
        posterior = self.idata.posterior
        if var_names is None:
            var_names = [
                v
                for v in posterior.data_vars
                if posterior[v].ndim >= 3 and posterior[v].shape[-1] == self.K
            ]

        cols, labels = [], []
        for v in var_names:
            arr = posterior[v].values  # (chain, draw, K) or scalar per comp
            flat = arr.reshape(-1, *arr.shape[2:])  # collapse chains & draws
            if flat.ndim == 1:  # scalar per draw
                cols.append(flat)
                labels.append(v)
            else:  # one column per component
                for k in range(flat.shape[-1]):
                    cols.append(flat[..., k])
                    labels.append(f"{v}[{k}]")

        X = jnp.column_stack(cols)  # (samples, features)
        C = jnp.corrcoef(X, rowvar=False)

        # -------- plot --------
        plt.figure(figsize=figsize)
        im = plt.imshow(C, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title("Posterior correlation matrix")
        plt.tight_layout()
        plt.show()

    def get_peak_summaries(self) -> dict[str, dict[str, float]]:
        """Extract summary statistics for each peak.

        Returns:
            Dictionary with peak parameters (areas, centers, widths)
        """
        if self.samples is None:
            raise RuntimeError("Must call fit() before get_peak_summaries()")

        summaries = {}
        for k in range(self.K):
            summaries[f"peak_{k + 1}"] = {
                "area_mean": float(self.samples["A"][:, k].mean()),
                "area_std": float(self.samples["A"][:, k].std()),
                "mu_mean": float(self.samples["mu"][:, k].mean()),
                "mu_std": float(self.samples["mu"][:, k].std()),
                "sigma_mean": float(self.samples["sigma"][:, k].mean()),
                "sigma_std": float(self.samples["sigma"][:, k].std()),
                "tau_mean": float(self.samples["tau"][:, k].mean()),
                "tau_std": float(self.samples["tau"][:, k].std()),
            }

        return summaries

    def plot_prior_draws(
        self,
        num_draws: int = 30,
        seed: int = 0,
        figsize: tuple[int, int] = (10, 4),
        alpha: float = 0.25,
        lw: float = 1.0,
        colors: Optional[list[str]] = None,
    ) -> None:
        """
        Overlay the raw data with *num_draws* prior-predictive curves.

        Each thin line is one component from one prior sample – a fast,
        visual sanity-check that your priors are plausible.

        This method:
        1. Draws samples from the prior (y=None → no conditioning on data)
        2. Stores results in self.idata.prior_predictive (ArviZ compatible)
        3. Plots individual component curves overlaid on data

        Parameters
        ----------
        num_draws : int
            How many prior samples to plot.
        seed      : int
            RNG seed used for the prior draw.
        figsize   : (w, h)
            Figure size in inches.
        alpha     : float
            Opacity of each prior curve (keep low!).
        lw        : float
            Line width of each prior curve.
        colors    : list[str] | None
            Matplotlib colours to cycle through for the K components.
            Default = matplotlib's default colour cycle.

        Example
        -------
        >>> fitter = ChromFitter(x, y, mu_lo, mu_hi)
        >>> fitter.plot_prior_draws(num_draws=50, alpha=0.2)
        >>> # Check if priors are reasonable before running MCMC
        >>> fitter.fit(num_warmup=500, num_samples=500)
        """
        from numpyro.infer import Predictive

        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # ===================================================================
        # 1) PRIOR PREDICTIVE SAMPLING
        # ===================================================================
        # Sample from prior by passing y=None (no conditioning on data)
        rng_key = jax.random.PRNGKey(seed)
        predictive = Predictive(
            emg_mixture_model,
            num_samples=num_draws,
        )

        prior_samples = predictive(
            rng_key,
            x=self.x,
            y=None,  # ← KEY: None means prior predictive (no data conditioning)
            mu_lo=self.mu_lo,
            mu_hi=self.mu_hi,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            r_min=self.r_min,
            r_max=self.r_max,
            auc_hat=self.auc_hat,
            h_anchor=self.h_anchor,
            alpha_dirichlet=self.alpha_dirichlet,
            use_baseline=self.use_baseline,
        )

        # ===================================================================
        # 2) ARVIZ INTEGRATION
        # ===================================================================
        # Convert to ArviZ format with proper structure
        # Predictive returns shape (num_samples, ...), ArviZ expects (chain, draw, ...)
        # We treat all samples as coming from a single chain
        prior_dict = {}
        for k, v in prior_samples.items():
            # Add chain dimension: (num_samples, ...) → (1, num_samples, ...)
            prior_dict[k] = jnp.expand_dims(v, axis=0)

        # Create/update InferenceData with prior_predictive group
        if self.idata is None:
            self.idata = az.from_dict(prior_predictive=prior_dict)
        else:
            # Add prior_predictive group to existing InferenceData
            prior_group = az.from_dict(prior_predictive=prior_dict).prior_predictive
            self.idata.add_groups(prior_predictive=prior_group)

        # ===================================================================
        # 3) PLOTTING
        # ===================================================================
        fig, ax = plt.subplots(figsize=figsize)

        # Plot observed data
        ax.plot(self.x, self.y, ".", ms=2, alpha=0.5, label="data", color="k", zorder=1)

        K = self.K
        # Plot prior predictive component curves
        for d in range(num_draws):
            for k in range(K):
                # Compute individual component curve for this sample
                # prior_samples shapes: A[num_samples, K], mu[num_samples, K], etc.
                A_dk = prior_samples["A"][d, k]
                mu_dk = prior_samples["mu"][d, k]
                sigma_dk = prior_samples["sigma"][d, k]
                tau_dk = prior_samples["tau"][d, k]

                # emg_components_area expects arrays [K], returns [K, N]
                # We pass scalar wrapped in array, get [1, N], then squeeze to [N]
                curve = emg_components_area(
                    self.x,
                    jnp.array([A_dk]),
                    jnp.array([mu_dk]),
                    jnp.array([sigma_dk]),
                    jnp.array([tau_dk]),
                )
                # Squeeze to remove K=1 dimension: [1, N] → [N]
                curve = jnp.squeeze(curve, axis=0)

                ax.plot(
                    self.x,
                    curve,
                    lw=lw,
                    alpha=alpha,
                    color=colors[k % len(colors)],
                    zorder=0,
                )

        ax.set_xlabel("Retention time")
        ax.set_ylabel("Signal")
        ax.set_title(f"Prior predictive check: {num_draws} draws × {K} components")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

        console.print(
            "\n[green]✓[/green] Prior predictive samples saved to "
            "[cyan]self.idata.prior_predictive[/cyan]"
        )
        console.print(
            "  Use [yellow]az.plot_ppc(fitter.idata, group='prior_predictive')[/yellow] "
            "for more diagnostics\n"
        )

    @staticmethod
    def simulate(
        A: list[float],
        mu: list[float],
        sigma: list[float],
        tau: list[float],
        x_min: float,
        x_max: float,
        sampling_rate: float = 1200.0,
        noise_level: float = 0.02,
        seed: int = 0,
        baseline: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate a chromatographic spectrum with multiple EMG peaks.

        Args:
            A: Peak areas, length K
            mu: Peak centers (retention times), length K
            sigma: Gaussian widths, length K
            tau: Exponential tail parameters, length K
            x_min: Minimum retention time
            x_max: Maximum retention time
            sampling_rate: Samples per time unit (default: 1200 Hz = 20 Hz * 60)
            noise_level: Relative noise level (fraction of max signal)
            seed: Random seed for noise generation
            baseline: Tuple (b0, b1) for linear baseline b0 + b1*(x - x̄)

        Returns:
            Tuple of (x, y_clean, y_noisy) arrays

        Example:
            >>> x, y_clean, y_noisy = ChromFitter.simulate(
            ...     A=[100, 150, 80],
            ...     mu=[6.5, 6.7, 6.9],
            ...     sigma=[0.05, 0.07, 0.03],
            ...     tau=[0.04, 0.01, 0.06],
            ...     x_min=5.0,
            ...     x_max=8.0,
            ...     noise_level=0.02
            ... )
        """
        # Convert to arrays
        A = jnp.asarray(A, dtype=jnp.float32)
        mu = jnp.asarray(mu, dtype=jnp.float32)
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        tau = jnp.asarray(tau, dtype=jnp.float32)

        # Validate inputs
        K = len(A)
        if not (len(mu) == len(sigma) == len(tau) == K):
            raise ValueError("All peak parameter lists must have the same length")

        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")

        # Generate time points
        N = int((x_max - x_min) * sampling_rate) + 1
        x = jnp.linspace(x_min, x_max, N, dtype=jnp.float32)

        # Generate clean signal
        y_clean = emg_mixture_area(x, A, mu, sigma, tau)

        # Add baseline if specified
        b0, b1 = baseline
        if b0 != 0.0 or b1 != 0.0:
            x_mean = jnp.mean(x)
            y_clean = y_clean + b0 + b1 * (x - x_mean)

        # Add noise
        rng = jax.random.PRNGKey(seed)
        rel_noise = 0.05  # 5 % of the local signal height

        abs_noise = 10.0  # μV (or whatever units) baseline SD

        rng = jax.random.PRNGKey(seed)

        # per-point standard deviation:  σ_i = sqrt((ρ·y_i)² + σ₀²)
        sigma_pts = jnp.sqrt((rel_noise * y_clean) ** 2 + abs_noise**2)

        noise = jax.random.normal(rng, shape=y_clean.shape) * sigma_pts
        y_noisy = y_clean + noise

        return x, y_clean, y_noisy.astype(jnp.float32)


if __name__ == "__main__":
    # Simulate data using the new simulate method
    x, y_clean, y = ChromFitter.simulate(
        A=[14.0, 100, 150],
        mu=[6.5, 6.7, 6.9],
        sigma=[0.05, 0.07, 0.02],
        tau=[0.04, 0.01, 0.06],
        x_min=5.0,
        x_max=8.0,
        sampling_rate=1200.0,  # 20 Hz * 60
        noise_level=0.02,
        seed=0,
    )

    # Plot simulated data
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, ".", ms=2, alpha=0.5, label="noisy data", color="C0")
    plt.plot(x, y_clean, "-", lw=2, label="true signal", color="C1")
    plt.xlabel("Retention time")
    plt.ylabel("Signal")
    plt.legend()
    plt.title("Simulated Chromatographic Data")
    plt.tight_layout()
    plt.show()

    # Define peak search windows
    mu_lo = jnp.array([6.3, 6.6, 6.8], dtype=jnp.float32)
    mu_hi = jnp.array([6.6, 6.9, 7.1], dtype=jnp.float32)

    # Fit model
    fitter = ChromFitter(x, y, mu_lo, mu_hi)
    fitter.fit(num_warmup=1000, num_samples=1000)

    # Diagnostics
    print(fitter.summary())
    fitter.plot_fit(show_components=True)
    fitter.plot_trace(var_names=["A", "mu", "sigma", "tau"])

    # plot corner plot
    fitter.plot_pair(var_names=["A", "mu", "sigma", "tau"])
    plt.tight_layout()
    plt.show()

    # plot rank plot
    fitter.plot_rank(var_names=["A", "mu", "sigma", "tau"])
    plt.tight_layout()
    plt.show()

    # plot autocorrelation plot
    fitter.plot_autocorr(var_names=["A", "mu", "sigma", "tau"])
    plt.tight_layout()
    plt.show()
