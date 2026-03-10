import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

fps = 25
x = np.linspace(0.0, 10.0, 700)


def ease(a, b, n):
    t = np.linspace(0.0, 1.0, n)
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return a + (b - a) * s


def phi(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def Phi(z):
    # fast smooth approximation of the normal CDF; good enough for visualization
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def skew_normal(x, loc, sigma, alpha, area):
    z = (x - loc) / sigma
    return area * (2.0 / sigma) * phi(z) * Phi(alpha * z)


frames = []


def _expand(v, n):
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.size == 1:
        return np.full(n, float(arr.item()))
    if arr.size != n:
        raise ValueError(f"Expected size {n}, got {arr.size}")
    return arr


def add_segment(stage, n, alpha, sigma, loc, area, band_param=None, band_frac=None):
    a = _expand(alpha, n)
    s = _expand(sigma, n)
    l = _expand(loc, n)
    A = _expand(area, n)
    u = None if band_frac is None else _expand(band_frac, n)

    for i in range(n):
        frames.append(
            {
                "stage": stage,
                "alpha": float(a[i]),
                "sigma": float(s[i]),
                "loc": float(l[i]),
                "area": float(A[i]),
                "band_param": band_param,
                "band_frac": None if u is None else float(u[i]),
            }
        )


# base settings
alpha0 = -3.0
sigma0 = 0.35
loc0 = 5.0
area0 = 40.0

# 1) start with a skew-normal
add_segment("Skew-normal peak", fps, alpha0, sigma0, loc0, area0)

# 2) animate alpha
add_segment(
    "Change skew α: -3 → 3", 3 * fps, ease(-3.0, 3.0, 3 * fps), sigma0, loc0, area0
)

# 3) animate sigma
add_segment(
    "Change width σ: 0.2 → 0.8", 3 * fps, 3.0, ease(0.2, 0.8, 3 * fps), loc0, area0
)

# 4) animate location
add_segment(
    "Change location ξ: 3 → 7", 3 * fps, 3.0, 0.8, ease(3.0, 7.0, 3 * fps), area0
)

# 5) animate area
# I interpret area as true peak area here, so height changes as well.
add_segment(
    "Change area A: 10 → 100", 3 * fps, 3.0, 0.8, 7.0, ease(10.0, 100.0, 3 * fps)
)

# 6) change everything simultaneously
t = np.linspace(0.0, 1.0, 4 * fps)
add_segment(
    "All parameters move together",
    4 * fps,
    3.0 * np.cos(2.0 * np.pi * t),
    0.5 + 0.3 * np.cos(2.0 * np.pi * t),
    5.0 + 2.0 * np.cos(2.0 * np.pi * t),
    55.0 + 45.0 * np.cos(2.0 * np.pi * t),
)

# 7) go back to normal
last = frames[-1]
normal = {"alpha": 0.0, "sigma": 0.5, "loc": 5.0, "area": 40.0}
add_segment(
    "Back to normal",
    3 * fps,
    ease(last["alpha"], normal["alpha"], 3 * fps),
    ease(last["sigma"], normal["sigma"], 3 * fps),
    ease(last["loc"], normal["loc"], 3 * fps),
    ease(last["area"], normal["area"], 3 * fps),
)

# 8) hold for 1 second
add_segment(
    "Normal peak", fps, normal["alpha"], normal["sigma"], normal["loc"], normal["area"]
)

# 9) introduce uncertainty
add_segment(
    "Parameter uncertainty",
    fps,
    normal["alpha"],
    normal["sigma"],
    normal["loc"],
    normal["area"],
)

# 10) uncertainty sweeps, one parameter at a time
u = np.r_[ease(0.05, 0.30, 2 * fps), ease(0.30, 0.05, 2 * fps)]

for param, label in [
    ("alpha", "Uncertainty in α"),
    ("sigma", "Uncertainty in σ"),
    ("loc", "Uncertainty in ξ"),
    ("area", "Uncertainty in A"),
]:
    add_segment(
        label,
        len(u),
        normal["alpha"],
        normal["sigma"],
        normal["loc"],
        normal["area"],
        band_param=param,
        band_frac=u,
    )

fig, ax = plt.subplots(figsize=(9, 5))
(line,) = ax.plot([], [], lw=2)
band_artist = [None]

ax.set_xlim(0.0, 10.0)
ax.set_ylim(0.0, 65.0)
ax.set_xlabel("x")
ax.set_ylabel("intensity")

rng = np.random.default_rng(7)
z_alpha = rng.normal(size=180)
z_sigma = rng.normal(size=180)
z_loc = rng.normal(size=180)
z_area = rng.normal(size=180)


def uncertainty_band(x, alpha, sigma, loc, area, param, frac):
    n = z_alpha.size
    a = np.full(n, alpha)
    s = np.full(n, sigma)
    l = np.full(n, loc)
    A = np.full(n, area)

    if param == "alpha":
        a = alpha + (6.0 * frac) * z_alpha
    elif param == "sigma":
        s = np.clip(sigma * (1.0 + frac * z_sigma), 0.05, None)
    elif param == "loc":
        l = loc + loc * frac * z_loc
    elif param == "area":
        A = np.clip(area * (1.0 + frac * z_area), 1e-3, None)

    y = skew_normal(
        x[None, :],
        l[:, None],
        s[:, None],
        a[:, None],
        A[:, None],
    )

    lo = np.percentile(y, 5.0, axis=0)
    hi = np.percentile(y, 95.0, axis=0)
    mid = np.median(y, axis=0)
    return lo, hi, mid


def update(i):
    state = frames[i]

    if band_artist[0] is not None:
        band_artist[0].remove()
        band_artist[0] = None

    if state["band_param"] is None:
        y = skew_normal(x, state["loc"], state["sigma"], state["alpha"], state["area"])
        line.set_data(x, y)
        title = (
            f"{state['stage']}\n"
            f"α={state['alpha']:.2f}, σ={state['sigma']:.2f}, ξ={state['loc']:.2f}, A={state['area']:.1f}"
        )
    else:
        lo, hi, mid = uncertainty_band(
            x,
            state["alpha"],
            state["sigma"],
            state["loc"],
            state["area"],
            state["band_param"],
            state["band_frac"],
        )
        line.set_data(x, mid)
        band_artist[0] = ax.fill_between(x, lo, hi, alpha=0.5)
        title = (
            f"{state['stage']}: {100 * state['band_frac']:.0f}%\n"
            f"α={state['alpha']:.2f}, σ={state['sigma']:.2f}, ξ={state['loc']:.2f}, A={state['area']:.1f}"
        )

    ax.set_title(title)
    return (line,)


anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=1000 / fps,
    blit=False,
)

anim.save("skewnormal_dofs_and_uncertainty.gif", writer=PillowWriter(fps=fps))
plt.show()
