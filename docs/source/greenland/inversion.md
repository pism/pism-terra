---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: force
---

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
```

# Greenland Inverse Modeling

Inverting for the basal yield stress `tauc` means trading two things against each
other. Fit the observed velocities too closely and the inversion absorbs
observational noise into the bed, giving a `tauc` field full of structure that
no glaciological process put there. Regularize too hard and real features —
ice streams, sticky spots — get smoothed away. PISM's Tikhonov inversion
controls that trade-off with a single number, `inverse.tikhonov.penalty_weight`
(η), and there is no way to pick it *a priori*. You sweep it and look at the
result.

## Running the sweep

`pism_terra/uq/gris_inverse_penalty.toml` walks η over nine decades:

```toml
samples = 9

['inverse.tikhonov.penalty_weight']

choices = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
distribution = "choices"
```

Adjust `--ntasks` / `--tasks` and the template to match your system.

```bash
pism-ismip7-greenland-run-inverse --resolution 1500m \
    --data-path ismip7_input --output-path greenland_lcurve \
    pism_terra/config/ismip7_greenland_2007_inv_calib.toml \
    pism_terra/templates/debug.j2 \
    pism_terra/uq/gris_inverse_penalty.toml
```

That writes one `output/inverse/inv_*.nc` per member.

## Reducing the output

Each member is about 2 GB, nearly all of it 3-D fields the analysis never
touches. Two quantities are needed per member:

- the **model norm** $N=\sqrt{J_\mathrm{design}}$, how far the solution has been
  pushed from the prior — read from `J_design` at the last iteration;
- the **data misfit** $M$, the area-weighted RMS of `inv_residual` over the
  misfit domain, in m/yr. `inv_J_misfit` measures the same thing but is
  dimensionless, so the RMS is what to quote.

{download}`docs/make_data/lcurve_fixture.py <../../make_data/lcurve_fixture.py>`
collapses a sweep to those numbers plus the per-iteration traces:

```bash
python docs/make_data/lcurve_fixture.py \
    greenland_lcurve/output/inverse \
    docs/source/_data/gris_lcurve_penalty.nc
```

Nine 2 GB files become a 98 KB fixture — which is what this page reads, so the
figures below are real output rather than a sketch.

```{code-cell} ipython3
import numpy as np
import xarray as xr

# myst-nb's cwd is this page's directory, so ``_data/`` is one level up.
sweep = xr.open_dataset("../_data/gris_lcurve_penalty.nc")

eta = sweep["penalty_weight"].values
N = np.sqrt(sweep["J_design"].isel(inv_iter=-1).values)   # model norm
M = sweep["misfit_rms"].values                            # m/yr

for e, n, m in zip(eta, N, M):
    print(f"eta = {e:8.4g}    N = {n:6.3f}    M = {m:6.2f} m/yr")
```

## Finding the corner

Plotted against each other on log axes the pairs trace an L. The vertical arm is
over-regularization: η too small, the fit is bad and loosening it buys a large
drop in misfit for almost no extra model norm. The horizontal arm is
over-fitting: the misfit has stopped improving and further loosening only adds
structure to `tauc`. The corner is the compromise.

"Corner" can be made precise as the point of maximum curvature. Menger curvature
— the reciprocal radius of the circle through three consecutive points — is
enough here and needs no derivatives:

```{code-cell} ipython3
def corner(n, m):
    """Index of maximum Menger curvature on a (log N, log M) L-curve."""
    x, y = np.log10(n), np.log10(m)
    best, k_best = None, -np.inf
    for i in range(1, len(x) - 1):
        a = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        b = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        c = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        area = abs((x[i] - x[i - 1]) * (y[i + 1] - y[i - 1])
                   - (x[i + 1] - x[i - 1]) * (y[i] - y[i - 1]))
        k = 2 * area / (a * b * c) if a * b * c > 0 else 0.0
        if k > k_best:
            k_best, best = k, i
    return best


i = corner(N, M)
print(f"corner at eta = {eta[i]:g}:  N = {N[i]:.3f},  M = {M[i]:.2f} m/yr")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3.6, 2.8), layout="constrained")

ax.plot(N, M, "o-", ms=3, lw=0.9, color="0.3")
ax.plot(N[i], M[i], "*", ms=12, color="C3", zorder=5,
        label=rf"corner, $\eta={eta[i]:g}$")

# Points crowd together on the flat arm, so stagger the labels.
for k, (e, n, m) in enumerate(zip(eta, N, M)):
    ax.annotate(f"{e:g}", (n, m), fontsize=5, xytext=(3, 3 if k % 2 == 0 else -9),
                textcoords="offset points", color="0.4")

ax.set_xscale("log")
ax.set_yscale("log")
# Headroom so the staggered labels stay inside the axes.
ax.set_xlim(N.min() * 0.97, N.max() * 1.14)
ax.set_ylim(M.min() * 0.86, M.max() * 1.20)
ax.set_xlabel(r"model norm  $N=\sqrt{J_\mathrm{design}}$")
ax.set_ylabel(r"data misfit  $M$  (m yr$^{-1}$)")
ax.set_title("Tikhonov L-curve, Greenland 1500 m")
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=False, fontsize=6)
```

The corner sits at **η = 100**. Reading off the numbers makes the trade-off
concrete: relative to the weakest regularization in the sweep (η = 10⁷), the
corner accepts 15 % more misfit in exchange for a 57 % smaller model norm. Past
the corner the curve is flat — going from η = 10⁵ to 10⁷ improves the misfit by
0.02 m/yr while the model norm grows by a fifth. That is noise being fitted.

```{admonition} The corner is a starting point, not an answer
:class: note

Maximum curvature is a heuristic. It is sensitive to how the sweep is sampled —
a decade-spaced sweep can only locate the corner to within a decade — and to
the endpoints, since the first and last members have no curvature defined. Treat
it as identifying the right decade, then refine if the application warrants it,
and always look at the resulting `tauc` field.
```

## Checking the members converged

An L-curve built from unconverged inversions is meaningless: a member that
stopped early sits above and to the left of where it belongs, which can invent a
corner or move one. The fixture keeps the per-iteration traces so this is
checkable.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharex=True, layout="constrained")

colors = plt.cm.viridis(np.linspace(0, 0.9, len(eta)))
for k, (e, c) in enumerate(zip(eta, colors)):
    axs[0].plot(sweep["inv_iter"], sweep["inv_J_misfit"].isel(member=k),
                lw=0.8, color=c, label=f"{e:g}")
    axs[1].plot(sweep["inv_iter"], sweep["grad_sum"].isel(member=k), lw=0.8, color=c)

axs[0].set_ylabel(r"$J_\mathrm{misfit}$")
axs[1].set_ylabel(r"$\|\nabla J\|$")
for ax in axs:
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.grid(True, which="both", alpha=0.3)
axs[0].legend(frameon=False, fontsize=5, ncols=2, title=r"$\eta$", title_fontsize=5)
```

Both panels should flatten. `J_misfit` flattening says the fit has stopped
improving; $\|\nabla J\|$ flattening says the optimizer has reached a stationary
point and further iterations will not move it.

```{code-cell} ipython3
last = sweep.isel(inv_iter=-1)
print(f"{'eta':>9} {'iterations':>11} {'J_misfit change':>16} {'grad_state':>12} {'grad_design':>12}")
for k, e in enumerate(eta):
    J = sweep["inv_J_misfit"].isel(member=k).values
    tail = 100 * abs(J[-1] - J[-25]) / J[-1]        # over the final 25 iterations
    print(f"{e:9.4g} {len(J) - 1:11d} {tail:15.3f}% "
          f"{float(last['grad_state'].isel(member=k)):12.3g} "
          f"{float(last['grad_design'].isel(member=k)):12.3g}")
```

`grad_state` and `grad_design` are the two halves of the Tikhonov gradient. At a
true optimum they cancel, so seeing them converge towards each other is the
strongest signal that a member is genuinely done — and seeing them stay orders of
magnitude apart means it is not, whatever the misfit is doing.

```{admonition} These members hit the iteration cap
:class: warning

Every member here ran the full 250 iterations rather than stopping on a
convergence test. The sweep predates the current
`inverse.tikhonov.rtol = 0.15` default; with PISM's stock `rtol` the criterion
was unreachable for this problem, so the runs continued until the cap. The
misfit is flat well before then, so the L-curve is unaffected — but on a fresh
sweep expect members to stop themselves, and check the iteration counts printed
above rather than assuming.
```

## Using the result

Set the chosen weight in the campaign config and run the inversion for real:

```toml
[inverse]

'inverse.tikhonov.penalty_weight' = 100
```

The inverted `tauc` is then regridded into the forward run, held fixed by
`basal_yield_stress.model = "constant"` — `pism-ismip7-greenland-run-inverse`
wires that up for you. See {doc}`../features/run_configuration` for how the
init, inversion and forward legs chain together.
