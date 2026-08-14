#!/usr/bin/env python
"""
Smoke test for a TIPSY install: runs the real code paths on a small synthetic
streamer, so dependency incompatibilities show up in seconds instead of
halfway through a tutorial notebook.

Usage (from anywhere, with the tipsy environment active):
    python check_env.py
"""

import sys
import traceback

import matplotlib
matplotlib.use("Agg")  # no display needed; plt.show() becomes a no-op

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

import tipsy
import trivia_my  # noqa: F401  (import-only check)

DIST = 150 * u.pc
MS = 1.0
SVEL = 5.0  # km/s


def versions():
    import astropy, matplotlib as mpl, pandas, plotly, scipy, sklearn
    import spectral_cube, uncertainties
    mods = [np, scipy, pandas, mpl, astropy, spectral_cube, sklearn, plotly,
            uncertainties]
    for m in mods:
        print(f"  {m.__name__:16s} {getattr(m, '__version__', '?')}")
    print(f"  {'python':16s} {sys.version.split()[0]}")


def synthetic_cube():
    """Paint a theoretical infalling trajectory into a small PPV cube."""
    rt, vt, _, _, _ = tipsy.falling_trajectory(
        300.0, 300.0, 100.0, -1.0, -1.0, -0.5, MS, verbose=False,
        ang_range=0.6 * np.pi, ang_step=np.pi / 400)

    x_as = tipsy.au2arcsec(rt[0], DIST)
    y_as = tipsy.au2arcsec(rt[1], DIST)
    v_ms = (vt[2] + SVEL) * 1e3

    npix, nchan = 96, 40
    cdelt_as, cdelt_v = 0.08, 150.0
    w = WCS(naxis=3)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "VELO-LSR"]
    w.wcs.crpix = [npix // 2 + 1, npix // 2 + 1, nchan // 2 + 1]
    w.wcs.crval = [285.0, -36.0, SVEL * 1e3]
    w.wcs.cdelt = [-cdelt_as / 3600, cdelt_as / 3600, cdelt_v]
    w.wcs.cunit = ["deg", "deg", "m/s"]

    data = np.zeros((nchan, npix, npix), dtype="float32")
    cx = cy = npix // 2
    for xa, ya, vv in zip(x_as, y_as, v_ms):
        i = int(round(cx - xa / cdelt_as))          # R.A. axis runs negative
        j = int(round(cy + ya / cdelt_as))
        k = int(round(nchan // 2 + (vv - SVEL * 1e3) / cdelt_v))
        if 1 <= i < npix - 1 and 1 <= j < npix - 1 and 1 <= k < nchan - 1:
            data[k - 1:k + 2, j - 1:j + 2, i - 1:i + 2] += 1.0
    if data.max() == 0:
        raise RuntimeError("synthetic streamer fell outside the cube")
    data += np.random.default_rng(0).normal(0, 0.01, data.shape).astype("float32")

    hdu = fits.PrimaryHDU(data, header=w.to_header())
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.header["BMAJ"] = 3 * cdelt_as / 3600
    cube = SpectralCube.read(hdu)
    return cube.with_mask(cube > 0.5 * cube.unit)


def main():
    print("Versions:")
    versions()

    checks = []

    def run(name, fn):
        try:
            out = fn()
            checks.append((name, "PASS", out))
        except Exception:
            checks.append((name, "FAIL", traceback.format_exc()))

    # 1. Pure-python trajectory maths: astropy units, numpy, cumulative times.
    def t_traj():
        rt, vt, sr, sv, times = tipsy.falling_trajectory(
            300.0, 300.0, 100.0, -1.0, -1.0, -0.5, MS, verbose=True)
        assert np.isfinite(rt).all() and np.isfinite(vt).all()
        return f"{rt.shape[1]} points, t_max={np.nanmax(times):.0f} yr"
    run("falling_trajectory", t_traj)

    # 2. Cube handling: spectral-cube WCS, masking, slicing.
    cube = None
    def t_cube():
        nonlocal cube
        cube = synthetic_cube()
        return f"cube {cube.shape}, {int(np.isfinite(np.array(cube)).sum())} unmasked voxels"
    run("synthetic SpectralCube", t_cube)

    # 3. The full fit: sklearn ParameterGrid, scipy interp1d, pandas pivot /
    #    groupby / slice-assignment, matplotlib imshow. show_fit_cost=True is
    #    deliberate: that branch holds the pandas code most likely to break.
    params = None
    def t_fit():
        nonlocal params
        params, traj_m, comps = tipsy.traj_fitting(
            cube, MS, DIST, SVEL, N_elements=6,
            vxy_ang0_span=0.2, z0_lim=(-400, 400), z0_step=200,
            vxy0_lim=(0.5, 2.5), vxy0_step=0.5,
            show_dist_plots=False, show_vel_ang=False,
            show_fit_cost=True, show_fit_3d=False, verbose=True)
        assert len(params) > 0 and params["fit_fraction"].notna().any()
        return f"{len(params)} combinations, best fit_fraction={params.fit_fraction.max():.2f}"
    run("traj_fitting", t_fit)

    # 4. Error estimation: uncertainties (ufloat / umath) plus pandas indexing.
    def t_err():
        vals = tipsy.parameter_errors(params, threshold=-np.inf)
        assert {"vx0", "vy0"}.issubset(vals.index)
        return f"parameters: {list(vals.index)}"
    run("parameter_errors", t_err)

    # 5. Plotly figure construction (built, not shown).
    def t_plot():
        import plotly.graph_objects as go
        try:
            import nbformat
        except ImportError:
            raise ImportError(
                "nbformat is missing: plotly figures build fine here, but "
                "fig.show() will fail inside a notebook (pip install nbformat)")
        fig = go.Figure(data=[go.Scatter3d(
            x=[1, 2], y=[1, 2], z=[1, 2], mode="lines+markers",
            error_x=go.scatter3d.ErrorX(array=[.1, .1], width=2),
            marker=dict(size=9, color="red", opacity=.8, symbol="square"))])
        fig.update_layout(legend_orientation="h")
        return f"Scatter3d + ErrorX ok, nbformat {nbformat.__version__}"
    run("plotly figure", t_plot)
        
    # 6. rebound, only if the optional extra is installed.
    def t_rebound():
        try:
            import rebound  # noqa: F401
        except ImportError:
            return "not installed (optional extra), skipped"
        xyz, v_xyz, times = tipsy.rebound_trajectory(
            300.0, 300.0, 100.0, -1.0, -1.0, -0.5, MS,
            t_max=1e4, N_out=20, verbose=False)
        assert np.isfinite(xyz).all()
        return f"{xyz.shape[0]} steps"
    run("rebound_trajectory", t_rebound)

    print("\nResults:")
    failed = 0
    for name, status, out in checks:
        print(f"  [{status}] {name}")
        if status == "FAIL":
            failed += 1
            print("\n".join("      " + l for l in out.strip().splitlines()[-6:]))
        else:
            print(f"      {out}")

    print("\nFAILED" if failed else "\nAll checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
