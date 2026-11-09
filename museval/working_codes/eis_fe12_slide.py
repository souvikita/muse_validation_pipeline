"""One-raster EIS Fe XII 195.119 slide figure, the EIS counterpart of iris_si4_slide.py: intensity ECDF and Doppler
velocity histogram (observed vs synthetic) on the left, observed intensity and velocity maps on the right
(stretched to fill the panel, as in iris_si4_slide.py). Uses the same fits, recalibration, masks and
synthetic as eis_fe12_summary.py, so the medians and w68 match the summary figure: velocities have each raster
column's median removed; no noise is added to the synthetic (as for IRIS).

Usage: python -I eis_fe12_slide.py [dataset date, default 2019-08-17T04:33:00] [raster index, default 0]
"""
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script's folder off sys.path
import eis_fe12_summary as S  # noqa: E402

OUT_DIR = "/Users/souvikb/MUSE_outputs/figures_paper/QS_valid/eis_figs/"
OBS, SYN, SURF, INK, INK2 = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e"   # as iris_si4_slide.py


def main(date, iraster):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    import astropy.units as u
    import sunpy.map
    import sunpy.visualization.colormaps  # noqa: F401  (registers sdoaia193)

    dset = os.path.join(S.DATA_DIR, f"SDO_EIS_cutouts_{date}")
    f = sorted(glob.glob(os.path.join(dset, "eis_*.data.h5")))[iraster]
    cube, fit = S.fit_raster(f, os.path.join(dset, "fe12_fits"))
    obs = S.observed(cube, fit)
    pt = cube.meta["pointing"]
    syn = S.synthetic(float(pt["x_scale"]), float(pt["y_scale"]), float(np.nanmean(cube.meta["slit_width"])), {})

    mI_o, mI_s = np.nanmedian(obs["I"]), np.nanmedian(syn["I"])
    w_o, w_s = S.w68(obs["v"]), S.w68(syn["v"])

    fig = plt.figure(figsize=(10.5, 10.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.28, wspace=0.25)

    # --- intensity ECDF ---
    ax = fig.add_subplot(gs[0, 0])
    lo_b, hi_b = 10, 1000
    bins = np.logspace(np.log10(lo_b), np.log10(hi_b), 50)
    for data, col, ls, lab, med in ((obs["I"], OBS, "-", "EIS observed", mI_o), (syn["I"], SYN, "--", "Synthetic", mI_s)):
        d = np.clip(data[np.isfinite(data)], lo_b * 1.0001, hi_b * 0.9999)
        ax.hist(d, bins=bins, weights=np.full(d.size, 1 / d.size), histtype="step", lw=2.2, ls=ls, color=col,
                cumulative=True, label=f"{lab} (med: {med:.0f})")
        ax.axvline(med, lw=0.8, ls="-.", color=col)
    for p in ax.patches:                                          # drop the closing vertical edge of the step
        p.set_xy(p.get_xy()[:-1])
    ax.axhline(0.25, lw=1, ls="--", color=INK); ax.axhline(0.75, lw=1, ls="--", color=INK)
    ax.set_xscale("log"); ax.set_xlim(lo_b, hi_b); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 100, 1000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Fe XII 195.12 intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.set_title("Intensity", fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.tick_params(labelsize=11)

    # --- velocity histogram ---
    ax = fig.add_subplot(gs[1, 0])
    vb = np.linspace(-35, 35, 47)
    for data, col, ls, lab, w in ((obs["v"], OBS, "-", "EIS observed", w_o),
                                  (syn["v"], SYN, "--", "Synthetic", w_s)):
        ax.hist(data, bins=vb, weights=np.full(data.size, 1 / data.size), histtype="step", lw=2.2, ls=ls, color=col,
                label=f"{lab} ($w_{{68}}$ {w:.1f})")
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Velocity minus column median [km s$^{-1}$]", fontsize=12)
    ax.set_ylabel("Fraction of pixels", fontsize=12)
    ax.set_title("Doppler velocity", fontsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)                       # headroom for the legend
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.tick_params(labelsize=11)

    # --- observed maps (stretched to the panel as in iris_si4_slide.py; EIS rasters are ~2x taller than wide) ---
    meta = fit.get_map(component=0, measurement="intensity").meta  # eispac's WCS for the (y, x) fit arrays
    good = obs["good"]
    vmin, vmax = np.percentile(obs["I_map"][good], [1, 99.5])
    specs = [("Fe XII 195.12 [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", np.where(good, np.clip(obs["I_map"], vmin, None), np.nan),
              "sdoaia193", LogNorm(vmin, vmax), "0.6"),
             ("Doppler velocity [km s$^{-1}$]\n(masked pixels in black)", np.where(good, obs["v_map"], np.nan),
              "coolwarm", Normalize(-10, 10), "0.15")]
    for row, (title, arr, cmap, norm, bad_col) in enumerate(specs):
        mp = sunpy.map.GenericMap(arr, meta)
        ax = fig.add_subplot(gs[row, 1], projection=mp)
        im = mp.plot(axes=ax, cmap=matplotlib.colormaps[cmap].with_extremes(bad=bad_col), norm=norm,
                     aspect="auto", title=False)
        ax.set_title(title, fontsize=11)
        ax.coords[0].set_ticks(spacing=50 * u.arcsec)
        ax.coords[0].set_axislabel("Solar X [arcsec]" if row == 1 else " ")
        ax.coords[1].set_axislabel("Solar Y [arcsec]")
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
        if row == 0:
            cb.set_ticks([50, 100, 200], labels=["50", "100", "200"])
            cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.suptitle(f"EIS Fe XII 195.12 vs Bifrost QS | {S.raster_time(f).isot[:19]}", fontsize=14, fontweight="bold",
                 y=0.94)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"eis_fe12_195_slide_{date[:10]}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURF)
    print(f"Saved {out}.png/.pdf  med I obs {mI_o:.1f} syn {mI_s:.1f}  w68 obs {w_o:.2f} syn {w_s:.2f}  "
          f"masked {1 - good.mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2019-08-17T04:33:00", int(sys.argv[2]) if len(sys.argv) > 2 else 0)
