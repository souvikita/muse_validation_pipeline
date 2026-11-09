"""One-dataset IRIS Si IV 1402.77 slide figure, laid out like the EIS Fe XII one: intensity ECDF and Doppler velocity
histogram (observed vs synthetic) on the left, observed intensity and velocity maps on the right. Uses the same
inputs and definitions as iris_si4_summary.py, so the medians and w68 match the summary figure.

Usage: python -I iris_si4_slide.py [dataset folder name, default 2014-06-07T12:09:00]
"""
import os
import sys
import warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script's folder off sys.path
import iris_si4_summary as S  # noqa: E402

OUT_DIR = "/Users/souvikb/MUSE_outputs/figures_paper/QS_valid/iris_figs/"
OBS, SYN, SURF, INK, INK2 = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e"   # iris_obs_syn_comparison.ipynb


def main(date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from astropy.wcs import WCS
    import sunpy.map
    import sunpy.visualization.colormaps  # noqa: F401  (registers irissji1400)

    dset = os.path.join(S.IRIS_DIR, date)
    cal = S.calibrated_si_iv(dset)
    fr = np.load(S.fit_file(dset))
    l2 = S.level2(dset)
    h0, hs = l2["h0"], l2["hsi"]

    # observed, exactly as in iris_si4_summary.analyse
    wl = S.wave(hs)
    core = np.abs(wl - S.W0) < 0.3
    cont = (np.abs(wl - S.W0) > 0.5) & (np.abs(wl - S.W0) < 1.0)
    n_phot = S.photons(l2["dn_si"], hs, S.W0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.asarray(cal.data, dtype=float)
        c = np.where(np.isfinite(c) & (l2["dn_si"] > -100), c, np.nan)
        I_int = np.nansum(c[..., core] - np.nanmedian(c[..., cont], axis=-1, keepdims=True), axis=-1) * hs["CDELT1"]
    v = fr["core_shift"] - S.o_i_reference(l2)
    fill = np.all(np.isnan(l2["dn_si"]), axis=-1) | (np.ptp(fr["flux"], axis=0) == 0)[None, :]
    bad = fill | ~S.fit_ok(fr["flux"], v, fr["sigma"], n_phot)
    obs_I, obs_v = I_int[~fill], v[~bad]

    # synthetic, degraded to this raster
    syn = np.load(S.SYN_FILE)
    kpa = h0["DSUN_OBS"] * 1e-3 * np.deg2rad(1 / 3600)
    to_sim = kpa / S.DX_SIM_KM
    kw = dict(dx=1.0, psf_fwhm=S.PSF_FWHM_ARCSEC * to_sim, box=(S.SLIT_ARCSEC * to_sim, hs["CDELT2"] * to_sim),
              pixel=(hs["CDELT3"] * to_sim, hs["CDELT2"] * to_sim))
    syn_I2 = S.iris_degrade(syn["itot"], **kw)
    syn_I = syn_I2.ravel()
    syn_v = (S.iris_degrade(syn["itot"] * syn["utot"], **kw) / syn_I2).ravel()

    mI_o, mI_s = np.nanmedian(obs_I), np.nanmedian(syn_I)
    w_o, w_s = S.w68(obs_v), S.w68(syn_v)

    fig = plt.figure(figsize=(10.5, 10.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.28, wspace=0.25)

    # --- intensity ECDF (all pixels; values below the axis range still count) ---
    ax = fig.add_subplot(gs[0, 0])
    lo_b, hi_b = 1, 3000
    bins = np.logspace(np.log10(lo_b), np.log10(hi_b), 50)
    for data, col, ls, lab, med in ((obs_I, OBS, "-", "IRIS observed", mI_o), (syn_I, SYN, "--", "Synthetic", mI_s)):
        d = np.clip(data[np.isfinite(data)], lo_b * 1.0001, hi_b * 0.9999)
        ax.hist(d, bins=bins, weights=np.full(d.size, 1 / d.size), histtype="step", lw=2.2, ls=ls, color=col,
                cumulative=True, label=f"{lab} (med: {med:.0f})")
        ax.axvline(med, lw=0.8, ls="-.", color=col)
    for p in ax.patches:                                          # drop the closing vertical edge of the step
        p.set_xy(p.get_xy()[:-1])
    ax.axhline(0.25, lw=1, ls="--", color=INK); ax.axhline(0.75, lw=1, ls="--", color=INK)
    ax.set_xscale("log"); ax.set_xlim(lo_b, hi_b); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Si IV 1402.77 intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.set_title("Intensity", fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.tick_params(labelsize=11)

    # --- velocity histogram ---
    ax = fig.add_subplot(gs[1, 0])
    vb = np.linspace(-35, 35, 36)
    for data, col, ls, lab, w in ((obs_v, OBS, "-", "IRIS observed", w_o), (syn_v, SYN, "--", "Synthetic", w_s)):
        ax.hist(data, bins=vb, weights=np.full(data.size, 1 / data.size), histtype="step", lw=2.2, ls=ls, color=col,
                label=f"{lab} ($w_{{68}}$ {w:.1f})")
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("[km s$^{-1}$]", fontsize=12)
    ax.set_ylabel("Fraction of pixels", fontsize=12)
    ax.set_title("Doppler velocity", fontsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)                       # headroom for the legend
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.tick_params(labelsize=11)

    # --- observed maps ---
    obs_wcs = WCS(hs).dropaxis(0).swapaxes(0, 1)                  # (wave, slit y, step x) -> map axes (x, y)
    meta = obs_wcs.to_header()
    meta.update(date_obs=h0["DATE_OBS"], dsun_obs=h0["DSUN_OBS"])
    vmin, vmax = np.percentile(I_int[~fill & (I_int > 0)], [1, 99.5])
    # faint pixels (incl. noise <= 0) drawn at the bottom of the scale, not as missing data
    specs = [("Si IV 1402.77 [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", np.where(fill, np.nan, np.clip(I_int, vmin, None)),
              "irissji1400", LogNorm(vmin, vmax), "0.6"),
             ("Doppler velocity [km s$^{-1}$]", np.where(bad, np.nan, v), "coolwarm", Normalize(-20, 20), "0.15")]
    for row, (title, arr, cmap, norm, bad_col) in enumerate(specs):
        mp = sunpy.map.GenericMap(arr.T, meta)                    # sunpy wants (y, x)
        ax = fig.add_subplot(gs[row, 1], projection=mp)
        im = mp.plot(axes=ax, cmap=matplotlib.colormaps[cmap].with_extremes(bad=bad_col), norm=norm,
                     aspect="auto", title=False)
        ax.set_title(title + ("\n(masked pixels in black)" if row == 1 else ""), fontsize=11)
        ax.coords[0].set_axislabel("Solar X [arcsec]" if row == 1 else " ")
        ax.coords[1].set_axislabel("Solar Y [arcsec]")
        fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)

    fig.suptitle(f"IRIS Si IV 1402.77 vs Bifrost QS | {h0['DATE_OBS'][:19]}", fontsize=14, fontweight="bold", y=0.94)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"iris_si4_1402_slide_{date[:10]}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURF)
    print(f"Saved {out}.png/.pdf  med I obs {mI_o:.1f} syn {mI_s:.1f}  w68 obs {w_o:.2f} syn {w_s:.2f}  "
          f"masked {bad[~fill].mean():.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2014-06-07T12:09:00")
