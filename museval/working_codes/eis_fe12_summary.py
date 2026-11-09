"""Summary of observed (EIS, eispac) vs synthetic (eis_synthesis) Fe XII 195.119 for the five EIS QS datasets
of the AIA summary figure (valid_bigger_cuts.ipynb, cells 18-19), in the same style.

Per raster:  med(I_sim)/med(I_obs) and w68(v_sim)/w68(v_obs), w68 = half-width of the central 68% of the velocities
             ((p84 - p16)/2, = sigma for a Gaussian). Velocities have each raster column's median removed (eispac's
             default for EIS, whose absolute zero point drifts across columns; synthetic treated the same). No noise is
             added to the synthetic, as for IRIS (iris_si4_summary.py): a plain observed vs synthetic comparison.
Per dataset: dot = raster closest in time to the AIA image of the AIA summary figure,
             bar = 5-95% of the ratio over all rasters of the sequence (2014-04-28, 2016-03-20, 2016-03-22).

Usage: python -I eis_fe12_summary.py [--plot-only]
       Also writes the EIS + IRIS figure when iris_si4_summary.py has written summary_syn_obs_iris.json.
"""
import os
import sys
import glob
import json
import numpy as np

C_KMS, W0 = 299792.458, 195.119
DATA_DIR = "/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/QS_datasets/"
VDEM_FILE = "/Users/souvikb/MUSE_outputs/vdems/muse_vdem_qs_en024031_los2_JMS_fixedvdop_2May2025_010.zarr"
OUT_DIR = "/Users/souvikb/MUSE_outputs/figures_paper/QS_valid/"
OUT_JSON = os.path.join(OUT_DIR, "summary_syn_obs_eis.json")
EIS_DATES = ["2014-04-28T13:22:00", "2016-03-20T23:45:00", "2016-03-22T00:18:40",
             "2019-08-17T04:33:00", "2019-12-01T16:24:00"]


def raster_time(f):
    """Start time of an EIS raster from its file name, eis_YYYYMMDD_HHMMSS.data.h5."""
    from astropy.time import Time
    s = os.path.basename(f).split(".")[0].split("_")
    return Time(f"{s[1][:4]}-{s[1][4:6]}-{s[1][6:]}T{s[2][:2]}:{s[2][2:4]}:{s[2][4:]}", format="isot", scale="utc")


def aia_time(dset):
    """Time of the AIA 171 image used for this dataset in the AIA summary figure."""
    from astropy.io import fits
    from astropy.time import Time
    f = glob.glob(os.path.join(dset, "bigger_sdo_cutouts", "*.171.image_lev1.fits"))[0]
    return Time(fits.getheader(f, 1 if fits.getheader(f).get("NAXIS", 0) == 0 else 0)["DATE-OBS"], scale="utc")


def fit_raster(data_file, fit_dir):
    """eispac fit of Fe XII 195.119 (2c template), cached in fit_dir."""
    import eispac
    tmpl = eispac.read_template(eispac.data.get_fit_template_filepath("fe_12_195_119.2c.template.h5"))
    stem = os.path.basename(data_file).split(".")[0]
    saved = glob.glob(os.path.join(fit_dir, f"{stem}.fe_12_195_119.2c-0.fit.h5"))
    cube = eispac.read_cube(data_file, tmpl.central_wave)
    if saved:
        fit = eispac.read_fit(saved[0])
    else:
        fit = eispac.fit_spectra(cube, tmpl, ncpu="max")
        eispac.save_fit(fit, save_dir=fit_dir)
    return cube, fit


def observed(cube, fit):
    """Recalibrated Fe XII intensities and column-median-removed velocities of the good pixels."""
    from museval.eis_calibration.eis_calib_2023 import eis_ea, interpol_eis_ea
    date = str(np.atleast_1d(cube.meta["date_obs"])[0])
    recal = float((eis_ea(np.array([W0])) / interpol_eis_ea(date, np.array([W0]), quiet=True))[0])
    i0, e0 = fit.fit["int"][:, :, 0], fit.fit["err_int"][:, :, 0]                   # (y, x)
    good = np.isfinite(i0) & (i0 > 0) & np.isfinite(e0) & (e0 / np.where(i0 > 0, i0, np.nan) < 0.3)
    v = np.where(good, C_KMS * (fit.fit["params"][:, :, 1] - W0) / W0, np.nan)        # centroid, wave_corr applied
    v = v - np.nanmedian(v, axis=0, keepdims=True)                                    # per raster column
    return dict(date=date, recal=recal, n_good=int(good.sum()), n_total=int(good.size), I=i0[good] * recal, v=v[good],
                good=good, I_map=i0 * recal, v_map=v)                                 # (y, x), for the slide maps


def w68(a, axis=None):
    """Half-width of the central 68% of a distribution, (p84 - p16)/2; equals sigma for a Gaussian."""
    p16, p84 = np.nanpercentile(a, [16, 84], axis=axis)
    return 0.5 * (p84 - p16)


def synthetic(step, ypix, instr_fwhm, cache):
    """Synthetic Fe XII at the EIS pixel size (raster step x 1"), cached per geometry."""
    import xarray as xr
    from museval.eis_synthesis import eis_degrade_vdem, eis_line_response, eis_synthesis
    key = (round(step, 2), round(ypix, 2))
    if key not in cache:
        vdem = xr.open_zarr(VDEM_FILE)
        deg = eis_degrade_vdem(vdem, pixel=(step, ypix))
        resp, _ = eis_line_response(deg, instr_fwhm=instr_fwhm)                     # AIA's abundance and pressure
        syn = eis_synthesis(deg, resp).sel(line="Fe XII 195.119")
        I = syn.intensity.transpose("x", "y").values
        v = syn.vdop.transpose("x", "y").values                                      # VDEM: positive = redshift
        v = v - np.nanmedian(v, axis=1, keepdims=True)                                 # per column, as observed
        cache[key] = dict(I=I.ravel(), v=v.ravel(), abund=resp.attrs["abund"])
    return cache[key]


def compute():
    rows, syn_cache = [], {}
    for d in EIS_DATES:
        dset = os.path.join(DATA_DIR, f"SDO_EIS_cutouts_{d}")
        fit_dir = os.path.join(dset, "fe12_fits")
        os.makedirs(fit_dir, exist_ok=True)
        t_aia = aia_time(dset)
        rasters = []
        for f in sorted(glob.glob(os.path.join(dset, "eis_*.data.h5"))):
            cube, fit = fit_raster(f, fit_dir)
            pt = cube.meta["pointing"]
            step, ypix = float(pt["x_scale"]), float(pt["y_scale"])
            obs = observed(cube, fit)
            syn = synthetic(step, ypix, float(np.nanmean(cube.meta["slit_width"])), syn_cache)
            t0 = raster_time(f)
            dur = float(np.nansum(cube.meta["duration"])) if np.size(cube.meta["duration"]) > 1 \
                else float(np.atleast_1d(cube.meta["duration"])[0]) * cube.data.shape[1]
            r = dict(file=os.path.basename(f), t_start=t0.isot, t_mid=(t0 + dur / 2 / 86400).isot,
                     step=step, ypix=ypix, recal=obs["recal"], n_good=obs["n_good"], n_total=obs["n_total"],
                     n_syn=int(syn["I"].size),
                     med_I_obs=float(np.nanmedian(obs["I"])), med_I_syn=float(np.nanmedian(syn["I"])),
                     w68_v_obs=float(w68(obs["v"])), w68_v_syn=float(w68(syn["v"])))
            r["dt_aia_min"] = abs(((t0 + dur / 2 / 86400) - t_aia).to_value("min"))
            rasters.append(r)
            print(f"{d} {r['file']}  I obs/syn {r['med_I_obs']:7.1f}/{r['med_I_syn']:6.1f}  "
                  f"w68_v obs/syn {r['w68_v_obs']:5.2f}/{r['w68_v_syn']:5.2f}  "
                  f"dt_AIA {r['dt_aia_min']:6.1f} min", flush=True)
        iaia = int(np.argmin([r["dt_aia_min"] for r in rasters]))
        rows.append(dict(date=d, aia_time=t_aia.isot, aia_raster=rasters[iaia]["file"], abund=syn["abund"],
                         rasters=rasters, i_aia=iaia))
    json.dump(rows, open(OUT_JSON, "w"), indent=1)
    print("Saved", OUT_JSON)
    return rows


def plot(rows, iris=None):
    """EIS-only summary (summary_syn_obs_eis.*), or with IRIS Si IV 1402.77 columns beside Fe XII when `iris`
    (rows of iris_si4_summary.py's JSON) is given (summary_syn_obs_eis_iris.*)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from astropy.time import Time

    # EIS: one dot per dataset (raster closest to the AIA image) plus the 5-95% spread over its raster sequence
    eis = [dict(date=r["date"], q=[x for x in r["rasters"]], i=r["i_aia"]) for r in rows]
    eis_ratio = {"I": lambda x: x["med_I_syn"] / x["med_I_obs"], "v": lambda x: x["w68_v_syn"] / x["w68_v_obs"]}
    cols_spec = [("I", "EIS\nFe XII 195.12 Å", eis, eis_ratio["I"], "o"), ("I", None, None, None, None),
                 ("v", "EIS\nFe XII 195.12 Å", eis, eis_ratio["v"], "o"), ("v", None, None, None, None)]
    if iris is not None:
        iris_rows = [dict(date=r["date"], q=[r], i=0) for r in iris]
        cols_spec[1] = ("I", "IRIS\nSi IV 1402.77 Å", iris_rows, lambda x: x["med_I_syn"] / x["med_I_obs"], "s")
        cols_spec[3] = ("v", "IRIS\nSi IV 1402.77 Å", iris_rows, lambda x: x["w68_v_syn"] / x["w68_v_obs"], "s")
    cols_spec = [c for c in cols_spec if c[1] is not None]
    n_i = sum(c[0] == "I" for c in cols_spec)
    titles = {"I": ("EIS intensity $I$\n" if iris is None else "Intensity $I$\n")
                   + r"$\mathrm{med}(I_\mathrm{sim})\,/\,\mathrm{med}(I_\mathrm{obs})$",
              "v": ("EIS Doppler velocity $v$\n" if iris is None else "Doppler velocity $v$\n")
                   + r"$w_{68}(v_\mathrm{sim})\,/\,w_{68}(v_\mathrm{obs})$"}
    # Same date groups and colours as the AIA summary figure (Paul Tol's high-contrast palette)
    date_groups = ["2014", "2015–2016", "2019–2020"]
    group_colors = ["#BB5566", "#DDAA33", "#004488"]
    tdec = lambda d: Time(d if len(d) > 16 else d + ":00", format="isot").decimalyear
    color = lambda d: group_colors[0 if tdec(d) < 2015 else (1 if tdec(d) < 2018 else 2)]

    q_all = [f(x) for _, lab, rr, f, _ in cols_spec for r in rr for x in r["q"]]
    ylo, yhi = min(0.2, 0.8 * min(q_all)), max(6.6, 1.25 * max(q_all))

    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, ax = plt.subplots(figsize=(7.1 if iris is not None else 5.0, 3.0), layout="constrained")
        ax.axhspan(0.5, 2, color="0.92", zorder=0)
        ax.axhline(1, color="0.4", lw=0.8, ls="--", zorder=1)
        for x0, (k, lab, rr, f, mk) in enumerate(cols_spec):
            rr = sorted(rr, key=lambda r: tdec(r["date"]))
            offsets = np.linspace(-0.25, 0.25, len(rr))           # chronological order within each column
            for off, r in zip(offsets, rr):
                q = np.array([f(x) for x in r["q"]])
                c = color(r["date"])
                if q.size > 1:                                     # spread over the raster sequence
                    lo, hi = np.percentile(q, [5, 95])
                    ax.plot([x0 + off] * 2, [lo, hi], color=c, lw=1.6, alpha=0.6, solid_capstyle="round", zorder=2)
                ax.scatter(x0 + off, q[r["i"]], s=28 if mk == "o" else 24, marker=mk, color=c, edgecolor="k", lw=0.4,
                           zorder=3)
        for k, x_mid in (("I", (n_i - 1) / 2), ("v", n_i + (len(cols_spec) - n_i - 1) / 2)):
            ax.text(x_mid, yhi * 1.05, titles[k], ha="center", va="bottom", fontsize=8, linespacing=1.5)
        ax.axvline(n_i - 0.5, color="k", lw=2.0, zorder=2)          # thick divider: two panels side by side
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
        ax.set_yscale("log"); ax.set_ylim(ylo, yhi)
        ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8) if ylo < t < yhi]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks(range(len(cols_spec))); ax.set_xticklabels([c[1] for c in cols_spec])
        ax.set_xlim(-0.5, len(cols_spec) - 0.5)
        ax.set_ylabel("Simulated / observed")
        ax.tick_params(axis="y", which="both", direction="in", right=True)
        ax.tick_params(axis="x", length=0)
        ax.text(-0.45, yhi / 1.18, "↑ simulation higher", ha="left", va="center", fontsize=7, color="0.35")
        ax.text(-0.45, ylo * 1.18, "↓ simulation lower", ha="left", va="center", fontsize=7, color="0.35")
        if iris is not None:
            ax.legend(handles=[Line2D([], [], ls="", marker=m, ms=5, mfc="0.6", mec="k", mew=0.4, label=l)
                               for m, l in (("o", "EIS"), ("s", "IRIS"))],
                      loc="lower right", fontsize=7, frameon=False, handletextpad=0.2)

        cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm([0, 1, 2, 3], 3), cmap=ListedColormap(group_colors)),
                          ax=ax, ticks=[0.5, 1.5, 2.5], pad=0.01, aspect=20)
        cb.ax.set_yticklabels(date_groups)
        cb.ax.tick_params(which="both", length=0)
        cb.ax.minorticks_off()
        cb.set_label("Observation date")
        name = "summary_syn_obs_eis" if iris is None else "summary_syn_obs_eis_iris"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    print("Saved", os.path.join(OUT_DIR, f"{name}.pdf"))

    print(f'{"EIS date":12s}{"n":>4s}{"I syn/obs":>11s}{"w68 obs":>9s}{"syn":>7s}{"w68 ratio":>11s}{"rasters 5-95%":>16s}')
    for r in sorted(rows, key=lambda r: r["date"]):
        a = r["rasters"][r["i_aia"]]
        qv = np.percentile([eis_ratio["v"](x) for x in r["rasters"]], [5, 95])
        print(f'{r["date"][:10]:12s}{len(r["rasters"]):4d}{eis_ratio["I"](a):11.2f}{a["w68_v_obs"]:9.2f}'
              f'{a["w68_v_syn"]:7.2f}{eis_ratio["v"](a):11.2f}{qv[0]:10.2f}-{qv[1]:.2f}')
    if iris is not None:
        print(f'{"IRIS date":12s}{"I obs":>8s}{"I syn":>7s}{"I syn/obs":>11s}{"w68 obs":>9s}{"syn":>6s}{"w68 ratio":>11s}{"masked":>8s}')
    for r in sorted(iris or [], key=lambda r: r["date"]):
        print(f'{r["date"][:10]:12s}{r["med_I_obs"]:8.1f}{r["med_I_syn"]:7.1f}{r["med_I_syn"] / r["med_I_obs"]:11.2f}'
              f'{r["w68_v_obs"]:9.2f}{r["w68_v_syn"]:6.2f}{r["w68_v_syn"] / r["w68_v_obs"]:11.2f}{r["frac_masked"]:8.1%}')


if __name__ == "__main__":
    rows = json.load(open(OUT_JSON)) if "--plot-only" in sys.argv else compute()
    plot(rows)
    iris_json = os.path.join(OUT_DIR, "summary_syn_obs_iris.json")      # from iris_si4_summary.py
    if os.path.exists(iris_json):
        plot(rows, json.load(open(iris_json)))
