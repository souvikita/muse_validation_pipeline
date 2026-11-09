"""Observed (IRIS) vs synthetic (Bifrost en024031, snapshot 10) Si IV 1402.77 for the six IRIS QS datasets,
for the EIS + IRIS summary figure (plotted by eis_fe12_summary.py).

Fits:      rad_cal_iris_gfit.ipynb, unchanged: first raster of the series, irispy radiometric calibration,
           Const1D + Gaussian1D fit with TRFLSQFitter through parallel_fit_dask, saved as
           rad_cal_iris_fit_results_1402_<date>.npz (flux, core_shift, sigma) in each dataset folder.
           Existing files are read, not refit.
Intensity: med(I_sim)/med(I_obs). I_obs is the radiometrically calibrated line flux integrated over 1402.77 +- 0.3 A
           (continuum subtracted) in ALL slit pixels with data, so the median is not biased by dropping faint pixels.
Velocity:  w68(v_sim)/w68(v_obs), w68 = (p84 - p16)/2 as for EIS Fe XII. v_obs from the Gaussian fits, masked as in
           iris_obs_syn_comparison.ipynb (slit-end fill, < 11 photons, failed fits) and referenced to O I 1355.6.
           No noise correction: v_obs includes measurement scatter, which raises w68(v_obs) most in the 8-s rasters.
Synthetic: en024031_si4_1402_it=10.npz (itot, utot) degraded to each raster's pixels with the IRIS PSF and slit
           (iris_degrade from iris_obs_syn_comparison.ipynb); velocity intensity-weighted.

Usage: python -I iris_si4_summary.py
"""
import os
import glob
import json
import warnings
import numpy as np

C_KMS = 299792.458
W0, W_OI = 1402.77, 1355.598                      # Angstrom
IRIS_DIR = "/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/QS_datasets/IRIS_only/IRIS_datasets/"
SYN_FILE = "/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/QS_datasets/Bifrost/en024031_si4_1402_it=10.npz"
OUT_JSON = "/Users/souvikb/MUSE_outputs/figures_paper/QS_valid/summary_syn_obs_iris.json"
DX_SIM_KM = 31.25                                 # en024031: 768 px over 24 Mm
SLIT_ARCSEC, PSF_FWHM_ARCSEC = 0.33, 0.33
IRIS_DATES = ["2014-06-07T12:09:00", "2014-08-21T23:21:00", "2015-07-10T09:03:00",
              "2019-08-17T04:40", "2019-12-12T12:37:00", "2020-11-19T17:33:00"]


def w68(a, axis=None):
    """Half-width of the central 68% of a distribution, (p84 - p16)/2; equals sigma for a Gaussian."""
    p16, p84 = np.nanpercentile(a, [16, 84], axis=axis)
    return 0.5 * (p84 - p16)


def raster_file(dset):
    return sorted(glob.glob(os.path.join(dset, "*_raster", "*_raster_t000_r00000.fits")))[0]


def fit_file(dset):
    return os.path.join(dset, f"rad_cal_iris_fit_results_1402_{os.path.basename(dset)[:10]}.npz")


def calibrated_si_iv(dset, window="Si IV 1403"):
    """Radiometrically calibrated Si IV 1403 window of the first raster, erg / (s sr cm2 A)."""
    from irispy.io import read_files
    from irispy.utils.spectrograph import radiometric_calibration
    raster = read_files(raster_file(dset), memmap=False)
    return radiometric_calibration(raster[window][0])


def fit_si_iv(cal, out, rest_A=W0):
    """rad_cal_iris_gfit.ipynb cells 13-26, unchanged apart from taking the calibrated cube as input."""
    import astropy.units as u
    import astropy.constants as constants
    from astropy.modeling import models as m
    from astropy.modeling.fitting import TRFLSQFitter, parallel_fit_dask
    from dask.distributed import Client

    si_iv_1402_core = rest_A / 10 * u.nm
    spatial_mean = cal.rebin((*cal.data.shape[:-1], 1))[0, 0, :]
    wavelength_axis = cal.axis_world_coords("em.wl")[0].to(u.nm)
    core_window = np.abs(wavelength_axis - si_iv_1402_core) < 0.15 * u.nm
    initial_model = m.Const1D(amplitude=1 / cal.exposure_time.value[0] * cal.unit) + m.Gaussian1D(
        amplitude=np.nanmax(spatial_mean.data[core_window]) * cal.unit, mean=si_iv_1402_core, stddev=0.005 * u.nm)
    fitter = TRFLSQFitter()
    average_fit = fitter(initial_model, wavelength_axis, spatial_mean.data * spatial_mean.unit, filter_non_finite=True)
    filtered_data = np.where(cal.data < 0, 0, cal.data)
    filtered_data = np.where(np.isfinite(filtered_data), filtered_data, 0)

    client = Client(n_workers=10, threads_per_worker=1, memory_limit="5GB", dashboard_address=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_params = parallel_fit_dask(data=filtered_data, data_unit=cal.unit, model=average_fit, fitting_axes=2,
                                          world=(wavelength_axis,), fitter=fitter, scheduler=client)
    client.close()

    net_flux = np.sqrt(2 * np.pi) * fitted_params.amplitude_1 * fitted_params.stddev_1.quantity
    core_shift = (fitted_params.mean_1.quantity.to(u.nm) - si_iv_1402_core) / si_iv_1402_core * constants.c.to(u.km / u.s)
    sigma = fitted_params.stddev_1.quantity.to(u.nm) / si_iv_1402_core * constants.c.to(u.km / u.s)
    np.savez(out, flux=net_flux.to(u.erg / u.s / u.cm**2 / u.sr).data, core_shift=core_shift.data, sigma=sigma.data)
    print("Saved", out, flush=True)


def level2(dset):
    """Si IV 1403 and O I 1356 windows of the level-2 raster in DN, plus the headers."""
    from astropy.io import fits
    with fits.open(raster_file(dset), memmap=False) as hdul:
        h0 = hdul[0].header
        win = {h0[f"TDESC{i}"]: i for i in range(1, h0["NWIN"] + 1)}
        out = dict(h0=h0, hsi=hdul[win["Si IV 1403"]].header, dn_si=hdul[win["Si IV 1403"]].data.astype(float),
                   hoi=hdul[win["O I 1356"]].header, dn_oi=hdul[win["O I 1356"]].data.astype(float))
    for k in ("dn_si", "dn_oi"):
        out[k][out[k] < -100] = np.nan                                  # -200 marks missing data in level 2
    return out


def wave(h):
    return h["CRVAL1"] + (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"]) * h["CDELT1"]


def o_i_reference(l2):
    """Doppler shift of O I 1355.6 in the FOV-averaged spectrum (iris_obs_syn_comparison.ipynb cell 10)."""
    from astropy.modeling import models as m
    from astropy.modeling.fitting import TRFLSQFitter
    wl = wave(l2["hoi"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = np.nanmean(l2["dn_oi"].reshape(-1, l2["dn_oi"].shape[-1]), axis=0)
    w = np.abs(wl - W_OI) < 0.25
    fit = TRFLSQFitter()(m.Gaussian1D(np.ptp(spec[w]), W_OI, 0.04) + m.Const1D(spec[w].min()), wl[w], spec[w])
    return (fit.mean_0.value - W_OI) / W_OI * C_KMS


def photons(dn, h, w0):
    """Continuum-subtracted line counts within w0 +- 0.3 A (FUV: 4 photons / DN)."""
    wl = wave(h)
    core = np.abs(wl - w0) < 0.3
    cont = (np.abs(wl - w0) > 0.5) & (np.abs(wl - w0) < 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 4 * np.nansum(dn[..., core] - np.nanmedian(dn[..., cont], axis=-1, keepdims=True), axis=-1)


def fit_ok(flux, v, sigma, n):
    """Pixels kept in iris_obs_syn_comparison.ipynb: >= 11 photons and a sane Gaussian."""
    return (n >= 11) & (flux > 0) & (sigma > 3) & (sigma < 60) & (np.abs(v) < 60)



def iris_degrade(img, dx, psf_fwhm, box, pixel):
    """iris_obs_syn_comparison.ipynb cell 8: convolve a periodic 2D (x, y) image with a Gaussian PSF and the pixel
    boxcar, then sample it on the IRIS raster grid. Lengths in simulation pixels; box and pixel are (x, y)."""
    from scipy.ndimage import map_coordinates
    kx = np.fft.fftfreq(img.shape[0], d=dx)[:, None]
    ky = np.fft.rfftfreq(img.shape[1], d=dx)[None, :]
    sig = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
    mtf = np.exp(-2 * (np.pi * sig) ** 2 * (kx ** 2 + ky ** 2)) * np.sinc(box[0] * kx) * np.sinc(box[1] * ky)
    smooth = np.fft.irfft2(np.fft.rfft2(img) * mtf, s=img.shape)
    nx, ny = int(img.shape[0] * dx // pixel[0]), int(img.shape[1] * dx // pixel[1])
    xs, ys = np.meshgrid((np.arange(nx) + 0.5) * pixel[0] / dx, (np.arange(ny) + 0.5) * pixel[1] / dx, indexing="ij")
    return map_coordinates(smooth, [xs, ys], order=1, mode="grid-wrap")


def analyse(dset, syn):
    cal = calibrated_si_iv(dset)
    if not os.path.exists(fit_file(dset)):
        fit_si_iv(cal, fit_file(dset))
    fr = np.load(fit_file(dset))
    l2 = level2(dset)
    h0, hs = l2["h0"], l2["hsi"]

    # ---------------- observed ----------------
    wl = wave(hs)
    core = np.abs(wl - W0) < 0.3
    cont = (np.abs(wl - W0) > 0.5) & (np.abs(wl - W0) < 1.0)
    n_phot = photons(l2["dn_si"], hs, W0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.asarray(cal.data, dtype=float)
        c = np.where(np.isfinite(c) & (l2["dn_si"] > -100), c, np.nan)
        I_int = np.nansum(c[..., core] - np.nanmedian(c[..., cont], axis=-1, keepdims=True), axis=-1) * hs["CDELT1"]
    v_ref = o_i_reference(l2)
    I_fit, v, s = fr["flux"], fr["core_shift"] - v_ref, fr["sigma"]
    fill = np.all(np.isnan(l2["dn_si"]), axis=-1) | (np.ptp(I_fit, axis=0) == 0)[None, :]
    bad = fill | ~fit_ok(I_fit, v, s, n_phot)
    good = ~bad
    obs_I = I_int[~fill]

    # ---------------- synthetic ----------------
    kpa = h0["DSUN_OBS"] * 1e-3 * np.deg2rad(1 / 3600)                  # km per arcsec
    to_sim = kpa / DX_SIM_KM
    kw = dict(dx=1.0, psf_fwhm=PSF_FWHM_ARCSEC * to_sim, box=(SLIT_ARCSEC * to_sim, hs["CDELT2"] * to_sim),
              pixel=(hs["CDELT3"] * to_sim, hs["CDELT2"] * to_sim))
    syn_I = iris_degrade(syn["itot"], **kw)
    syn_v = (iris_degrade(syn["itot"] * syn["utot"], **kw) / syn_I).ravel()   # intensity-weighted

    r = dict(date=os.path.basename(dset), file=os.path.basename(raster_file(dset)), date_obs=h0["DATE_OBS"],
             exptime=float(h0["EXPTIME"]), step=float(hs["CDELT3"]), ypix=float(hs["CDELT2"]),
             n_pix=int((~fill).sum()), frac_masked=float(bad[~fill].mean()), n_syn=int(syn_I.size), v_ref_OI=float(v_ref),
             med_I_obs=float(np.nanmedian(obs_I)), med_I_obs_fit_masked=float(np.nanmedian(I_fit[good])),
             int_over_fit=float(np.nanmedian(I_int[good] / I_fit[good])), med_I_syn=float(np.nanmedian(syn_I)),
             med_v_obs=float(np.nanmedian(v[good])), med_v_syn=float(np.nanmedian(syn_v)),
             w68_v_obs=float(w68(v[good])), w68_v_syn=float(w68(syn_v)))
    print(f"{r['date'][:10]}  masked {r['frac_masked']:5.1%}  I obs {r['med_I_obs']:6.1f} (fit, masked {r['med_I_obs_fit_masked']:6.1f}; "
          f"int/fit {r['int_over_fit']:.2f})  syn {r['med_I_syn']:5.1f}  ratio {r['med_I_syn'] / r['med_I_obs']:.2f} | "
          f"w68 obs {r['w68_v_obs']:5.2f} syn {r['w68_v_syn']:5.2f} ratio {r['w68_v_syn'] / r['w68_v_obs']:.2f} | "
          f"med v obs {r['med_v_obs']:+.1f} syn {r['med_v_syn']:+.1f}", flush=True)
    return r


if __name__ == "__main__":
    syn = np.load(SYN_FILE)
    syn = dict(itot=syn["itot"], utot=syn["utot"])
    rows = [analyse(os.path.join(IRIS_DIR, d), syn) for d in IRIS_DATES]
    json.dump(rows, open(OUT_JSON, "w"), indent=1)
    print("Saved", OUT_JSON)
