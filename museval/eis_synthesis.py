"""
Synthesis of Hinode/EIS line profiles, intensities and Doppler moments from a VDEM.

The EIS response uses the same CHIANTI settings as the AIA response built by
``museval.utils.get_response`` (abundance, pressure, and logT/vdop grids taken from the
VDEM; minimum abundance too when all ions in the window are used), so synthetic EIS and AIA observables are directly
comparable. The defaults are read from get_response's signature, so they follow it.

The response uses unit effective area, so intensities come out in erg cm-2 s-1 sr-1,
the same units as eispac's fitted intensity maps; no EIS calibration is needed.
"""
import inspect
import numpy as np
import xarray as xr
from muse import logger
from museval.utils import get_response

__all__ = [
    "eis_line_response",
    "eis_synthesis",
    "eis_degrade_vdem",
]

# CHIANTI settings of the AIA response; EIS uses the same unless told otherwise.
AIA_DEFAULTS = {k: p.default for k, p in inspect.signature(get_response).parameters.items()
                if k in ("abund", "press", "minimum_abundance")}
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# **************************************************

def eis_line_response(vdem,
                      ionstr = ["fe_12"],
                      wvlr = [195.10, 195.20],   # Fe XII 195.119 + its 195.179 blend, as in eispac's fe_12_195_119.2c
                      abund = None, press = None, minimum_abundance = None,
                      min_frac = 1e-2,
                      instr_fwhm = 0.0,          # Angstrom
                      pad = 0.3,                 # Angstrom
                      dma = 4.9,                 # mAngstrom
                      ):
    '''
    Computes the spectral response of EIS line(s) on the logT/vdop grid of the VDEM.

    Parameters:
    -------------------------
    vdem:     xr.Dataset, VDEM whose logT and vdop grids the response is built on
              (as get_response does by default), so eis_synthesis needs no regridding.
    ionstr:   list, CHIANTI ions, default ["fe_12"]. None takes every ion in wvlr
              (blends from other elements), filtered by minimum_abundance as for AIA.
    wvlr:     list, wavelength range [A] from which lines are taken, default [195.10, 195.20]
    abund, press, minimum_abundance: CHIANTI settings, default to those of get_response
              (minimum_abundance only with ionstr=None: muse ignores ionList when it is set)
              Pass the same values here if you override them there.
    min_frac: float, keep lines whose peak G(T) is at least this fraction of the strongest
              line in wvlr; drops the many negligible transitions, default 1e-2
    instr_fwhm: float, instrumental FWHM [A]. Default 0 gives the intrinsic profile (as for
              AIA). For EIS use cube.meta['slit_width'] from eispac (averaged over the FOV).
    pad:      float, spectral window extends this far [A] beyond the outermost lines
    dma:      float, wavelength sampling [mA] of the synthetic spectrum

    Returns:
    -------------------------
    resp:      xr.Dataset, response (SG_resp) in 1e-27 cm3 erg / (Angstrom s sr)
    line_list: xr.Dataset, CHIANTI line list with G(T) of the lines kept
    '''
    import astropy.units as u
    from muse.instr.utils import chianti_gofnt_linelist, create_resp_func, create_eff_area_xarray

    abund = AIA_DEFAULTS["abund"] if abund is None else abund
    press = AIA_DEFAULTS["press"] if press is None else press
    if ionstr is None and minimum_abundance is None:
        minimum_abundance = AIA_DEFAULTS["minimum_abundance"]

    logT = xr.DataArray(np.asarray(vdem.logT.values, dtype=float), dims="logT")
    vdop_kms = np.asarray(vdem.vdop.values, dtype=float)
    if vdop_kms.size > 1 and np.median(np.diff(vdop_kms)) > 20:
        logger.info(f"*** WARNING: VDEM vdop step is {np.median(np.diff(vdop_kms)):.0f} km/s, too coarse for"
                    " line profiles and Doppler moments. Make the VDEM on the MUSE or IRIS vdop grid.")

    pressure = xr.DataArray(np.array([press]), dims="pressure")
    logger.info(f"*** Constructing {ionstr} line list in {wvlr} A using pressure = {press:0.1e}, abundance {abund}")
    line_list = chianti_gofnt_linelist(temperature = 10**logT,
                                       pressure = pressure,
                                       abundance = abund,
                                       wavelength_range = np.array(wvlr),
                                       ionList = ionstr,
                                       minimum_abundance = minimum_abundance,
                                       )
    strength = line_list.gofnt.max("logT").isel(pressure=0, abundance=0)
    keep = np.flatnonzero((strength >= min_frac * strength.max()).values)
    line_list = line_list.isel(trans_index=keep)
    logger.info(f"*** Keeping {keep.size} line(s): {np.round(np.atleast_1d(line_list.wvl.values), 3)} A")

    wvlmin = float(line_list.wvl.min()) - pad
    wvlmax = float(line_list.wvl.max()) + pad
    # Unit effective area keeps the response in erg, like eispac's calibrated intensities.
    effective_area = create_eff_area_xarray(np.ones(2), [wvlmin, wvlmax], [wvlr[0]])
    resp = create_resp_func(line_list,
                            vdop = vdop_kms * u.km / u.s,
                            instr_width = instr_fwhm * FWHM_TO_SIGMA,  # create_resp_func wants sigma in A
                            effective_area = effective_area.eff_area,
                            wvlr = [wvlmin, wvlmax],
                            dma = dma,
                            num_lines_keep = keep.size,
                            )
    # muse labels the response as if an area in cm2 were folded in; here the area is unity.
    resp.SG_resp.attrs["units"] = "1e-27 cm3 erg / (Angstrom s sr)"
    resp = resp.assign_attrs(abund = abund, press = press, instr_fwhm = instr_fwhm)
    return resp, line_list

# **************************************************

def eis_synthesis(vdem, resp,
                  chunk_size = 128,
                  return_spectrum = False,
                  vdop_sign = 1,
                  ):
    '''
    Synthesizes EIS spectra from a VDEM and returns intensity, Doppler shift and width
    of every line in the response.

    Parameters:
    -------------------------
    vdem:      xr.Dataset or str (path to zarr), VDEM in 1e27 cm-5, dims x, y, logT, vdop
    resp:      xr.Dataset, response from eis_line_response(vdem)
    chunk_size: int, number of x columns synthesized at a time, to bound memory
    return_spectrum: bool, also return the spectra (large for a full simulation box)
    vdop_sign: int, multiplies the Doppler shift. The output follows the VDEM's vdop sign;
               EIS has positive = redshift. Use -1 if the VDEM's convention is opposite.

    Returns:
    -------------------------
    xr.Dataset with, per line (dim "line", e.g. "Fe XII 195.119"):
        intensity [erg cm-2 s-1 sr-1], comparable to eispac's fitted intensity
        vdop      [km/s], first moment
        width     [km/s], Gaussian sigma (second moment)
        spectrum  [erg cm-2 s-1 sr-1 A-1], only if return_spectrum
    '''
    from muse.synthesis.synthesis import vdem_synthesis
    from muse.utils.utils import calculate_moments

    if isinstance(vdem, str):
        vdem = xr.open_zarr(vdem)
    # vdem_synthesis() is a raw einsum, so the grids must match element for element.
    for dim in ("logT", "vdop"):
        if vdem.sizes[dim] != resp.sizes[dim] or not np.allclose(vdem[dim], resp[dim], atol=1e-6):
            raise ValueError(f"VDEM and response {dim} grids differ: build the response with eis_line_response(vdem).")

    out = []
    for ix in range(0, vdem.sizes["x"], chunk_size):
        sub = vdem.isel(x=slice(ix, ix + chunk_size)).compute()
        spec = vdem_synthesis(sub, resp, sum_over=["logT", "vdop"])
        mom = calculate_moments(spec, moment_dim="wavelength")
        dlam = float(np.median(np.diff(spec.wavelength)))
        res = xr.Dataset({"intensity": spec.flux.sum("wavelength") * dlam,
                          "vdop": vdop_sign * mom["1st"],
                          "width": mom["2nd"]})
        if return_spectrum:
            res["spectrum"] = spec.flux
        out.append(res)
    syn = xr.concat(out, dim="x")
    syn = syn.squeeze([d for d in ("pressure", "abundance", "band") if d in syn.dims], drop=True)
    syn.intensity.attrs["units"] = "erg / (cm2 s sr)"
    syn.vdop.attrs["units"] = "km / s"
    syn.width.attrs["units"] = "km / s"
    if return_spectrum:
        syn.spectrum.attrs["units"] = "erg / (cm2 Angstrom s sr)"
    return syn.assign_attrs(resp.attrs)

# **************************************************

def eis_degrade_vdem(vdem, dx = None,
                     pixel = (1.0, 1.0),      # arcsec, x (raster step) and y
                     psf_fwhm = (3.0, 3.0),   # arcsec
                     arcsec2Mm = 0.725,
                     ):
    '''
    Smooths the VDEM with a Gaussian PSF and rebins it to EIS pixels. Synthesis is linear
    in the VDEM, so this equals degrading the synthetic spectra, at a fraction of the cost.

    Parameters:
    -------------------------
    vdem:     xr.Dataset, VDEM with dims x, y, logT, vdop
    dx:       float, simulation pixel size [Mm], same in x and y. Default: spacing of the
              x coordinate, converted with its units attribute (make_vdem writes cm;
              taken as Mm if the attribute is missing)
    pixel:    tuple, EIS pixel size [arcsec] in x (raster step) and y, default (1, 1)
    psf_fwhm: tuple, EIS PSF FWHM [arcsec] in x and y, default (3, 3)
    arcsec2Mm: float, Mm per arcsec, default 0.725 (1 AU)
    '''
    import astropy.units as u
    from scipy.ndimage import gaussian_filter

    if dx is None:
        dx = (float(np.median(np.diff(vdem.x.values))) * u.Unit(vdem.x.attrs.get("units", "Mm"))).to_value(u.Mm)
        logger.info(f"*** Simulation pixel size from the x coordinate: {dx:.4f} Mm")
    sigma = [f * FWHM_TO_SIGMA * arcsec2Mm / dx for f in psf_fwhm]
    nbin = [max(1, int(round(p * arcsec2Mm / dx))) for p in pixel]
    logger.info(f"*** PSF sigma {sigma[0]:.1f} x {sigma[1]:.1f} sim pixels, rebinning by {nbin[0]} x {nbin[1]}")
    out = []
    # One temperature bin at a time so a full simulation box never sits in memory.
    for it in range(vdem.sizes["logT"]):
        sl = vdem.vdem.isel(logT=it).transpose("x", "y", ...).compute()
        sm = gaussian_filter(sl.values, sigma=sigma + [0] * (sl.ndim - 2), mode="wrap")
        out.append(sl.copy(data=sm).coarsen(x=nbin[0], y=nbin[1], boundary="trim").mean())
    return xr.concat(out, dim="logT").transpose(*vdem.vdem.dims).to_dataset(name="vdem").assign_attrs(vdem.attrs)
