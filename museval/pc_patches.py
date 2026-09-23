"""PlasmaCalcs fixes needed by ``museval.make_vdem``.

Two bugs made every Bifrost VDEM wrong (verified against PlasmaCalcs 2026.6.0,
Bifrost pl072050 snap 014).  Both are fixed here rather than in PlasmaCalcs so
that museval keeps working against an unpatched install.

1.  ``eos_mode='aux'`` for Bifrost -- temperature was clamped at the EOS table ceiling
    ----------------------------------------------------------------------------------
    For Bifrost, ``eos_mode`` defaults to ``'table'``, so T comes from ``eostable.dat``.
    That table stops at the energy corresponding to T = 4.968e6 K (log10 T = 6.696).
    ``vdem_pipeline`` sets ``tabin.extrapolate_kind='constant'``, which clips ``eperm``
    to the table range, so every voxel hotter than that is assigned *exactly* 4.968e6 K.

    In pl072050 snap 014, 41.4% of the voxels along the line of sight are above the
    ceiling, with true temperatures up to 9.53e6 K (log10 T = 6.979).  All of them
    collapsed into one logT bin, which then held 55% of the emission measure while
    being spatially featureless.  Worse, once clamped, neighbouring voxels have
    identical T, so ``interp_masked``'s correction factor |dclip/dT| evaluates 0/0,
    is filled with 0, and those segments contribute *no* emission at all: the run
    lost ~28% of the total EM on top of misplacing the rest.

    Bifrost writes its own, correctly extrapolated, temperature to aux as ``tg``.
    ``eos_mode='aux'`` reads that.  ne and P stay on the table, which is accurate
    there: in the corona the gas is fully ionised, so ne follows rho, not e
    (measured ne/(rho/m_H) = 0.863 against 0.868 predicted from the tabparam.in
    abundances -- agreement to better than 1%).

2.  ``vdem_on_grid`` -- the logT/vdop grids must be stated, not guessed
    ------------------------------------------------------------------
    ``vdem_pipeline`` derives both grids from a histogram of the raw data, keeping
    only bins above ``tg_percent``/``vel_percent`` (0.1%) of the *peak* density.
    That statistic is volume-weighted, and the box is overwhelmingly chromosphere,
    so the corona and the fast flows fall under the threshold and get cut off.
    The ``*_cut`` arguments are applied with ``np.min``/``np.max``, so they can only
    ever *shrink* what the histogram returned -- they cannot put back what it
    dropped.  ``np.arange(lo, hi, step)`` is then half-open, losing the top bin.

    Net effect for AIA settings ``aia_vdop=[-500, 500, 100]``: the saved VDEM had
    ``vdop = [-100, 0]``.  Two bins, asymmetric about zero, with no bin at all
    covering +50..+150 km/s -- that emission was silently discarded.

    ``vdem_on_grid`` below takes both grids as explicit bin centres and skips the
    histogram step entirely.
"""

import datetime
import re
import os

import numpy as np

__all__ = ['patch_plasmacalcs', 'grid_from_spec', 'vdem_on_grid', 'vdem_to_dataset']


_PATCHED = False


def patch_plasmacalcs():
    """Apply the Bifrost fixes to PlasmaCalcs. Idempotent; safe to call repeatedly."""
    global _PATCHED
    if _PATCHED:
        return

    import PlasmaCalcs as pc
    from PlasmaCalcs.hookups.bifrost.bifrost_snaps import BifrostScrSnap

    # --- snapname.idl.scr parsing: PlasmaCalcs 2026.6.0 reads the wrong regex group ---
    @classmethod
    def _idl_filename_to_snap_s(cls, filename, *, snapname):
        match = cls._check_idl_filename_pattern(snapname, filename)
        return match.group(1)

    BifrostScrSnap._idl_filename_to_snap_s = _idl_filename_to_snap_s

    # The real class, even if something already wrapped pc.BifrostCalculator
    # (e.g. an older copy of this shim pasted into a notebook).
    cls = getattr(pc, '_OriginalBifrostCalculator', None)
    if cls is None:
        cls = pc.BifrostCalculator
        if not isinstance(cls, type):
            raise TypeError('pc.BifrostCalculator is already wrapped by something that did '
                            'not record pc._OriginalBifrostCalculator; cannot patch safely.')
        pc._OriginalBifrostCalculator = cls

    # --- eos_mode='aux' for Bifrost: T from aux 'tg' instead of the EOS table ---
    cls.EOS_MODE_OPTIONS = {
        **cls.EOS_MODE_OPTIONS,
        'aux': ("read T directly from aux 'tg'. Required whenever the corona is hotter "
                "than the EOS table ceiling, which 'table' would silently clamp to. "
                "ne and P still come from the table."),
    }

    _orig_get_T = cls.get_T

    def get_T(self):
        """Temperature. With eos_mode='aux', read 'tg' from aux instead of the EOS table."""
        if self.eos_mode == 'aux':
            return self.load_maindims_var_across_dims('tg', u='K', dims=['snap'])
        return _orig_get_T(self)

    cls.get_T = get_T
    cls._EOS_MODE_TO_NE_VAR = {**cls._EOS_MODE_TO_NE_VAR, 'aux': 'ne_fromtable'}
    cls._EOS_MODE_TO_P_VAR = {**cls._EOS_MODE_TO_P_VAR, 'aux': 'P_fromtable'}

    # --- accept the 'snapname_NNN.idl' spelling that make_vdem uses ---
    # PlasmaCalcs 2026.6.0 wants the bare snapname and raises KeyError: 'u_l' on the
    # filename form, because it then fails to locate the .idl and gets empty params.
    def _compatible_bifrost_calculator(snapname=None, *args, **kwargs):
        if isinstance(snapname, (str, os.PathLike)):
            match = re.fullmatch(r'(.+)_\d+\.idl', os.path.basename(os.fspath(snapname)))
            if match:
                snapname = match.group(1)
        return cls(snapname, *args, **kwargs)

    pc.BifrostCalculator = _compatible_bifrost_calculator

    _PATCHED = True


def grid_from_spec(spec):
    """Turn a ``[min, max, step]`` triple into bin centres, inclusive of ``max``.

    ``np.arange`` is half-open, which is what dropped the top bin of every grid
    museval asked for.  ``grid_from_spec([-500, 500, 100])`` gives 11 bins from
    -500 to +500 (and, because the span is a whole number of steps, a bin centred
    on zero), where ``np.arange(-500, 500, 100)`` gives 10 bins stopping at +400.
    """
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 10)


def vdem_on_grid(ec, logT, vdop, *, los_dim='z', iz0=None, emiss_norm=1e27,
                 modelname='', author='', chunks=256, ncpu=12,
                 dst='pc_vdem_pipeline', plots=True, photosphere_check=False):
    """Compute a VDEM on explicit ``logT``/``vdop`` bin centres.

    Replaces ``MhdRadiativeLoader.vdem_pipeline``'s grid-choosing step; everything
    else (slicing, emiss, moments, metadata) follows it.  See the module docstring
    for why the grids must not be inferred from a histogram.

    Parameters
    ----------
    ec : a PlasmaCalcs calculator, already configured (units, emiss_mode, eos_mode).
    logT, vdop : 1D arrays of bin centres. Bin k spans centre +/- step/2, so these
        must bracket the data; anything outside is dropped silently.
    photosphere_check : bool
        Reproduce vdem_pipeline's photospheric moment plot (Doppler sign check).
        Off by default: it is a second full VDEM pass and took 1.7 h of the
        original 8.8 h pl072050 run, for a sanity plot.

    Returns
    -------
    xarray.DataArray over (x, y, logT, vdop).
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from PlasmaCalcs.tools import xarray_update_call_history

    logT = np.asarray(logT, dtype=float)
    vdop = np.asarray(vdop, dtype=float)
    os.makedirs(dst, exist_ok=True)

    ec.emiss_norm = emiss_norm
    ec.component = los_dim

    zcoord = ec.get_maindims_coords()[los_dim]
    top = int(np.argmax(zcoord))
    if iz0 is None:
        iz0 = int(np.argmin(np.abs(zcoord)))
    # Set the LOS slice but keep any others already on ec, so callers can subset
    # x/y for a quick test run. (vdem_pipeline replaced the whole dict instead.)
    ec.slices = {**dict(ec.slices),
                 los_dim: slice(top, iz0) if top < iz0 else slice(iz0, top)}

    ec.rcoords_logT = logT
    ec.rcoords_vdop_kms = vdop
    ec.tabin.extrapolate_kind = 'constant'   # for ne; T no longer comes from the table

    code = type(ec).__name__.replace('Calculator', '')
    stamp = f'{code}{modelname}_{author}_{datetime.datetime.now().date()!s}_{ec.snap}'

    if plots:
        _plot_inputs(ec, logT, vdop, os.path.join(dst, f'T_ulos_hist_{stamp}.png'))

    # chunks=None -> no chunking. PlasmaCalcs refuses to chunk a dim that is also
    # sliced, so this is what you want when computing a subset for a test.
    chunk_kw = {} if chunks is None else dict(chunks=dict(x=chunks))
    vdem = ec('vdem', ncpu=ncpu, **chunk_kw)
    if ec.vdem_varies_logD:
        vdem = vdem.sum(dim='logD')

    if plots:
        mom = vdem.pc.moments(dim='vdop')
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        mom.moment_0.sum(dim='logT').plot.imshow(
            norm=colors.LogNorm(vmin=1e-5, vmax=1e5), ax=ax[0])
        mom.moment_1.sum(dim='logT').plot.imshow(ax=ax[1], cmap='bwr')
        mom.moment_2.sum(dim='logT').plot.imshow(ax=ax[2])
        fig.tight_layout()
        fig.savefig(os.path.join(dst, f'vdem_moment_{stamp}.png'))
        plt.close(fig)

    if photosphere_check:
        cut = ec('vdem',
                 slices=dict(z=slice(iz0, iz0 + 5)),
                 emiss_mode='notrac_noopa',
                 vdem_logT=(0, np.arange(3.0, 4.5, 0.1)),
                 rcoords_vdop_kms=np.arange(-1.0e1, 1.0e1, 1),
                 rcoords_logD_cgs=None,
                 vdem_mode='nointerp',
                 vdem_ignores_photosphere=False,
                 vdem_loopdim=None,
                 ncpu=ncpu, component=los_dim, **chunk_kw)
        mom = cut.pc.moments(dim='vdop')
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        mom.moment_0.sum(dim='logT').plot(ax=ax[0])
        mom.moment_1.sum(dim='logT').plot(vmin=-5, vmax=5, cmap='bwr', ax=ax[1])
        ax[1].set_title('Doppler velocity (positive must be downward)')
        fig.tight_layout()
        fig.savefig(os.path.join(dst, f'vdem_moment_ph_{stamp}.png'))
        plt.close(fig)

    xarray_update_call_history(vdem, locals(), vdem_on_grid)
    return vdem


def vdem_to_dataset(vdem, ec=None):
    """Wrap a VDEM DataArray as the Dataset ``vdem_pipeline`` used to return.

    Same variable name, coord attributes and provenance, so anything downstream
    (museval's own save block, muse.synthesis, the synth notebooks) is unaffected.
    """
    from PlasmaCalcs.tools import _xarray_save_prep

    out = vdem.to_dataset(name='vdem')
    # _xarray_save_prep drops PlasmaCalcs' rich metadata (Snap/Component objects,
    # which zarr cannot hold) and adds pc__version__, commit hash and datetime.
    out = _xarray_save_prep(out, add_code_snapshot_info=dict(default='_undefined_'))[0]
    out.x.attrs['long_name'] = 'X'
    out.y.attrs['long_name'] = 'Y'
    out.x.attrs['units'] = 'cm'
    out.y.attrs['units'] = 'cm'
    out.vdem.attrs['units'] = '1e27 / cm5'
    out.vdem.attrs['description'] = 'DEM(T,vel,x,y)'
    out.vdop.attrs['long_name'] = r'v$_{Doppler}$'
    out.vdop.attrs['units'] = 'km/s'
    out.logT.attrs['long_name'] = r'log$_{10}$(T)'
    out.logT.attrs['units'] = r'log$_{10}$ (K)'
    if ec is not None:
        out.attrs['eos_mode'] = ec.eos_mode
        out.attrs['T_source'] = "aux 'tg'" if ec.eos_mode == 'aux' else 'EOS table'
    return out


def _plot_inputs(ec, logT, vdop, savepath, stride=4):
    """Histograms of log10(T) and u_los, with the chosen grid edges overlaid.

    Subsampled by ``stride`` in x and y -- the full box is ~2e9 points and
    materialising T and u_los over all of it costs ~17 GB for a picture.
    The y axis is logarithmic so that the corona stays visible beside the far
    more voluminous chromosphere: it is exactly that contrast which defeated
    vdem_pipeline's 0.1%-of-peak threshold.
    """
    import matplotlib.pyplot as plt
    sub = dict(ec.slices)
    for d in ('x', 'y'):
        sub[d] = slice(None, None, stride)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    with ec.using(slices=sub):
        np.log10(ec('T')).plot.hist(density=True, bins=60, ax=ax[0])
        (ec('u', units='si') / 1e3).plot.hist(density=True, bins=60, ax=ax[1])
    for axis, grid, label in ((ax[0], logT, 'logT'), (ax[1], vdop, 'vdop')):
        half = (grid[1] - grid[0]) / 2
        axis.axvline(grid[0] - half, color='r', ls='--', label=f'{label} grid edges')
        axis.axvline(grid[-1] + half, color='r', ls='--')
        axis.set_yscale('log')
        axis.legend()
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
