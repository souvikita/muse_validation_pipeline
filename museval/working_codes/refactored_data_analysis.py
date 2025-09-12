import os
import numpy as np
import glob
import shutil
import xarray as xr
from astropy.time import Time
import museval
from museval.utils import get_response, find_response, aia_synthesis, wavelength_in_cube
from museval.io import create_session, is_complete
from muse.synthesis.synthesis import transform_resp_units, vdem_synthesis
import matplotlib.pyplot as plt
from muse.instr.utils import convert_resp2muse_ciresp
import astropy.units as u
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
import sunpy.map
import sunpy.visualization.colormaps as cm
from sunpy.time import parse_time
from dateutil.tz import gettz
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import eispac
from museval.eis_calibration.eis_calib_2023 import calib_2023 as eis_calib_2023
from matplotlib import gridspec
import warnings
import tkinter as tk
from tkinter import filedialog

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings(
    "ignore",
    message="Failed to open Zarr store with consolidated metadata.*",
    category=RuntimeWarning
)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Paths - modify these as needed
RESPONSE_PATH = '/Users/souvikb/MUSE_outputs/response/'
TEXT_FILES_PATH = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/HOP_307/'
EIS_DATA_PATH = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/HOP_307/'
VDEM_PATH = '/Users/souvikb/MUSE_outputs/vdems/'
WORK_DIR = '/Users/souvikb/MUSE_outputs/pipeline_figs_sims/'

# Constants
AIA_CHANNELS = [131, 171, 193, 211]
EIS_LINES = [195.120, 197.860, 284.160]  # Fe XII, Fe IX and Fe XV
TARGET_TEMPLATES = [
    'fe_12_195_119.2c.template.h5',
    'fe_09_197_862.1c.template.h5',
    'fe_15_284_160.2c.template.h5'
]
EIS_CMAPS = ['sdoaia193', 'Blues_r', 'sohoeit284']
COLORS_CHANNEL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
NUM_BINS = 40

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_environment():
    """Set up environment variables"""
    os.environ['RESPONSE'] = RESPONSE_PATH
    os.environ['text_files_path'] = TEXT_FILES_PATH
    os.environ['eis_data_path'] = EIS_DATA_PATH
    os.environ['vdem_path'] = VDEM_PATH

def select_vdem_file():
    """Interactive VDEM file selection"""
    root = tk.Tk()
    root.withdraw()
    vdem_file = filedialog.askdirectory(
        title="Select a VDEM .zarr file",
        initialdir=VDEM_PATH
    )
    print(f"\nSelected folder: {vdem_file}")
    return vdem_file

def get_text_files():
    """Get sorted list of text files"""
    txt_files = glob.glob(os.path.join(TEXT_FILES_PATH, "*.txt"))
    return sorted(txt_files)

def add_colorbar_and_styling(fig, ax, im, label):
    """Add colorbar and styling to axis"""
    ax.grid(False)

    # Configure coordinate labels
    lon, lat = ax.coords[0], ax.coords[1]
    lon.set_ticks_position('b')
    lon.set_ticklabel_position('b')
    lat.set_ticks_position('l')
    lat.set_ticklabel_position('l')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(direction='out')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label(label)

def save_figure(fig, output_dir, filename):
    """Save figure to specified location"""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# AIA DATA PROCESSING FUNCTIONS
# ============================================================================

def create_aia_histogram(muse_aia_data, mhd_code, snapname):
    """Create and save AIA synthesis histograms"""
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(4, 8), sharex=True)

    for i, channel in enumerate(AIA_CHANNELS):
        synth_data = muse_aia_data.flux.sel(channel=channel).to_numpy()
        min_val = max(muse_aia_data.flux.sel(channel=channel).min(), 1e-1)
        max_val = muse_aia_data.flux.sel(channel=channel).max()
        bins = np.logspace(np.log10(min_val), np.log10(max_val), NUM_BINS)

        weights = np.ones(len(synth_data.ravel())) / len(synth_data.ravel())

        # Regular histogram
        ax[0].hist(synth_data.ravel(), bins=bins, histtype='step',
                  color=COLORS_CHANNEL[i], label=f'AIA {channel}', weights=weights)

        # Cumulative histogram
        ax[1].hist(synth_data.ravel(), bins=bins, histtype='step',
                  color=COLORS_CHANNEL[i], cumulative=True, weights=weights)
        ax[1].axvline(x=np.mean(synth_data.ravel()),
                     color=COLORS_CHANNEL[i], lw=0.5, ls='--',
                     label=f'$\\mu_{{{channel}}}$ {np.mean(synth_data.ravel()):.2f}')

    # Configure axes
    ax[0].legend(loc='best')
    ax[0].set_xscale('log')
    ax[0].set_ylabel('Fraction')

    ax[1].set_xlabel(r'Intensity [DNs$^{-1}$]')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('ECDF')
    ax[1].set_xlim([1, muse_aia_data.flux.max().to_numpy() * 1.1])
    ax[1].legend(loc='best', fontsize='small')

    fig.suptitle(f'{mhd_code} simulation: {snapname}', y=0.98, ha='center')
    plt.tight_layout()

    # Save figure
    output_dir = f'{WORK_DIR}/{mhd_code}_{snapname}_synth_histograms/'
    save_figure(fig, output_dir, 'histograms.png')

def process_aia_cutouts(cutout_files, date):
    """Process AIA cutout files and create visualization"""
    aia_data = {}
    n_lines = len(AIA_CHANNELS)

    # Load first map to calculate dimensions
    sample_map = sunpy.map.Map(cutout_files[0])
    ny, nx = sample_map.data.shape
    aspect = ny / nx

    panel_width = 4
    fig_width = panel_width * n_lines
    fig_height = panel_width * aspect

    fig = plt.figure(figsize=(fig_width, fig_height))

    for idx, (line, cutout_file) in enumerate(zip(AIA_CHANNELS, cutout_files)):
        aia_map = sunpy.map.Map(cutout_file)
        aia_data[line] = aia_map

        exp_time = aia_map.exposure_time
        normalized_data = aia_map / exp_time

        ax = fig.add_subplot(1, n_lines, idx + 1, projection=aia_map.wcs)
        im = normalized_data.plot(axes=ax, vmin=0)

        add_colorbar_and_styling(fig, ax, im, r'DNs$^{-1}$')

    plt.tight_layout()
    return aia_data, fig

def create_comparison_histogram(aia_data, muse_aia_data, mhd_code, snapname, date):
    """Create comparison histogram between observed and synthesized AIA data"""
    region = 'plage'
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for i, channel in enumerate(AIA_CHANNELS):
        row, col = i // 2, i % 2

        # Observed data
        exp_time = aia_data[channel].exposure_time
        obs_data = np.ravel(aia_data[channel].data / exp_time)

        # Synthesized data
        synth_data = muse_aia_data.flux.sel(channel=channel).to_numpy()

        # Create combined bins
        combined = np.concatenate([obs_data.value, synth_data.ravel()])
        min_val = max(combined.min(), 1e-1)
        max_val = combined.max()
        bins = np.logspace(np.log10(min_val), np.log10(max_val), NUM_BINS)

        # Plot observed data
        weights_obs = np.ones(len(obs_data.value)) / len(obs_data.value)
        ax[row][col].hist(obs_data.value, bins=bins, label=f'AIA {region}',
                         cumulative=True, histtype='step', weights=weights_obs,
                         color=COLORS_CHANNEL[i])
        ax[row][col].axvline(x=np.mean(obs_data.value), color=COLORS_CHANNEL[i],
                           lw=1, ls='-',
                           label=f'$\\mu_{{\\mathrm{{obs}}}}$: {np.mean(obs_data.value):.2f}')

        # Plot synthesized data
        weights_synth = np.ones(len(synth_data.ravel())) / len(synth_data.ravel())
        ax[row][col].hist(synth_data.ravel(), bins=bins, label=f'{mhd_code} {snapname}',
                         cumulative=True, histtype='step', weights=weights_synth,
                         color='tab:purple', ls='-.', lw=2.)
        ax[row][col].axvline(x=np.mean(synth_data.ravel()), color='tab:purple',
                           lw=1, ls='-.',
                           label=f'$\\mu_{{\\mathrm{{syn}}}}$: {np.mean(synth_data.ravel()):.2f}')

        # Styling
        ax[row][col].set_xlabel(fr'{channel} Intensity [DNs$^{-1}$]')
        ax[row][col].set_xlim([1, max_val * 1.1])
        ax[row][col].set_xscale('log')
        ax[row][col].set_ylabel('ECDF')
        ax[row][col].legend(loc='lower right')
        ax[row][col].axhline(y=0.25, color='black', ls='--')
        ax[row][col].axhline(y=0.75, color='black', ls='--')

    title = f"AIA {region} {date} vs {mhd_code} {snapname}"
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    return fig

# ============================================================================
# EIS DATA PROCESSING FUNCTIONS
# ============================================================================

def process_single_eis_line(data_file, eis_line, target_template, cutouts_path, date):
    """Process a single EIS spectral line"""
    try:
        data_cube = eispac.read_cube(data_file, eis_line, apply_radcal=True)
        template_list = eispac.match_templates(data_cube)

        # Find matching template
        template_index = None
        for i, path in enumerate(template_list):
            if path.name == target_template:
                template_index = i
                break

        if template_index is None:
            print(f"Template {target_template} not found")
            return None

        # Copy and process template
        shutil.copy(template_list[template_index], cutouts_path)
        template_filepath = os.path.join(cutouts_path, target_template)

        tmplt = eispac.read_template(template_filepath)
        data_cube = eispac.read_cube(data_file, tmplt.central_wave, apply_radcal=True)
        fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu='max')

        inten_map = fit_res.get_map(component=0, measurement='intensity')
        inten_map = eis_calib_2023(inten_map)

        vel_map = fit_res.get_map(component=0, measurement='velocity')

        # Create visualization
        create_eis_visualization(inten_map, vel_map, eis_line, target_template, cutouts_path, date)

        return inten_map

    except Exception as e:
        print(f"Error processing EIS line {eis_line}: {e}")
        return None

def create_eis_visualization(inten_map, vel_map, eis_line, template, cutouts_path, date):
    """Create EIS intensity and velocity visualization"""
    ny, nx = inten_map.data.shape
    aspect = ny / nx
    panel_width = 4

    fig_width = panel_width * 2.4
    fig_height = panel_width * aspect
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.25)

    # Intensity map
    ax1 = fig.add_subplot(gs[0], projection=inten_map)
    cmap_idx = TARGET_TEMPLATES.index(template)
    im = inten_map.plot(axes=ax1, cmap=EIS_CMAPS[cmap_idx])

    cbar = fig.colorbar(im, extend='both', fraction=0.05, pad=0.05)
    cbar.ax.set_ylabel(r'[erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')

    # Set title based on template
    if 'fe_12' in template:
        ax1.set_title(f'Fe XII {eis_line}', size=10)
    elif 'fe_09' in template:
        ax1.set_title(f'Fe IX {eis_line}', size=10)
    else:
        ax1.set_title(f'Fe XV {eis_line}', size=10)
    ax1.grid(False)

    # Velocity map
    vel_map.plot_settings["norm"] = None
    ax2 = fig.add_subplot(gs[1], projection=vel_map)
    im = vel_map.plot(axes=ax2, vmax=10, vmin=-10, cmap='coolwarm')

    cbar = fig.colorbar(im, extend='both', fraction=0.046, pad=0.05)
    cbar.ax.set_ylabel(r'[kms$^{-1}$]')
    ax2.set_title('Core shift', size=10)
    ax2.grid(False)

    plt.tight_layout()

    # Save figure
    os.chdir(cutouts_path)
    output_dir = './pipeline_figs'
    save_figure(fig, output_dir, f"{date}_EIS_{eis_line}.png")

def process_eis_data(cutouts_data_path, date_begin_eis, date):
    """Process EIS spectroscopic data"""
    downloaded_data_h5 = glob.glob(f"{cutouts_data_path}/*.data.h5")
    downloaded_head_h5 = glob.glob(f"{cutouts_data_path}/*.head.h5")

    if not downloaded_data_h5 or not downloaded_head_h5:
        print(f"No .data.h5 or .head.h5 files found in {cutouts_data_path}")
        return [], []

    eis_list_data = []
    data_intensity_maps = []

    for target_template, eis_line in zip(TARGET_TEMPLATES, EIS_LINES):
        line_available = wavelength_in_cube(downloaded_data_h5[0], str(eis_line))
        eis_list_data.append(line_available)

        if line_available:
            intensity_map = process_single_eis_line(
                downloaded_data_h5[0], eis_line, target_template, cutouts_data_path, date
            )
            if intensity_map is not None:
                data_intensity_maps.append(intensity_map.data)
        else:
            print(f"Skipping wavelength {eis_line} Å: Not found in data")

    return eis_list_data, data_intensity_maps

def create_eis_histograms(eis_list_data, data_intensity_maps, date_begin_eis, cutouts_data_path):
    """Create EIS intensity histograms"""
    n_panels = sum(eis_list_data)
    if n_panels == 0:
        print("No EIS data available for histograms")
        return

    fig, ax = plt.subplots(nrows=1, ncols=n_panels, figsize=(12, 4))
    if n_panels == 1:
        ax = [ax]  # Make it iterable for single panel

    panel_idx = 0
    for i, (eis_line, intensity_map) in enumerate(zip(EIS_LINES, data_intensity_maps)):
        if eis_list_data[i]:
            intensity_data = intensity_map.ravel()
            min_int = max(intensity_data.min(), 1e-1)
            max_int = intensity_data.max()
            bins = np.logspace(np.log10(min_int), np.log10(max_int), 60)

            # Main histogram
            ax[panel_idx].hist(intensity_data, bins=bins, histtype='step',
                             weights=np.ones(len(intensity_data)) / len(intensity_data),
                             color='blue')
            ax[panel_idx].set_xscale('log')
            ax[panel_idx].set_xlabel(r'Intensity [erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')
            ax[panel_idx].set_ylabel('Fraction')
            ax[panel_idx].set_title(f'EIS {eis_line} {date_begin_eis}', fontsize=10)
            ax[panel_idx].axvline(x=np.mean(intensity_data), color='red', lw=1, ls='--',
                                label=f'$\\mu_{{\\mathrm{{EIS}}}}$: {np.mean(intensity_data):.2f}')
            ax[panel_idx].legend(loc='best', fontsize='small')

            # Twin axis for ECDF
            ax_twin = ax[panel_idx].twinx()
            ax_twin.hist(intensity_data, bins=bins, histtype='step', cumulative=True,
                       weights=np.ones(len(intensity_data)) / len(intensity_data),
                       color='green', ls='-')
            ax_twin.axhline(y=0.25, linestyle='dashed', color='green')
            ax_twin.axhline(y=0.75, linestyle='dashed', color='green')
            ax_twin.set_ylabel('ECDF', color='green')
            ax_twin.tick_params(axis='y', labelcolor='green')

            panel_idx += 1

    plt.tight_layout()

    # Save figure
    os.chdir(cutouts_data_path)
    output_dir = './pipeline_figs'
    save_figure(fig, output_dir, f"{date_begin_eis}_EIS_intensity_histograms.png")

# ============================================================================
# HMI DATA PROCESSING FUNCTIONS
# ============================================================================

def process_hmi_data(cutouts_data_path, aia_data, date):
    """Process HMI magnetogram data"""
    print('Now plotting the HMI LOS flux density cutouts for the same date and region...')

    # Get first AIA map for coordinate reference
    first_aia_map = list(aia_data.values())[0]

    # Define cutout region
    bottom_left = SkyCoord(
        first_aia_map.meta['extent_arcsec'][0]*u.arcsec,
        first_aia_map.meta['extent_arcsec'][2]*u.arcsec,
        obstime=first_aia_map.meta['mod_index']['date_obs'],
        observer="earth",
        frame="helioprojective"
    )
    top_right = SkyCoord(
        first_aia_map.meta['extent_arcsec'][1]*u.arcsec,
        first_aia_map.meta['extent_arcsec'][3]*u.arcsec,
        obstime=first_aia_map.meta['mod_index']['date_obs'],
        observer="earth",
        frame="helioprojective"
    )

    # Load and process HMI data
    hmi_map = sunpy.map.Map(cutouts_data_path+'/*magnetogram.fits')
    aia_map_fdisk = sunpy.map.Map(cutouts_data_path+'/*.193.image_lev1.fits')
    out_hmi = hmi_map.reproject_to(aia_map_fdisk.wcs)
    cutout_hmi_aligned = out_hmi.submap(bottom_left, top_right=top_right)

    # Calculate figure dimensions
    ny, nx = cutout_hmi_aligned.data.shape
    aspect = ny / nx
    panel_width = 4
    fig_width = panel_width * 2
    fig_height = aspect * fig_width * 0.9

    # Create figure
    cutout_hmi_aligned.plot_settings["norm"] = None
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Plot signed B_LOS
    ax = fig.add_subplot(2, 2, 1, projection=cutout_hmi_aligned.wcs)
    vmax_val = 0.3 * np.max(cutout_hmi_aligned.data)
    im = cutout_hmi_aligned.plot(axes=ax, vmax=vmax_val, vmin=-vmax_val)
    ax.grid(False)
    add_colorbar_and_styling(fig, ax, im, r'B$_{\mathrm{LOS}}$[G]')

    # Plot |B_LOS|
    ax = fig.add_subplot(2, 2, 2, projection=cutout_hmi_aligned.wcs)
    abs_data = np.abs(cutout_hmi_aligned.data)
    abs_map = sunpy.map.Map(abs_data, cutout_hmi_aligned.meta)
    im = abs_map.plot(axes=ax, clip_interval=(1, 99.99)*u.percent, cmap='YlGnBu')
    ax.grid(False)
    add_colorbar_and_styling(fig, ax, im, r'|B$_{\mathrm{LOS}}$| [G]')

    # Histogram of signed B_LOS
    ax = fig.add_subplot(2, 2, 3)
    bins = np.linspace(cutout_hmi_aligned.data.min(), cutout_hmi_aligned.data.max(), 100)
    data_hmi = cutout_hmi_aligned.data
    ax.hist(data_hmi.ravel(), bins=bins, histtype='step',
           weights=np.ones(len(data_hmi.ravel()))/len(data_hmi.ravel()),
           log=True, color='blue')
    ax.set_ylabel('Fraction')
    ax.set_xlabel(r'B$_{\mathrm{LOS}}$[G]')
    ax.axvline(x=0, ls='-', color='black')

    # Histogram of |B_LOS|
    ax = fig.add_subplot(2, 2, 4)
    bins = np.linspace(abs_data.min(), abs_data.max(), 100)
    ax.hist(abs_data.ravel(), bins=bins, histtype='step',
           weights=np.ones(len(abs_data.ravel()))/len(abs_data.ravel()),
           log=True, color='blue')
    ax.set_xlabel(r'|B$_{\mathrm{LOS}}$| [G]')
    ax.axvline(x=np.mean(abs_data), ls='--', color='black',
              label=f'$\\mu$: {np.mean(abs_data):.2f} [G]')
    ax.legend(loc='best')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.2)

    # Save figure
    os.chdir(cutouts_data_path)
    output_dir = './pipeline_figs'
    save_figure(fig, output_dir, f"{date}_HMI.png")

    # Plot full disk AIA with cutout region
    exp_time_fdisk = aia_map_fdisk.exposure_time
    data_fdisk = aia_map_fdisk / exp_time_fdisk
    fig = plt.figure()
    ax = fig.add_subplot(projection=data_fdisk)
    im = data_fdisk.plot(axes=ax, vmin=0)
    ax.grid(False)
    add_colorbar_and_styling(fig, ax, im, 'DN/s')
    cutout_hmi_aligned.draw_extent(axes=ax)

    # Save full disk figure
    save_figure(fig, output_dir, f"{date}_AIA_fdisk.png")

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_date(date, obs_dates, vdem_file, mhd_code, snapname):
    """Process a single observation date"""
    try:
        date_obj = Time(date, format='isot', scale='utc')
        print(f'Processing Date Sensitive AIA DN Response from {date_obj.strftime("%b %y")}')

        # Get AIA response and synthesize
        aia_resp = get_response(
            date=date_obj,
            channels=AIA_CHANNELS,
            save_response=True,
            units='DN'
        )
        muse_aia = aia_synthesis(aia_resp, WORK_DIR, vdem_file, swap_dims=False)

        # Create synthesis histograms
        print('Creating histograms from the synthesized AIA data')
        create_aia_histogram(muse_aia, mhd_code, snapname)

        # Process cutout files
        date_begin_eis = obs_dates.loc[date]["date_begin_EIS"]
        cutouts_data_path = os.path.join(EIS_DATA_PATH, f"SDO_EIS_cutouts_{date_begin_eis}")
        cutout_files = sorted(glob.glob(os.path.join(cutouts_data_path, 'cutout_*.fits')))

        if not cutout_files:
            print(f"No cutout files found in {cutouts_data_path}")
            return

        print('The cutout files are:\n' + '\n'.join(cutout_files))

        # Process AIA cutouts
        aia_data, aia_fig = process_aia_cutouts(cutout_files, date)

        # Save AIA cutouts
        os.chdir(cutouts_data_path)
        output_dir = './pipeline_figs'
        save_figure(aia_fig, output_dir, f"{date}_cutouts.png")

        # Create comparison histogram
        comparison_fig = create_comparison_histogram(
            aia_data, muse_aia, mhd_code, snapname, date
        )
        save_figure(comparison_fig, output_dir, f"{date}_{mhd_code}_{snapname}_comparison_hist.png")

        # Process EIS data
        print(f'Processing EIS data for date: {date_begin_eis}')
        eis_list_data, data_intensity_maps = process_eis_data(cutouts_data_path, date_begin_eis, date)

        # Create EIS histograms if data is available
        if any(eis_list_data) and data_intensity_maps:
            create_eis_histograms(eis_list_data, data_intensity_maps, date_begin_eis, cutouts_data_path)

        # Process HMI data
        process_hmi_data(cutouts_data_path, aia_data, date)

    except Exception as e:
        print(f"Error processing date {date}: {e}")

def main():
    """Main execution function"""
    # Set up environment
    setup_environment()

    # Get user inputs
    vdem_file = select_vdem_file()
    txt_files = get_text_files()

    mhd_code, snapname = [
        s.strip() for s in
        input("Enter the MHD code and snapname, separated by a comma: ").split(",")
    ]

    # Ask user about processing method
    use_parallel = input("Use parallel processing? (y/n, default=y): ").lower()
    use_parallel = use_parallel != 'n'  # Default to True unless explicitly 'n'

    n_processes = None
    if use_parallel:
        try:
            n_proc_input = input(f"Number of processes (default={max(1, cpu_count()-1)}, max={cpu_count()}): ")
            if n_proc_input.strip():
                n_processes = min(max(1, int(n_proc_input)), cpu_count())
        except ValueError:
            print("Invalid input, using default number of processes")

    # Process each text file
    all_results = []
    for txt_file in txt_files:
        obs_dates = ascii.read(txt_file)
        obs_dates.add_index("date_begin_EIS")
        print(f"\nProcessing observation dates from file: {txt_file}")

        # Convert obs_dates to dictionary for multiprocessing
        obs_dates_dict = {}
        for row in obs_dates:
            date_key = row["date_begin_EIS"]
            obs_dates_dict[date_key] = dict(row)

        dates_list = list(obs_dates['date_begin_EIS'])

        # Process dates
        if use_parallel and len(dates_list) > 1:
            print(f"Processing {len(dates_list)} dates in parallel...")
            results = process_dates_parallel(dates_list, obs_dates_dict, vdem_file,
                                           mhd_code, snapname, n_processes)
        else:
            print(f"Processing {len(dates_list)} dates sequentially...")
            results = process_dates_sequential(dates_list, obs_dates_dict, vdem_file,
                                             mhd_code, snapname)

        all_results.extend(results)

    # Final summary
    total_successful = sum(1 for r in all_results if r.startswith("Success"))
    total_failed = sum(1 for r in all_results if r.startswith("Failed"))

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"Total dates processed: {len(all_results)}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_successful/len(all_results)*100:.1f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    __spec__ = None
    main()
