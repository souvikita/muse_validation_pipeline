import warnings
import matplotlib.pyplot as plt
import numpy as np
import pooch
import astropy.units as u
from astropy import constants
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.modeling import models as m
from astropy.modeling.fitting import LevMarLSQFitter, parallel_fit_dask
from astropy.visualization import time_support
import os
from sunpy.coordinates import frames
from irispy.io import read_files
import glob
from tqdm import tqdm
from dask.distributed import Client
import json
import astropy.io.ascii as ascii
import re
import requests
from irispy.spectrograph import SpectrogramCube


def download_hek_json(url_mod, obs_data_dir):
    """
    Downloads HEK JSON file from the given URL.
    
    Parameters:
    -----------
    url_mod : str
        URL to download the HEK JSON file from
    obs_data_dir : str
        Directory where the JSON file will be saved
        
    Returns:
    --------
    str
        Path to the downloaded JSON file
    """
    os.makedirs(obs_data_dir, exist_ok=True)
    filename = url_mod.split("/")[-1]
    filepath = os.path.join(obs_data_dir, filename)

    response = requests.get(url_mod, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filepath


def find_iris_raster_url(data, min_exptime=4):
    """
    Parses HEK JSON data to find IRIS raster URL.
    
    Parameters:
    -----------
    data : dict
        JSON data loaded from HEK response
    min_exptime : float, optional
        Minimum exposure time in seconds (default: 4)
        
    Returns:
    --------
    str or None
        URL to the IRIS raster file, or None if not found
    """
    target_url = None
    for event in data.get("Events", []):
        if event.get("instrument") != "IRIS":
            continue
        goal = event.get("goal")
        xx = re.search('sit-and-stare', goal, re.IGNORECASE)
        if xx:
            print("\nThis is a sit-and-stare observation. Abort downloading the data.")
            continue
        for group in event.get("groups", []):
            name_group = group.get("group_name", "").lower()
            if 'raster' in name_group:
                exptime = group.get("max_exptime")
                if exptime >= min_exptime:
                    url = group.get("comp_data_url")
                    print(f"\n*** Maximum exposure time for this IRIS observation: {exptime} s")
                    if url and url.endswith("_raster.tar.gz"):
                        target_url = url
    return target_url


def download_iris_raster(target_url, obs_data_dir):
    """
    Downloads IRIS raster file using pooch.
    
    Parameters:
    -----------
    target_url : str
        URL to the raster file
    obs_data_dir : str
        Directory where the raster file will be saved
        
    Returns:
    --------
    str
        Path to the downloaded raster file
    """
    raster_filename = os.path.join(obs_data_dir, os.path.basename(target_url))
    if os.path.exists(raster_filename):
        print(f"*** IRIS raster data already downloaded: {raster_filename}")
        return raster_filename
    else:
        raster_filename = pooch.retrieve(target_url, known_hash=None, path=obs_data_dir, progressbar=True)
        return raster_filename


def fit_iris_si_iv(raster, client):
    """
    Performs full fitting analysis on IRIS Si IV 1403.
    
    Parameters:
    -----------
    raster : dict
        IRIS raster data loaded from read_files
    client : dask.distributed.Client
        Dask client for parallel processing
        
    Returns:
    --------
    tuple
        (iris_model_fit, si_iv_1403) - Fitted model and Si IV 1403
    """
    raster_keys = list(raster.keys())
    if "Si IV 1403" in raster_keys:
        si_iv_1403 = raster["Si IV 1403"][-1]  # last raster in the series
        si_iv_core = 140.277 * u.nm
        normalized_unit = si_iv_1403.unit / u.s
        exposure_time_reshaped = si_iv_1403.exposure_time.value[:, np.newaxis, np.newaxis]
        clipped_data = np.maximum(si_iv_1403.data,0) #Filtering out non-positive values
        normalized_data = clipped_data/ exposure_time_reshaped
        normalized_data = np.where(np.isnan(normalized_data), 0, normalized_data) # can be nan because of the zero exposure time
        normalized_si_iv_spec = SpectrogramCube(
            normalized_data,
            si_iv_1403.wcs,
            meta=si_iv_1403.meta,
            unit=normalized_unit,
            uncertainty=si_iv_1403.uncertainty,
            copy=True,
        )
        wl_sum = normalized_si_iv_spec.rebin((1, 1, normalized_si_iv_spec.data.shape[-1]), operation=np.sum)[0]
        spatial_mean = normalized_si_iv_spec.rebin((*normalized_si_iv_spec.data.shape[:-1], 1))[0, 0, :]
        initial_model = m.Const1D(amplitude=1/si_iv_1403.exposure_time.value[0] * normalized_si_iv_spec.unit) + m.Gaussian1D(
            amplitude=np.nanmax(spatial_mean.data) * normalized_si_iv_spec.unit, mean=si_iv_core, stddev=0.005 * u.nm
        )

        fitter = LevMarLSQFitter()
        average_fit = fitter(
            initial_model,
            spatial_mean.axis_world_coords("em.wl")[0].to(u.nm),
            spatial_mean.data * spatial_mean.unit,
            filter_non_finite = True,  # Allow fitting with non-finite values
        )
        # We want to do some basic data sanitization.
        # Remove negative values and set them to zero and remove non-finite values.
        filtered_data = np.where(normalized_si_iv_spec.data < 0, 0, normalized_si_iv_spec.data)
        filtered_data = np.where(np.isfinite(filtered_data), filtered_data, 0)
        # We can therefore fit the cube
        with warnings.catch_warnings():
            # There are several WCS warnings we just want to ignore
            warnings.simplefilter("ignore")
            iris_model_fit = parallel_fit_dask(
                data=filtered_data,
                data_unit=normalized_si_iv_spec.unit,
                fitting_axes=2,
                world=normalized_si_iv_spec.wcs,
                model=average_fit,
                fitter=LevMarLSQFitter(),
                scheduler=client,
            )
        
        return iris_model_fit, si_iv_1403
    else:
        print("No Si IV 1403 data found in the raster. Aborting fitting.")
        return None, None


def save_fit_results(iris_model_fit, si_iv_1403, date_begin, obs_data_dir):
    """
    Saves fit results to NPZ file.
    
    Parameters:
    -----------
    iris_model_fit : astropy.modeling.Model
        Fitted model from parallel_fit_dask
    si_iv_1403 : irispy.Spectrogram
        Si IV 1403 spectrogram data
    date_begin : str
        Beginning date string (used for filename)
    obs_data_dir : str
        Directory where the NPZ file will be saved
    """
    si_iv_core = 140.277 * u.nm
    net_flux = (
        np.sqrt(2 * np.pi)
        * (iris_model_fit.amplitude_0 + iris_model_fit.amplitude_1)
        * iris_model_fit.stddev_1.quantity
        / np.mean(si_iv_1403.axis_world_coords("wl")[0][1:] - si_iv_1403.axis_world_coords("wl")[0][:-1])
    )
    core_shift = ((iris_model_fit.mean_1.quantity.to(u.nm)) - si_iv_core) / si_iv_core * (constants.c.to(u.km / u.s))
    sigma = (iris_model_fit.stddev_1.quantity.to(u.nm)) / si_iv_core * (constants.c.to(u.km / u.s))

    npz_filename = f"iris_fit_results_{date_begin.replace(':','-')}.npz"
    npz_path = os.path.join(obs_data_dir, npz_filename)
    np.savez(
        npz_path,
        net_flux=net_flux.value,
        net_flux_unit=str(net_flux.unit),
        core_shift=core_shift.value,
        core_shift_unit=str(core_shift.unit),
        sigma=sigma.value,
        sigma_unit=str(sigma.unit),
    )
    print(f"\nSaved fit results to {npz_path}")


warnings.filterwarnings("ignore", category=UserWarning, append=True)
if __name__ == "__main__":
    client = Client()
    # QS datasets directory structure
    qs_data_path = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/QS_datasets/IRIS_only/'
    iris_data_file = os.path.join(qs_data_path, 'iris_data.txt')
    
    # Read iris_data.txt with comment handling
    # astropy.io.ascii should auto-detect the format (space-separated with " - " separator)
    obs_dates = ascii.read(iris_data_file, comment='#')
    obs_dates.add_index("date_begin_IRIS")
    
    for date in obs_dates["date_begin_IRIS"]:
        url_base = 'https://www.lmsal.com/hek/hcr?cmd=search-events3&outputformat=json&startTime=2020-11-19T17:00&stopTime=2020-11-20T17:00&hasData=true&hideMostLimbScans=true&limit=200'#'https://www.lmsal.com/hek/hcr?cmd=search-events-corr&outputformat=json&startTime=2025-07-08T00:00&stopTime=2025-07-09T00:00&instrument=IRIS&hasData=true&hideMostLimbScans=true&optionalcorr=SOT&optionalcorr=SOTSP&optionalcorr=XRT&requiredcorr=EIS'
        date_begin_IRIS = obs_dates.loc[date]["date_begin_IRIS"]
        date_end_IRIS = obs_dates.loc[date]["date_end_IRIS"]
        url_mod = url_base.replace('startTime=2020-11-19T17:00', f'startTime={date_begin_IRIS}').replace('stopTime=2020-11-20T17:00', f'stopTime={date_end_IRIS}')
        print(f"\n*** Checking IRIS rasters in the interval {date_begin_IRIS} - {date_end_IRIS}")
        obs_data_dir = os.path.join(qs_data_path, "IRIS_datasets", date_begin_IRIS)
        os.makedirs(obs_data_dir, exist_ok=True)
        
        # Download HEK JSON file
        downloaded_filename = download_hek_json(url_mod, obs_data_dir)
        
        # Parse JSON and find raster URL
        with open(downloaded_filename, "r") as f:
            data = json.load(f)
        
        target_url = find_iris_raster_url(data, min_exptime=4)
        
        if target_url is None:
            print(f"*** No suitable IRIS raster found for {date_begin_IRIS}. Skipping.")
            continue
        
        # Download raster file
        raster_filename = download_iris_raster(target_url, obs_data_dir)
        
        # Read raster and perform fitting
        raster = read_files(raster_filename, memmap=False)
        iris_model_fit, si_iv_1403 = fit_iris_si_iv(raster, client)
        if iris_model_fit is None or si_iv_1403 is None:
            print(f"*** Skipping save for {date_begin_IRIS} - fitting was aborted.")
            continue
        
        # Save fit results
        save_fit_results(iris_model_fit, si_iv_1403, date_begin_IRIS, obs_data_dir)

