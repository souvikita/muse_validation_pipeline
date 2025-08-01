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
from dask.distributed import Client
# import subprocess
import requests

def download_file(url_mod, obs_data_dir):
    # os.makedirs(obs_data_dir, exist_ok=True)
    filename = url_mod.split("/")[-1]
    filepath = os.path.join(obs_data_dir, filename)

    response = requests.get(url_mod, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filepath


warnings.filterwarnings("ignore", category=UserWarning, append=True)
if __name__ == "__main__":
    client = Client()
    eis_data_path = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/' #This is an absolute path. I don't like it :/
    obs_dates = ascii.read(eis_data_path+'Plage_observations_viggo.txt') #Plage_observations_viggo.txt is from Viggo 
    obs_dates.add_index("date_begin_EIS")
    for date in obs_dates["date_begin_EIS"]:
        url_base = 'https://www.lmsal.com/hek/hcr?cmd=search-events-corr&outputformat=json&startTime=2025-07-08T00:00&stopTime=2025-07-09T00:00&instrument=IRIS&hasData=true&hideMostLimbScans=true&optionalcorr=SOT&optionalcorr=SOTSP&optionalcorr=XRT&requiredcorr=EIS'
        date_begin_EIS = obs_dates.loc[date]["date_begin_EIS"]
        date_end_EIS = obs_dates.loc[date]["date_end_EIS"]
        url_mod = url_base.replace('startTime=2025-07-08T00:00', f'startTime={date_begin_EIS}').replace('stopTime=2025-07-09T00:00', f'stopTime={date_end_EIS}')
        print(f"\n*** Checking IRIS rasters in the interval {date_begin_EIS} - {date_end_EIS}")
        obs_data_dir = os.path.join(eis_data_path,"IRIS_datasets",date_begin_EIS)
        os.makedirs(obs_data_dir, exist_ok=True)
        # output = !wget -P "{obs_data_dir}" "{url_mod}"
        downloaded_filename = download_file(url_mod, obs_data_dir)
        
        with open(downloaded_filename, "r") as f:
            data = json.load(f)
        target_url = None
        for event in data.get("Events", []):
            if event.get("instrument") != "IRIS":
                    continue
            goal=event.get("goal")
            xx= re.search('sit-and-stare', goal, re.IGNORECASE)
            if xx:
                print("\nThis is a sit-and-stare observation. Abort downloading the data.")
                continue
            for group in event.get("groups", []):
                name_group = group.get("group_name","").lower()
                if 'raster' in name_group:
                    exptime = group.get("max_exptime")
                    if exptime >= 4:
                        url = group.get("comp_data_url")
                        print(f"\n*** Maximum exposure time for this IRIS observation: {exptime} s")
                # url = group.get("comp_data_url")
                # print(url)
                    if url and url.endswith("_raster.tar.gz"):
                        target_url = url
                        raster_filename = os.path.join(obs_data_dir, os.path.basename(target_url))
                        if os.path.exists(raster_filename):
                            print(f"*** IRIS raster data already downloaded: {raster_filename}")
                        else:
                            raster_filename = pooch.retrieve(target_url, known_hash=None, path=obs_data_dir, progressbar=True)
            raster = read_files(raster_filename, memmap=False)
            si_iv_1403 = raster["Si IV 1403"][-1] # last raster in the series
            si_iv_core = 140.277 * u.nm
            wl_sum = si_iv_1403.rebin((1, 1, si_iv_1403.data.shape[-1]), operation=np.sum)[0]
            spatial_mean = si_iv_1403.rebin((*si_iv_1403.data.shape[:-1], 1))[0, 0, :]
            initial_model = m.Const1D(amplitude=1 * si_iv_1403.unit) + m.Gaussian1D(
                amplitude=np.nanmax(spatial_mean.data) * si_iv_1403.unit, mean=si_iv_core, stddev=0.005 * u.nm
            )

            fitter = LevMarLSQFitter()
            average_fit = fitter(
                initial_model,
                spatial_mean.axis_world_coords("em.wl")[0].to(u.nm),
                spatial_mean.data * spatial_mean.unit,
            )
            # We want to do some basic data sanitization.
            # Remove negative values and set them to zero and remove non-finite values.
            filtered_data = np.where(si_iv_1403.data < 0, 0, si_iv_1403.data)
            filtered_data = np.where(np.isfinite(filtered_data), filtered_data, 0)
            # We can therefore fit the cube
            with warnings.catch_warnings():
                # There are several WCS warnings we just want to ignore
                warnings.simplefilter("ignore")
                iris_model_fit = parallel_fit_dask(
                    data=filtered_data,
                    data_unit=si_iv_1403.unit,
                    fitting_axes=2,
                    world=si_iv_1403.wcs,
                    model=average_fit,
                    fitter=LevMarLSQFitter(),
                    scheduler=client,
                )
            client.close()
            net_flux = (
                np.sqrt(2 * np.pi)
                * (iris_model_fit.amplitude_0 + iris_model_fit.amplitude_1)
                * iris_model_fit.stddev_1.quantity
                / np.mean(si_iv_1403.axis_world_coords("wl")[0][1:] - si_iv_1403.axis_world_coords("wl")[0][:-1])
            )       
            core_shift = ((iris_model_fit.mean_1.quantity.to(u.nm)) - si_iv_core) / si_iv_core * (constants.c.to(u.km / u.s))
            sigma = (iris_model_fit.stddev_1.quantity.to(u.nm)) / si_iv_core * (constants.c.to(u.km / u.s))

            npz_filename = f"iris_fit_results_{date_begin_EIS.replace(':','-')}.npz"
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