import os
import glob
def find_response(obs_date,
                  resp_dir = None,
                  delta_month = 12,
                  units = 'DN',
                  verbose = True):
    '''
    Looks for response function closest in time to obs_date, returns file name if less than delta_months avay else None.

    Parameters:
    -------------------------
    obs_date: str, format YYYY-MM-DDTH:M:S, eg 2015-07-05T05:16:15
              the obs date will be output as an astropy time object.
    resp_dir: string, optional, directory where response functions
              are stored, default is None.
    delta_month: int, optional max number of months away from obs_date before
              new response function is suggested, default 12 months.
    units: str, optional, response function intensity units, default DN.
    verbose: bool, optional, be verbose, default True
    '''
    from astropy.time import Time
    from datetime import datetime
    import numpy as np
    if resp_dir is None:
        if 'RESPONSE' not in os.environ:
            raise EnvironmentError("The environment variable 'RESPONSE' is not set. Set it to the directory where response functions are stored.")
        resp_dir = os.environ['RESPONSE']

    if obs_date is None:
        resp_file = os.path.join(resp_dir, f'aia_resp_{units}.zarr')
        if os.path.exists(resp_file):
            zarr_file = resp_file
        else:
            zarr_file = None
    else:
        obs_date = Time(obs_date,format='isot',scale='utc')
        comp = int(obs_date.datetime.strftime("%y"))*12+int(obs_date.datetime.strftime("%m"))
        resp_files = glob.glob(os.path.join(resp_dir, f'aia_resp_{units}_*.zarr'))
        avail = []
        zarr_file = None
        for f in resp_files:
            head = f.split('.')[0]
            rdate = datetime.strptime(head.split('_')[-1],"%b%y")
            avail.append(int(rdate.strftime("%y"))*12+int(rdate.strftime("%m")))
        if len(avail) > 0:
            iresp = np.argmin(np.abs(np.array(avail)-comp))
            if verbose:
                print(f'*** Nearest response function is {np.abs(avail[iresp]-comp)} months from obs_date ')
            if np.abs(avail[iresp]-comp) <= delta_month:
                zarr_file = resp_files[iresp]
    return zarr_file, obs_date

# **************************************************

def get_response(date = None,
                 save_response = False,
                 units = 'DN',
                 lgtgmax=7.0,lgtgmin=4.4, lgtgstep=0.1,
                 uzmax = 200., uzmin = -200., uzstep = 50.,
                 abund = "sun_photospheric_2021_asplund",
                 press = 3e15,
                 dx_pix=0.6, dy_pix=0.6,
                 channels = [94, 131, 171, 193, 211, 304, 335],
                 resp_dir = None,
                 delta_month = 12,
                 ):

    '''
    Looks for and reads or computes response function closest in time to obs_date, returns response function

    Parameters:
    -------------------------
    date:     str, format YYYY-MM-DDTH:M:S, eg 2015-07-05T05:16:15
    units:    str, units of response, currently only 'DN' is available
    lgtgmax, lgtgmin, lgtgstep: float, temperature span and delta
    abund:    str, abundance file to use, default "sun_coronal_2021_chianti"
    press:    float, pressure, default 3e15
    dx_pix, dy_pix: float, size of instrument pixels in arcsec, default 0.6.
    bands:    list, integers, bands to calculate default all AIA bands
    resp_dir: string, optional, directory where response functions
              are stored, default is RESPONSE environment variable.
    delta_month: int, optional max number of months away from obs_date before
              new response function is suggested, default 12 months.
    units: str, optional, response function intensity units, default DN.
    verbose: bool, optional, be verbose, default True
    '''
    if resp_dir is None:
        if 'RESPONSE' not in os.environ:
            raise EnvironmentError("The environment variable 'RESPONSE' is not set. Set it to the directory where response functions are stored.")
        resp_dir = os.environ['RESPONSE']
    import aiapy
    import numpy as np
    import xarray as xr
    import astropy.constants as const
    import astropy.units as u
    from aiapy.response import Channel
    from muse.utils.utils import read_response
    from muse.synthesis.synthesis import transform_resp_units
    from muse.instr.utils import create_eff_area_xarray
    from muse.instr.utils import chianti_gofnt_linelist
    from muse.instr.utils import create_resp_func, create_resp_line_list, create_resp_func_ci
    from muse.synthesis.synthesis import transform_resp_units
    from muse.instr.utils import convert_resp2muse_ciresp
#  Temperature limits, abundance, pressure, and pixel size
#  NB note that available abundance files depend on Chianti version!
#  Other possible abundance files to look for...
#  abund = "sun_photospheric_2011_caffau"
#  abund = "sun_photospheric_2021_asplund"
#    if use_QS_bands:
#        aia_goes_lines.remove('AIA 304')
#        lines = aia_goes_lines[0:6]
#        bands = [int(s) for s in " ".join(lines).split("AIA ")[1:7]]
# bands = list(map(int," ".join(lines.split("AIA ")[1:5]))
#    else:
#        print("*** You should set use_QS_bands to true unless you know what you are doing!!!")
#        sys.exit()
#
    zarr_file,obs_date = find_response(date, units = units)
    if zarr_file is not None:
        print(f'*** {zarr_file} already exists! Reading...') #But it may happen that the same bands are not requested.
    # it is not quite clear how to find the number of gains asked for... should be equal to the number of
    # lines/bands that the response function was constructed with
        response_all = xr.open_zarr(zarr_file).compute() #read_response(zarr_file).compute()
        if  np.array_equal(channels, response_all.channel):
            print("The channels of the response function match.")
            return response_all ##this is fine, no need to create a new response function
        else:
           print("The channels of the response function do not match the requested channels. Creating a new response function.")
           need_new_response = True ##Treating this as a flag to create a new response function


    else:
        need_new_response = True
    if need_new_response:
        logT = xr.DataArray(np.arange(lgtgmin,lgtgmax, lgtgstep),dims='logT')
        vdop = np.arange(uzmin, uzmax, uzstep) * u.km / u.s
        pressure = xr.DataArray(np.array([press]), dims= 'pressure')
        print(f"*** Constructing line list")
        line_list = chianti_gofnt_linelist(temperature = 10**logT,
                                           pressure=pressure,
                                           abundance = abund,
                                           wavelength_range = [80,850],
                                           )
        for band in channels:
            ch = Channel(band*u.angstrom)
            if obs_date is None:
                #print(f'*** Computing {units} response function for {ch.channel.to_string()}')
                #response = ch.wavelength_response() * ch.plate_scale
                print(f'*** Computing {units} response function for {ch.channel.to_string()}')
                try:
                    response = ch.wavelength_response() * ch.plate_scale
                except:
                    print('*** Warning: correction table taken from local JSOC installation')
                    response = ch.wavelength_response(correction_table = aiapy.calibrate.util.get_correction_table('JSOC'))
            else:
                print(f'*** Computing {units} response function for {ch.channel.to_string()}'
                       ' date {obs_date.strftime("%b%Y")}')
                try:
                    response = ch.wavelength_response(obstime = obs_date, correction_table = aiapy.calibrate.util.get_correction_table('JSOC'))
                except:
                    print('*** Warning: correction table taken from local JSOC installation')
                    response = ch.wavelength_response(obstime = obs_date, correction_table = aiapy.calibrate.util.get_correction_table('JSOC'))
                response = ch.wavelength_response(obstime = obs_date, correction_table = aiapy.calibrate.util.get_correction_table('JSOC'))
            # else:
            # print(f'*** Computing {units} response function for {ch.channel.to_string()} date {obs_date.strftime("%b%Y")}')
            # response = ch.wavelength_response(obstime = obs_date) * ch.plate_scale
            eff_xr = create_eff_area_xarray(response.value, ch.wavelength.value, [ch.channel.value])
            area = eff_xr.eff_area.interp(wavelength=line_list.wvl).fillna(0).drop_vars('wavelength')
            line_list["resp_func"] = line_list.gofnt.sum(['logT']) * area.isel(band=0)
            line_list = line_list.drop_vars('band')
            sort_index = np.argsort(-line_list.resp_func,
                        axis=line_list.resp_func.get_axis_num('trans_index'))
            line_list_sort = line_list[dict(trans_index=sort_index)]
            line_list_sort_c = line_list_sort.isel(trans_index=np.arange(1000))
            ''' Important, considering here 1000 lines!!!!!!!
                this creates the resposne function. Note that now we provide pressure
                (it can also be an array) or not sum lines, but
                if you have many it becomes a huge array!
            '''
            n = line_list_sort_c.sizes['trans_index']
            resp = create_resp_func(
                line_list_sort_c,
                vdop=vdop,
                instr_width=0,
                effective_area=eff_xr.eff_area,
                wvlr=[80, 800],
                num_lines_keep=0,
                )
            resp_dn = transform_resp_units(resp,
                                           new_units="1e-27 cm5 DN / (Angstrom s)",
                                           wvl=np.array(resp.wavelength.data),
                                           dx_pix=dx_pix, dy_pix=dy_pix,
                                           gain =18,
                                           )
            ci_resp = convert_resp2muse_ciresp(resp_dn)
            line_list = line_list.drop_vars("resp_func")
            ci_resp = ci_resp.drop_vars("band")
            if band == channels[0]:
                response_all = ci_resp
            else:
                response_all = xr.concat([response_all, ci_resp], dim="channel")
        response_all["SG_resp"] = response_all.SG_resp.fillna(0)
        response_all = response_all.assign_coords(channel = ("channel", channels))
        if "band" in response_all.SG_resp.dims:
            response_all["SG_resp"] = response_all["SG_resp"].squeeze("band", drop=True)
            if "band" in response_all.dims:
                response_all = response_all.drop_dims("band")
        response_all = response_all.compute()
        save_response = True
#
#    response_all = response_all.assign_coords(line = ("band",['AIA '+f'{int(s)}' for s in response_all.band.data]))
    if obs_date is None:
        response_all = response_all.assign_attrs(date = "None")
    else:
        response_all = response_all.assign_attrs(date = obs_date.strftime("%d-%b-%Y"))
#
    response_all = response_all.compute()
#
    if obs_date is None:
        zarr_file = f'aia_resp_{units}.zarr'
    else:
        zarr_file = f'aia_resp_{units}_{obs_date.strftime("%b%y")}.zarr'
    zarr_file = os.path.join(os.environ['RESPONSE'],zarr_file)
    if save_response:
        try:
            response_all.to_zarr(f'{zarr_file}', mode = "w")
            print(f"Saved response to {f'{zarr_file}'}")
        except:
            print(f"*** Error: Could not save zarr file {f'{zarr_file}'}. Using NetCDF.")
            response_all.to_netcdf(f'{zarr_file}.nc', mode = "w")
            print(f"Saved response to {f'{zarr_file}.nc'}")
    return response_all

# **************************************************

def aia_synthesis(aia_resp, work_dir, vdem_path, swap_dims = True):
    import xarray as xr
    from muse.synthesis.synthesis import vdem_synthesis
    import os
    import glob
    print(f"*** Work directory is {work_dir}")
    os.chdir(work_dir)

    files = vdem_path #glob.glob(os.path.join(vdem_dir,'*'))
    print(f'*** Loading {files} into vdem')
    vdem = xr.open_zarr(files).compute()

    # vdem_cut
    vdem_cut = vdem.sel(logT=aia_resp.logT, method = "nearest")
    vdem_cut = vdem_cut.compute()
    #Synthesis AIA observations using the response function and VDEM
    muse_AIA = vdem_synthesis(vdem_cut.sum(dim=["vdop"]),
                              aia_resp,
                              sum_over=["logT"])
    if swap_dims:
       muse_AIA = muse_AIA.swap_dims({"band":"line"}) # Needed in the below?
    return muse_AIA


# **************************************************

def readFits(filename, ext=0):
  from astropy.io import fits
  import numpy as np
  """
  Just defining a simple readFits function without
  bothering about the complex header options of astropy fits.
  """
  io = fits.open(filename, 'readonly',memmap=True)
  #print('reading -> {0}'.format(filename))
  dat = np.ascontiguousarray(io[ext].data, dtype='float32')
  io.close()

  return dat
# **************************************************

def save_eis_iris_dates(urls, output_file, alternate_only=False):
    """
    Downloads JSON data from multiple LMSAL HEK URLs,
    extracts start/stop times, and saves them in:
    YYYY-MM-DDTHH:MM:SS - YYYY-MM-DDTHH:MM:SS        ''
    in a text file.
    Parameters
    ----------
    urls : list of str
        List of JSON URLs to fetch.
    output_file : str
        Path to output text file.
    """
    import requests
    from datetime import datetime
    all_lines = set()

    for url in urls:
        print(f"Fetching: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            print(f"Number of events in JSON: {len(data.get('Events', []))}")
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        for event in data.get("Events", []):
            instruments = event.get("instrument", [])
            if "IRIS" not in instruments:
                continue
            start_time = event.get("startTime")
            stop_time = event.get("stopTime")
            if start_time and stop_time:
                start_fmt = start_time.replace(" ", "T")
                stop_fmt = stop_time.replace(" ", "T")
                all_lines.add(f"{start_fmt} - {stop_fmt}        ''")
    # Sort by start date
    all_lines = list(all_lines)
    all_lines = sorted(list(all_lines), key=lambda line: datetime.strptime(line.split(" - ")[0], "%Y-%m-%dT%H:%M:%S"))

    # alternate_only= True
    if alternate_only:
        all_lines = all_lines[::2]
    # Save to file
    with open(output_file, "w") as f:
        f.write("date_begin_EIS  -  date_end_EIS                   Comment\n")
        for line in all_lines:
            f.write(line + "\n")

    print(f"Saved {len(all_lines)} date ranges to {output_file}")

# **************************************************

def wavelength_in_cube(data_file, target_wave_str):
    import eispac
    """
    Check if the target wavelength (as string) is present in the EIS data file.

    Parameters:
        data_file (str): Path to the EIS data file.
        target_wave_str (str): Target wavelength as string, e.g., '195.120'.

    Returns:
        bool: True if the wavelength is found in any line_id, False otherwise.
    """
    try:
        wininfo = eispac.read_wininfo(data_file)
        for wvl_min, wvl_max in (zip(wininfo.wvl_min, wininfo.wvl_max)):
            if wvl_min <= float(target_wave_str) <= wvl_max:
                return True
        return False
        # available_lines = [win.line_id for win in wininfo]
        # return any(target_wave_str in line for line in available_lines)
    except Exception as e:
        print(f"Error checking wavelengths in {data_file}: {e}")
        return False

# **************************************************

def save_hmi_c_outs(magnetogram_path, output_dir, eis_data_list):
    """
    Saves HMI cutouts corresponding to the EIS data.

    Parameters:
    -----------
    magnetogram_path : str
        Path to the directory containing HMI magnetogram data files.
    output_dir : str
        Directory where the output cutouts will be saved.
    eis_data_list : list
        List of EIS data arrays for which corresponding HMI cutouts are to be saved.
        Need to use eispac.read_cube(downloaded_data_h5[0])
    """
    import os
    from glob import glob
    if not eis_data_list:
        print("No EIS data found for:", magnetogram_path)
        return
    # Find the first matching magnetogram and AIA file
    mag_files = glob(os.path.join(magnetogram_path, '*magnetogram.fits'))
    aia_files = glob(os.path.join(magnetogram_path, '*.193.image_lev1.fits'))
    if not mag_files or not aia_files:
        print(f"Missing HMI or AIA files in {magnetogram_path}")
        return
    hmi_map = sunpy.map.Map(mag_files[0])
    aia_map_fdisk = sunpy.map.Map(aia_files[0])
    out_hmi = hmi_map.reproject_to(aia_map_fdisk.wcs)
    for eis_data in eis_data_list:
        try:
            meta = eis_data.meta
            bottom_left = SkyCoord(meta['extent_arcsec'][0]*u.arcsec, meta['extent_arcsec'][2]*u.arcsec, obstime=meta['mod_index']['date_obs'], observer="earth", frame="helioprojective")
            top_right = SkyCoord(meta['extent_arcsec'][1]*u.arcsec, meta['extent_arcsec'][3]*u.arcsec, obstime=meta['mod_index']['date_obs'], observer="earth", frame="helioprojective")
            cutout_hmi_aligned = out_hmi.submap(bottom_left, top_right=top_right)
            output_file = os.path.join(output_dir, f"HMI_Cutout_{meta['mod_index']['date_obs']}.fits")
            cutout_hmi_aligned.save(output_file)
            print(f"Saved HMI cutout to {output_file}")
        except Exception as e:
            print(f"Error processing EIS data: {e}")

# **************************************************
