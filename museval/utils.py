import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helita.sim import bifrost as br
from muse import logger
import xarray as xr

__all__ = [
    "find_response",
    "get_response",
    "make_vdem",
    "get_vdem_bz",
    "aia_synthesis",
    "readFits",
    "save_eis_iris_dates",
    "wavelength_in_cube",
    "save_hmi_c_outs",
    "pick_sim",
    "plot_aia_overview",
    "iny_smear",
    "gauss_kernel",
]

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
                logger.info(f'*** Nearest response function is {np.abs(avail[iresp]-comp)} months from obs_date ')
            if np.abs(avail[iresp]-comp) <= delta_month:
                zarr_file = resp_files[iresp]
    return zarr_file, obs_date

# **************************************************

def get_response(vdem, date = None, 
                 save_response = False,
                 units = 'DN',
                 lgtgmax=7.5,lgtgmin=4.5, lgtgstep=0.1,
                 uzmax = 500., uzmin = -500., uzstep = 100.,
                 abund =  "sun_photospheric_2021_asplund", # "sun_coronal_2021_chianti",
                 press = 3e15,
                 dx_pix=0.6, dy_pix=0.6,
                 channels = [94, 131, 171, 193, 211, 304, 335],  
                 resp_dir = None,
                 wavelength_range = [80,850],
                 minimum_abundance = 1.e-20, 
                 delta_month = 12,
                 ):
    
    from muse import logger
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
    zarr_file,obs_date = find_response(date, units = units, delta_month = delta_month)
    if zarr_file is not None:
        logger.info(f'*** {zarr_file} already exists! Reading...') #But it may happen that the same bands are not requested.
    # it is not quite clear how to find the number of gains asked for... should be equal to the number of 
    # lines/bands that the response function was constructed with
        ntg = int((lgtgmax-lgtgmin)/lgtgstep) + 1
        lgtaxis = np.linspace(lgtgmin,lgtgmax,ntg)
        logT = lgtaxis #np.arange(lgtgmin,lgtgmax, lgtgstep)
        vdop = np.arange(uzmin, uzmax, uzstep) * u.km / u.s
        response_all = read_response(zarr_file,
                                     logT=vdem.logT, 
                                     vdop=vdem.vdop, vdopmethod="linear",
                                     gain = np.ones((len(channels)))*18).compute()
        if  np.array_equal(channels, response_all.channel):
            logger.info("The channels of the response function match.")
            return response_all ##this is fine, no need to create a new response function
        else:
            logger.info("The channels of the response function do not match the requested channels. Creating a new response function.")
            need_new_response = True ##Treating this as a flag to create a new response function
            

    else:
        need_new_response = True
    if need_new_response:
        logT = xr.DataArray(np.arange(lgtgmin,lgtgmax, lgtgstep),dims='logT')
        vdop = np.arange(uzmin, uzmax, uzstep) * u.km / u.s
        pressure = xr.DataArray(np.array([press]), dims= 'pressure')
        logger.info(f"*** Constructing line list using pressure = {press:0.1e}, abundance {abund}")
        line_list = chianti_gofnt_linelist(temperature = 10**logT,
                                           pressure=pressure,
                                           abundance = abund,
                                           wavelength_range = wavelength_range,
                                           minimum_abundance = minimum_abundance,
                                           ) 
        try:
            correction_table = aiapy.calibrate.util.get_correction_table('JSOC')
            logger.info('*** Correction table taken from local JSOC installation')
        except:
            logger.info('*** Correction table taken from local SSW installation')
            correction_table = aiapy.calibrate.util.get_correction_table('SSW')
        for band in channels:
            ch = Channel(band*u.angstrom)
            if date is None:
                logger.info(f'*** Computing {units} response function for {ch.channel.to_string()}')
            else:
                logger.info(f'*** Computing {units} response function for {ch.channel.to_string()}'
                  f' date {obs_date.strftime("%b%Y")}')
            response = ch.wavelength_response(obstime = obs_date, correction_table = correction_table) 
            eff_xr = create_eff_area_xarray(response.value, ch.wavelength.value, [ch.channel.value])
            area = eff_xr.eff_area.interp(wavelength=line_list.wvl).fillna(0).drop_vars('wavelength')
            line_list["resp_func"] = line_list.gofnt.sum(['logT']) * area.isel(band=0)
            line_list = line_list.drop_vars('band')
            sort_index = np.argsort(-line_list.resp_func, 
                        axis=line_list.resp_func.get_axis_num('trans_index'))
            line_list_sort = line_list[dict(trans_index=sort_index)]
            line_list_sort_c = line_list_sort.isel(trans_index=np.arange(1000))
            ''' Important, considering here 1000 lines!!!!!!! 
                this creates the response function. Note that now we provide pressure 
                (it can also be an array) or not sum lines, but 
                if you have many it becomes a huge array!
            ''' 
            n = line_list_sort_c.sizes['trans_index']
            resp = create_resp_func(
                line_list_sort_c,
                vdop=vdop,
                instr_width=0,  
                effective_area=eff_xr.eff_area,
                wvlr=wavelength_range,
                num_lines_keep=0,
                )
            resp_dn = transform_resp_units(resp,
                                           new_units="1e-27 cm5 DN / (Angstrom s)",
                                           wvl=np.array(resp.wavelength.data),
                                           dx_pix=dx_pix, dy_pix=dy_pix,
                                           gain = 18,
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
            logger.info(f"Saved response to {f'{zarr_file}'}")
        except:
            logger.info(f"*** Error: Could not save zarr file {f'{zarr_file}'}. Using NetCDF.")
            response_all.to_netcdf(f'{zarr_file}.nc', mode = "w")
            logger.info(f"Saved response to {f'{zarr_file}.nc'}")
    response_all = read_response(zarr_file,
                                 logT=vdem.logT, 
                                 vdop=vdem.vdop, vdopmethod="linear",
                                 gain = np.ones((len(channels)))*18).compute()
    return response_all

# **************************************************

def make_vdem(snapname, snap, 
                  code = 'Bifrost',
                  workdir = './',
                  save = False, save_netcdf = False,
                  save_bz = False, z0 = -0.15, # height at which to save Bz [Mm] 
                  compute = True,            # -> roughly equal to formation height of 617.3 nm HMI line
                  zarr_format = 2, opa_wvl = 171,
                  telescope = 'muse',
                  aia_vdop = [-500, 600, 100],
                  muse_vdop = [-200, 210, 10],
                  iris_vdop = [-100, 100, 2],
                  ):
    import numpy as np
    from muse import logger
    import PlasmaCalcs as pc
    from PlasmaCalcs.tools import _xarray_save_prep
    os.chdir(workdir)
    vdem_dir = os.path.join(workdir,"vdem")
    zarr_file = os.path.join(vdem_dir,f"{telescope}_vdem_{snap:03d}")
    if compute:      
        pc.DEFAULTS.ARRAY_MBYTES_MAX = 8.e+4
        ec = pc.BifrostCalculator(snapname)
        ec.component='z'
        ec.snap = f'{snap:03d}'
        ec.emiss_mode = 'notrac_noopa'
        ec.vdem_mode = 'allinterp'
        ec.units = 'cgs'
        ec.tabin.extrapolate_type = "constant"
        if telescope == 'muse':
            ec.rcoords_vdop_kms = np.arange(muse_vdop[0], muse_vdop[1], muse_vdop[2])
            ec.rcoords_logT = np.arange(4.5,7.1,0.1)
            logger.info(f'*** Running with vdop/logT set to MUSE standard')
        elif telescope == 'aia':
            ec.rcoords_vdop_kms = np.arange(aia_vdop[0], aia_vdop[1], aia_vdop[2])
            ec.rcoords_logT = np.arange(4.5,7.1,0.1)
            logger.info(f'*** Running with vdop set to AIA standard')
        elif telescope == 'iris':
            ec.rcoords_vdop_kms = np.arange(iris_vdop[0], iris_vdop[1], iris_vdop[2])
            ec.rcoords_logT = np.arange(4.2,6.1,0.1)
            logger.info(f'*** Running with vdop set to IRIS standard')
        else:
            logger.info(f'*** No such telescope {telescope}. Returning')
            return None, None
        ec.rcoords_wavelength_A = opa_wvl # wavelength of opacity if 'opa' is chosen in mode
        vdem = ec('vdem', chunks=dict(x=256), ncpu=12)      
        vdem = _xarray_save_prep(vdem)
        vdem0 = vdem[0]
        vdem0.to_zarr("_pc_caches_zarr_saving.zarr", zarr_format = zarr_format, mode = "w")
        vdem_temp0 = xr.open_zarr("_pc_caches_zarr_saving.zarr", zarr_format = zarr_format).compute()
        vdem0 = xr.Dataset()
        vdem0["vdem"] = vdem_temp0.vdem
        vdem0.attrs = vdem[0].attrs
        vdem0.x.attrs["long_name"] = "X"
        vdem0.y.attrs["long_name"] = "Y"
        vdem0.x.attrs["units"] = "cm"
        vdem0.y.attrs["units"] = "cm"
        vdem0.vdem.attrs["units"] = "1e27 / cm5"
        vdem0.vdem.attrs["description"] = "DEM(T,vel,x,y)"
        vdem0.vdop.attrs["long_name"] = r"v$_{Doppler}$"
        vdem0.vdop.attrs["units"] = "km/s"
        vdem0.logT.attrs["long_name"] = r"log$_{10}$(T)"
        vdem0.logT.attrs["units"] = r"log$_{10}$ (K)"
        vdem = vdem0.compute()
    else:
        try:
            vdem = xr.open_zarr(f'{zarr_file}.zarr').compute()
        except:
            vdem = xr.open_dataset(f'{zarr_file}.nc')
        save = False
    if save:
        try:
            vdem.to_zarr(f'{zarr_file}.zarr', mode = "w", zarr_format = zarr_format)
            logger.info(f"Saved vdem to {f'{zarr_file}.zarr'}")
        except:
            logger.warning(f"*** Warning: Could not save zarr file {f'{zarr_file}.zarr'}. Using NetCDF.")
            save_netcdf = True
    if save_netcdf:
            vdem.to_netcdf(f'{zarr_file}.nc', mode = "w")
            logger.info(f"Saved vdem to {f'{zarr_file}.nc'}")
    if save_bz:
        dd = br.BifrostData(snapname,snap)
        dd.set_snap(snap)
        iz0 = np.argmin(np.abs(dd.z - z0))
        bz = dd.get_var('bz')
        bz0 = bz[:,:,iz0]*dd.params['u_b'][0]
        bz_file = os.path.join(vdem_dir,f'Bz_z={-1.0*z0:0.2f}_{snap:03d}.npy')
        np.save(bz_file, bz0, allow_pickle = True)
        logger.info(f"Saved {bz_file}")
    else:
        try:
            logger.info(f'*** Attempting read of Bz0 from {vdem_dir} snap {snap}')
            bz0 = get_vdem_bz(vdem_dir, snap)
        except:
            logger.info(f'*** Cound not find any Bz file, returning None')
            bz0 = None
    return vdem, bz0

# **************************************************

def get_vdem_bz(bzdir, snap, z0 = -0.15):
       import numpy as np
       bzfile = os.path.join(bzdir,f'Bz_z={-1.0*z0:0.2f}_{snap:03d}.npy')
       f = np.load(bzfile)
       return f

# **************************************************

def aia_synthesis(aia_resp, work_dir, vdem_path, 
                  snap = None, swap_dims = True):
    import xarray as xr
    from muse.synthesis.synthesis import vdem_synthesis
    import os
    import glob
    logger.info(f"*** Work directory is {work_dir}")
    os.chdir(work_dir)
    
    if snap is None:
        files = vdem_path #glob.glob(os.path.join(vdem_dir,'*'))
    else:
        files = os.path.join('vdem',f'vdem_{snap:03d}.zarr')
    logger.info(f'*** Loading {files} into vdem')
    vdem = xr.open_zarr(files).compute()

    # vdem_cut
    #vdem_cut = vdem.sel(logT=aia_resp.logT, method = "nearest")
    #vdem_cut = vdem_cut.compute()
    #Synthesis AIA observations using the response function and VDEM
    muse_AIA = vdem_synthesis(vdem,
                              aia_resp,
                              sum_over=["logT","vdop"]) 
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
        logger.info(f"Fetching: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Number of events in JSON: {len(data.get('Events', []))}")
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
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

    logger.info(f"Saved {len(all_lines)} date ranges to {output_file}")

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
        logger.error(f"Error checking wavelengths in {data_file}: {e}")
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
        logger.error("No EIS data found for:", magnetogram_path)
        return
    # Find the first matching magnetogram and AIA file
    mag_files = glob(os.path.join(magnetogram_path, '*magnetogram.fits'))
    aia_files = glob(os.path.join(magnetogram_path, '*.193.image_lev1.fits'))
    if not mag_files or not aia_files:
        logger.error(f"Missing HMI or AIA files in {magnetogram_path}")
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
            logger.info(f"Saved HMI cutout to {output_file}")
        except Exception as e:
            logger.error(f"Error processing EIS data: {e}")

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
        logger.error("No EIS data found for:", magnetogram_path)
        return
    # Find the first matching magnetogram and AIA file
    mag_files = glob(os.path.join(magnetogram_path, '*magnetogram.fits'))
    aia_files = glob(os.path.join(magnetogram_path, '*.193.image_lev1.fits'))
    if not mag_files or not aia_files:
        logger.error(f"Missing HMI or AIA files in {magnetogram_path}")
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
            logger.info(f"Saved HMI cutout to {output_file}")
        except Exception as e:
            logger.error(f"Error processing EIS data: {e}")

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
        logger.error("No EIS data found for:", magnetogram_path)
        return
    # Find the first matching magnetogram and AIA file
    mag_files = glob(os.path.join(magnetogram_path, '*magnetogram.fits'))
    aia_files = glob(os.path.join(magnetogram_path, '*.193.image_lev1.fits'))
    if not mag_files or not aia_files:
        logger.error(f"Missing HMI or AIA files in {magnetogram_path}")
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
            logger.info(f"Saved HMI cutout to {output_file}")
        except Exception as e:
            logger.error(f"Error processing EIS data: {e}")

# **************************************************

def pick_sim(sim, work='/mn/stornext/d19/RoCS/viggoh/3d/',help = False):
   """
   Given short simulation name 'sim' cd to its directory and return snapname and workdir name.

   If workdir or simulation not found, instructions for fixing code given.

   Parameters
   ----------
   sim   : `str`
      Short memnonic name for simulation.
   work  : `str`
      Root directory for simulations, default is viggoh's workdir in Oslo.

   Returns
   -------
   snapname : `str`
      Full snapname of simulation.
   simdir : `str`
      Full directory specification of simulation.
   """
   from astropy.io import ascii
   from tabulate import tabulate
   Simulations = """
   'mnemonic'    'snapname'                         'simdir'
   'en'          'en024031_emer3.0str'              'en024031_emer3.0/str'
   'qs'          'qs072100'                         'qs072100'
   'qs50'        'qs072050'                         'qs072050'
   'qsd2'        'qs072100_d2'                      'qs072100_d2'
   'qsd2n'       'qs072100_d2n'                     'qs072100_d2'
   'qsx2n'       'qs072100_x2n'                     'qs072100_x2'
   'qsx2s'       'qs072100_x2s'                     'qs072100_x2'
   'qs50d2'      'qs072050'                         'qs072050_d2'
   'qsd4'        'qs072100_d4'                      'qs072100_d4'
   'qs50d4'      'qs072050'                         'qs072050_d4'
   'pl072100'    'pl072100'                         'pl072100'
   'pl072050'    'pl072050'                         'pl072050'
   'pl24'        'pl024031'                         'pl024031'
   'pl24hion'    'pl024031'                         'pl024031_hion'
   'nw'          'nw072100'                         'nw072100_alt'
   'sw'          'sw072050'                         'sw072050'
   'cbp'         'mn4_np3d_10g_8mm'                 'np3D_10G_8Mm'
   'cbpx2'       'mn4_np3d_10g_8mm_2res'            'np3D_10G_8Mm_2xres'
   """
   Simulations_Table = ascii.read(Simulations)
   if not os.path.exists(work) or help==True:
       logger.warning(f"*** Warning: directory {work} not found!!! Give an available workdir")
       print("Available sims are:")
       hdr = ['mnemonic','snapname','simdir']
       print(tabulate(Simulations_Table, headers = hdr, tablefmt='grid'))
       return None, None
   Simulations_Table.add_index('mnemonic')
   workdir = os.path.join(work,Simulations_Table.loc[sim]['simdir'])
   os.chdir(workdir)        
   logger.info(f"*** Now in directory {workdir}, snapname is {Simulations_Table.loc[sim]['snapname']}")
   return Simulations_Table.loc[sim]['snapname'],workdir

# **************************************************

def plot_aia_overview(muse_AIA, bz0, channels = [171, 193, 131, 211],
                      snapname = None, snap = 0,
                      code = 'Bifrost', save = False, len_scale = 'Mm', fontsize = 'x-large' ):
    fig,ax = plt.subplots(2,3, figsize = (24,12))
    arcsec2Mm = 0.729
    extent = np.array([min(muse_AIA.x),max(muse_AIA.x),min(muse_AIA.y),max(muse_AIA.y)])/1.e8
    if len_scale == 'arcsec':  
        extent = np.array(extent_bf)/arcsec2Mm
    for i,channel in enumerate(channels): 
        if channel < 100: cmap = f'sdoaia{channel:02d}' 
        else: cmap = f'sdoaia{channel:03d}'
        flux = np.squeeze(muse_AIA.flux.sel(channel = channel).to_numpy())
        im = ax[i//2][i%2].imshow(flux.T, 
                                  norm = colors.PowerNorm(0.3), 
                                  cmap = cmap, extent = extent)
        ax[i//2][i%2].set_aspect('equal')
        ax[i//2][i%2].set_xlabel(f'X [{len_scale}]', fontsize = fontsize)
        ax[i//2][i%2].set_ylabel(f'Y [{len_scale}]', fontsize = fontsize)
        ax[i//2][i%2].set_title(f'AIA {channel} Mean Intensity {muse_AIA.flux.sel(channel = channel).mean().to_numpy():0.2f} DN/s', fontsize = fontsize)
        divider = make_axes_locatable(ax[i//2][i%2])
        cax = divider.append_axes('right', size='3%', pad=0.1, axes_class=plt.Axes)
        cbar = fig.colorbar(im, cax=cax, extend ='both')
        cbar.ax.tick_params(direction='out')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.tick_right()
        cbar.set_label(rf'AIA Intensity [DN/s]', fontsize = fontsize)
    #
    im = ax[0][2].imshow(np.rot90(bz0, k=1), vmin = -500, vmax = 500, cmap=cm.Greys_r,extent = extent, origin = 'lower')
    ax[0][2].set_xlabel(f'X [{len_scale}]', fontsize = fontsize)
    ax[0][2].set_ylabel(f'Y [{len_scale}]', fontsize = fontsize)
    ax[0][2].set_title(fr'Mean magnetic field $\sqrt{{B_z^2}}$ = {np.mean(np.sqrt(bz0**2)):0.2f} Gauss', fontsize = fontsize)
    #
    divider = make_axes_locatable(ax[0][2])
    cax = divider.append_axes('right', size='3%', pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, extend ='both')
    cbar.ax.tick_params(direction='out')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    cbar.set_label(rf'{code} Vertical Field [Gauss]', fontsize = fontsize)

    ax[1][2].set_xlabel(r'$\log_{10}$DN/s',fontsize = fontsize)
    ax[1][2].set_ylabel('Percentage',fontsize = fontsize)
    ax[1][2].tick_params(labelsize='large')
    for channel in channels:
        ndata = len(np.ravel(muse_AIA.flux.sel(channel=channel).to_numpy()))
        ax[1][2].hist(np.log10(np.ravel(muse_AIA.flux.sel(channel=channel).to_numpy())),
                               bins=30, histtype='step', label=f'AIA {channel:03d}', weights = np.ones(ndata)/ndata)
    ax[1][2].legend(loc='upper right')
    ax[1][2].set_xlim([0.,4.])
    ax[1][2].set_title(f'{code} simulation: {snapname} snap: {snap:03d}')
    if save:
        plt.savefig(os.path.join('./figs',f"{snapname}_{snap}_AIA_overview.png"))


# **************************************************

def iny_smear(iny, wvl, dx = 0.1, resolution = 0.33):
    from scipy import signal
    fwhm = 2*np.sqrt(2*np.log(2))
    sptbin = resolution*0.729/dx/fwhm
    gauss_kern = gauss_kernel(size=int(10*sptbin), sigma=sptbin)
    for iwvl in range(np.shape(wvl)[0]):
        iny[:,:,iwvl] = signal.convolve2d(iny[:,:,iwvl], 
                                          gauss_kern,
                                          mode='same',
                                          boundary='wrap')
    return iny

# **************************************************

def gauss_kernel(size=3,sigma=1):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff_sq = (i-center)**2+(j-center)**2
          kernel[i,j]=np.exp(-diff_sq/(2*sigma**2))
    return kernel/np.sum(kernel)
