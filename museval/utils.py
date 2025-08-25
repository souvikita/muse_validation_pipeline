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


def get_response(date = None, 
                 save_response = False,
                 units = 'DN',
                 lgtgmax=7.0,lgtgmin=4.4, lgtgstep=0.1,
                 abund = "sun_coronal_2021_chianti",
                 press = 3e15,
                 dx_pix=0.6, dy_pix=0.6,
                 bands = [94, 131, 171, 193, 211, 304, 335],  
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
        response_all = read_response(zarr_file).compute()
        if  np.array_equal(bands, response_all.band):
            print("The bands of the response function match.")
            return response_all ##this is fine, no need to create a new response function
        else:
           print("The bands of the response function do not match the requested bands. Creating a new response function.")
           need_new_response = True ##Treating this as a flag to create a new response function
            

    else:
        need_new_response = True
    if need_new_response:
        logT = xr.DataArray(np.arange(lgtgmin,lgtgmax, lgtgstep),dims='logT')
        pressure = xr.DataArray(np.array([press]), dims= 'pressure')
        print(f"*** Constructing line list")
        line_list = chianti_gofnt_linelist(temperature = 10**logT,
                                           pressure=pressure,
                                           abundance = abund,
                                           wavelength_range = [80,850],
                                           )
        for band in bands:
            ch = Channel(band*u.angstrom)
            if obs_date is None:
                #print(f'*** Computing {units} response function for {ch.channel.to_string()}')
                #response = ch.wavelength_response() * ch.plate_scale
                print(f'*** Computing {units} response function for {ch.channel.to_string()}')
                try:
                    response = ch.wavelength_response() * ch.plate_scale
                except:
                    print('*** Warning: correction table taken from local JSOC installation')
                    response = ch.wavelength_response(correction_table = aiapy.calibrate.util.get_correction_table('JSOC')) * ch.plate_scale
            else:
                print(f'*** Computing {units} response function for {ch.channel.to_string()}'
                       ' date {obs_date.strftime("%b%Y")}')
                try:
                    response = ch.wavelength_response(obstime = obs_date, correction_table = aiapy.calibrate.util.get_correction_table('JSOC')) * ch.plate_scale
                except:
                    print('*** Warning: correction table taken from local JSOC installation')
                    response = ch.wavelength_response(obstime = obs_date, correction_table = aiapy.calibrate.util.get_correction_table('JSOC')) \
                        * ch.plate_scale
                response = ch.wavelength_response(obstime = obs_date, correction_table = aiapy.calibrate.util.get_correction_table('JSOC')) * ch.plate_scale
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
            #    vdop=vdop,
                instr_width=0,  
                effective_area=eff_xr.eff_area,
                wvlr=[80, 800],
                num_lines_keep=0,
                )
            resp_dn = transform_resp_units(resp,
                                           new_units="1e-27 cm5 DN / (Angstrom s)",
                                           wvl=np.array(resp.wavelength.data),
                                           dx_pix=dx_pix, dy_pix=dy_pix,
                                           )
            ci_resp = convert_resp2muse_ciresp(resp_dn)
            line_list = line_list.drop_vars("resp_func")
            if band == bands[0]:
                response_all = ci_resp
            else:
                response_all = xr.concat([response_all, ci_resp], dim="band")
        response_all["SG_resp"] = response_all.SG_resp.fillna(0)
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
            response_all.to_zarr(zarr_file, mode = "w")
            print(f"Saved response to {zarr_file}  ")
        except:
            print(f"*** Error!! Could not write {zarr_file}")
    return response_all

def aia_synthesis(aia_resp, work_dir, vdem_dir, swap_dims = True):
    print(f"*** Work directory is {work_dir}")
    os.chdir(work_dir)
    
    files = glob.glob(os.path.join(vdem_dir,'*'))
    print(f'*** Loading {files[0]} into vdem')
    vdem = xr.open_zarr(files[0]).compute()

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

