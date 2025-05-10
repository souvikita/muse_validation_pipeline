def find_response(obs_date, 
                  resp_dir = os.environ['RESPONSE'], 
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
              are stored, default is RESPONSE environment variable.
    delta_month: int, optional max number of months away from obs_date before
              new response function is suggested, default 12 months.
    units: str, optional, response function intensity units, default DN.
    verbose: bool, optional, be verbose, default True
    '''
    obs_date = Time(obs_date,format='isot',scale='utc')
    comp = int(obs_date.strftime("%y"))*12+int(obs_date.strftime("%m"))
    resp_files = glob.glob(os.path.join(resp_dir, f'aia_resp_{units}_*.zarr'))
    avail = []
    for f in resp_files:
       head = f.split('.')[0]
       rdate = datetime.strptime(head.split('_')[-1],"%b%y")
       avail.append(int(rdate.strftime("%y"))*12+int(rdate.strftime("%m")))
    iresp = np.argmin(np.abs(np.array(avail)-comp))
    if verbose:
        print(f'*** Nearest response function is {np.abs(avail[iresp]-comp)} months from obs_date ')
    if np.abs(avail[iresp]-comp) <= delta_month:
        zarr_file = resp_files[iresp]
    else:
        zarr_file = None
    return zarr_file, obs_date

def get_response(date, 
                save_response = False,
                use_QS_bands = True,
                units = 'DN',
                tmax=7.6,
                tmin=4.4,
                 tbin=0.1,
                abund = "sun_coronal_2021_chianti",
                pres = 3e15,
                dx_pix=0.6,dy_pix=0.6,
                 ):
#  Channels (lines/bands) relevant for (in this case) MUSE QS validation
    aia_goes_lines = ['AIA 94', 'AIA 131', 'AIA 171', 'AIA 193', 
                  'AIA 211', 'AIA 304','AIA 335', 
                  'GOES 15 1-8', 'GOES 15 0.5-4']
#  Temperature limits, abundance, pressure, and pixel size
#  NB note that available abundance files depend on Chianti version!
#  Other possible abundance files to look for...
#  abund = "sun_photospheric_2011_caffau"
#  abund = "sun_photospheric_2021_asplund"
    if use_QS_bands:
        aia_goes_lines.remove('AIA 304')
        lines = aia_goes_lines[0:6]
        bands = [int(s) for s in " ".join(lines).split("AIA ")[1:7]]
#bands = list(map(int," ".join(lines.split("AIA ")[1:5]))
    else:
        print("*** You should set use_QS_bands to true unless you know what you are doing!!!")
        sys.exit()
#
    zarr_file,obs_date = find_response(date, units = units)
    if zarr_file is not None:
        print(f'*** Temporary(incomplete) {zarr_file} already exists! Reading...')
    # it is not quite clear how to find the number of gains asked for... should be equal to the number of 
    # lines/bands that the response function was constructed with
        response_all = read_response(zarr_file, logT=vdem.logT,gain=np.ones((len(lines)))).compute()
    else:
        for line,band in zip(lines,bands):
            print(f'*** Computing {units} response function for {line} date {obs_date.strftime("%b%Y")}')
            ch = Channel(band*u.angstrom)
            correction_table = get_correction_table("jsoc")    # Check what this does!
            resp_band = ch.wavelength_response(obstime=obs_date, correction_table=correction_table)
            eff_xr = create_eff_area_xarray(resp_band.value, ch.wavelength.value, [ch.channel.value])
            line_list = create_resp_line_list(eff_xr, temin=10**tmin, abundance=abund, eDensity=3e10, num_slits=1)
            line_list = line_list.sortby(line_list.resp_func)
            line_list = line_list.sel(trans_index = line_list.trans_index[::-1][:350].data)  
            ''' Important, I'm considering here only 350 lines!!!!!!! 
                this creates the resposne function. Note that now we provide pressure 
                (it can also be an array) or not sum lines, but 
                if you have many it becomes a huge array!
            ''' 
            resp_eff_dn = create_resp_func_ci(
                line_list,
                temp=10 ** np.arange(tmin, tmax, tbin),
                pres=pres,
                abundance=abund,
                effective_area=eff_xr,
                sum_lines=True,
                gain=18,
                units="1e-27 DN cm5 / s",
                dx_pix = dx_pix, 
                dy_pix = dy_pix,
            )
            if line == lines[0]:
                response_all = resp_eff_dn
            else:
                response_all = xr.concat([response_all, resp_eff_dn], dim="band")
        response_all["SG_resp"] = response_all.SG_resp.fillna(0)
        response_all = response_all.compute()
        response_all = response_all.drop_dims("line")
        save_response = True
#
    response_all = response_all.assign_coords(line = ("band",['AIA '+f'{int(s)}' for s in response_all.band.data]))
    response_all = response_all.assign_attrs(date = obs_date.strftime("%d-%b-%Y"))
#    response_all = response_all.assign_attrs(abundance = abund) # Needed, or already in?
    response
#
#response_all = response_all.swap_dims({"band":"line"})
    response_all = response_all.compute()
#
    zarr_file = f'aia_resp_{units}_{obs_date.strftime("%b%y")}.zarr'
    zarr_file = os.path.join(os.environ['RESPONSE'],zarr_file)
    if save_response:
        response_all.to_zarr(zarr_file)
        print(f"Saved response to {zarr_file}  ")
    else:
        print(f"Response function not saved {zarr_file}")
    return response_all, obs_date

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

