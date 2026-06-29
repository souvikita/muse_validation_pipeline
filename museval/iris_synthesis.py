import os
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
#import br_py.tools as btr
from muse.synthesis.synthesis import transform_resp_units, vdem_synthesis
from muse.instr.utils import convert_resp2muse_sgresp
from muse.utils.utils import calculate_moments
from muse import logger
from irispreppy.radcal import iris_get_response as igr
import datetime as dt


### Below is marked for removal - replaced by make_vdem (resides in museval.utils)
""" def make_iris_vdem(simulation, snap,
                   save = True, 
                   save_bz = False, z0 = -0.15, # height at which to save Bz [Mm] 
                   compute = True,            # -> roughly equal to formation height of 617.3 nm HMI line
                   code = 'Bifrost', 
                   minltg = 4.2, maxltg = 5.5, dltg = 0.1,
                   minuz = -200, maxuz = 200, duz = 10.,
                   workdir = './',
                   ):
    ntg = int((maxltg-minltg)/dltg) + 1; print(f'Number of temperature bins {ntg:03d}')
    lgtaxis = np.linspace(minltg,maxltg,ntg)
    nuz = int((maxuz-minuz)/duz)+1 ; print(f'Number of velocity bins {nuz:03d}')
    syn_dopaxis = np.linspace(minuz, maxuz, nuz)
    snapname,workdir = ms.pick_sim(simulation, work = workdir)
    vdem_dir = os.path.join(workdir,"vdem")
    zarr_file = os.path.join(vdem_dir,f"iris_vdem_{snap:03d}")
    if compute:
        ddbtr = btr.UVOTRTData(snapname, snap, fdir="./", _class=code.lower())
        vdem = ddbtr.from_hel2vdem(snap, syn_dopaxis, lgtaxis, axis=2, xyrotation=True)
    else:
        try:
            vdem = xr.open_zarr(f'{zarr_file}.zarr')
        except:
            vdem = xr.open_dataset(f'{zarr_file}.nc')
        save = False
    if save:
        try:
            vdem.to_zarr(f'{zarr_file}.zarr', mode = "w")
            print(f"Saved vdem to {f'{zarr_file}.zarr'}")
        except:
            print(f"*** Error: Could not save zarr file {f'{zarr_file}.zarr'}. Using NetCDF.")
            vdem.to_netcdf(f'{zarr_file}.nc', mode = "w")
            print(f"Saved vdem to {f'{zarr_file}.nc'}")
    if save_bz:
        ddbtr.set_snap(snap)
        iz0 = np.argmin(np.abs(ddbtr.z - z0))
        bz0 = bz[:,:,iz0]
        bz_file = os.path.join(vdem_dir,f'Bz_z={-1.0*z0:0.2f}_{snap:03d}.npy')
        np.save(bz_file, bz0, allow_pickle = True)
        print(f"Saved {bz_file}")
    else:
        bz0 = get_vdem_bz(workdir, snap)
    return vdem, bz0 """

def make_line_response(ionstr = ["si_4"],
                       wvlr = np.array([1393.75,1393.76]),
                       abundance   = "sun_coronal_2021_chianti",
                       lgtgmin     = 4.0, lgtgmax     = 6.0, lgtgstep    = 0.1,                  # in log10
                       uzmax       = 200.0, uzmin       = -200.0, uzstep      = 5.0,             # km/s
                       instr_width = 0.0, # km/s
                       iris_file = None,
                       minimum_abundance = None,
                       ):
    import astropy.units as u
    from muse.instr.utils import create_eff_area_xarray
    from muse.instr.utils import chianti_gofnt_linelist, create_resp_func
    # Make sure the wavelength range is small to select a single line 
    # The VDEM could be expanded to include density dependence (and the response function)
    logT = xr.DataArray(np.arange(lgtgmin,lgtgmax, lgtgstep),dims='logT')
    pressure = xr.DataArray(np.array([3e15]), dims= 'pressure')
    vdop = np.arange(uzmin, uzmax, uzstep) * u.km / u.s
    # finds in Chiantypy the line list with the properties listed above
    line_list = chianti_gofnt_linelist(
        temperature = 10**logT,
        pressure=pressure,
        abundance = abundance,
        wavelength_range = wvlr,
        ionList=ionstr,
        minimum_abundance=minimum_abundance,
    )
    # Generates the response function
    wvlmin=line_list.wvl.min().values - 0.9
    wvlmax=line_list.wvl.max().values + 0.9
    n = line_list.sizes['trans_index']
    # We assume no instrumental broadening.
    if iris_file == None:
        eff_area = np.ones(2)
    else:
        logger.info(f'Reading {iris_file} to compute effective area')
        eff_area = iris_eff_area([wvlmin, wvlmax], iris_file = iris_file)
    effective_area = create_eff_area_xarray(eff_area, [wvlmin,wvlmax], [wvlr[0]])
    resp = create_resp_func(
        line_list,
        vdop = vdop,
        instr_width=instr_width,  # ??? 2.4 pix -> ~13.5mA
        effective_area=effective_area.eff_area,
        wvlr=[wvlmin, wvlmax],
        num_lines_keep=n,
    )
    return resp, line_list

def transform_iris_resp_units(
    resp_input: xr.Dataset,
    iris_file,
    line,
    new_units: str = "1e-27 cm5 DN / s",
) -> xr.Dataset:
    """
    Convert intensity units of a response function.

    It works from erg -> ph (erg*lambda/(h*c)), ph -> erg (or any other energy flux units).
    It allows also to go from sr or arsec to pixel size.

    Parameters
    ----------
    resp_input : `xarray.Dataset`
        Response function.
    iris_file : `str`
        IRIS fits file name, method uses parameters in fits header to set pixel size, effective area, etc.
    line      : `str` line name in IRIS fits file, e.g. 'Si IV 1394'
    new_units : `str`
        New units, it uses the format of str(astropy.units.XXX), by default "1e-27 cm5 DN / s"

    Returns
    -------
    resp_input : `xarray.Dataset`
        A new Dataset of the response function in new_units.
    """
    from weno4 import weno4
    from muse.utils.utils import add_history
    resp = resp_input.copy(deep=True)
    if "line" not in resp.dims:
        resp = resp.swap_dims({"channel": "line"})
    if "units" not in resp.SG_resp.attrs:
        # assumes that units are in 1.e-27 cm^5 erg / angstrom s sr 
        resp.SG_resp.attrs["units"] = str(1.e-27*u.cm**5*u.ergu / (u.angstrom*u.s*u.sr))

    iris = iris_radiometric(iris_file)

    logger.info("Transform Response Function Units")
    resp_ph = transform_resp_units(
        resp,
        new_units="1e-27 cm5 DN / (Angstrom s)",
        wvl=np.array(resp.wavelength.data),
        dx_pix=iris['wslit'],
        dy_pix=iris['cdelty'],
        gain = 10. # from Paul Boerner May 5, 2026
    )

    logger.info("Map response function to SG detector")
    resp_dn = convert_resp2muse_sgresp(
        resp_ph,
        wvlo=resp.wavelength.data[0],
        npix=1000,
        nslits=1,
        dma_per_pix=iris["cdeltw"][line] * 1e3,
        pixel_per_slit=1,
    )

    add_history(resp_dn, locals(), transform_iris_resp_units)
    return resp_dn, iris['fovx'], iris['fovy'], iris['cdelty'], iris['cdeltw'] 

def iris_radiometric(iris_file,
                     set_exptime = False):
        from astropy.io import fits
        from weno4 import weno4
        hdr = fits.getheader(iris_file)
        if 'STARTOBS' not in hdr:
                begin=dt.datetime.strptime(hdr['DATE_OBS'], '%Y-%m-%dT%H:%M:%S.%f')
        else:
                begin=dt.datetime.strptime(hdr['STARTOBS'], '%Y-%m-%dT%H:%M:%S.%f')
        if 'ENDOBS' not in hdr:
                end=dt.datetime.strptime(hdr['DATE_END'], '%Y-%m-%dT%H:%M:%S.%f')
        else:
                end=dt.datetime.strptime(hdr['ENDOBS'], '%Y-%m-%dT%H:%M:%S.%f')
        midtime=dt.datetime.strftime((begin+((end-begin)/2)), '%Y-%m-%dT%H:%M:%S.%fZ')

        response=(igr.iris_get_response(midtime, quiet=False))[0]

        if response['NAME_SG'][0]==b'FUV' and response['NAME_SG'][1]==b'NUV':
                FUVind=0
                NUVind=1
        elif response['NAME_SG'][1]==b'FUV' and response['NAME_SG'][0]==b'NUV':
                FUVind=1
                NUVind=0
        else:
                print("[NAME_SG]="+str(response['NAME_SG']))
                raise RuntimeError("FUV and NUV cannot be found automatically. Please check ['NAME_SG'] from irisresponse above.")

        indices={hdr[name]: ind+1 for ind, name in enumerate(hdr['TDESC*'])}

        if indices=={}:
                #Full disc mosaic
                if hdr['CRVAL3'] > 2000:
                        indices={'fdNUV':0}
                else:
                        indices={'fdFUV':0}

        ###################
        # FOVX, FOVY.     #
        ###################
        fovx = hdr['FOVX']
        fovy = hdr['FOVY']

        ###################
        # Wavelength axes #
        ###################
        FUV=np.where(response['AREA_SG'][FUVind]>0)[0]
        FUVcutoff=np.where((FUV[1:]-FUV[:-1])>1)[0][0]+1

        FUV1=FUV[:FUVcutoff]
        FUV1=response['LAMBDA'][FUV1]*10

        FUV2=FUV[FUVcutoff:]
        FUV2=response['LAMBDA'][FUV2]*10

        NUV=(response['LAMBDA'][response['AREA_SG'][NUVind]>0])*10

        #################
        # Photon Energy #
        #################
        #Lambda is in A. 1e7 ergs = 1 Joule
        h=6.62607004e-34
        c=3e8
        eFUV1=1e7*h*c/(FUV1*1e-10)
        eFUV2=1e7*h*c/(FUV2*1e-10)
        eNUV=1e7*h*c/(NUV*1e-10)

        ##################
        # Effective Area #
        ##################
        aFUV1=np.trim_zeros(response['AREA_SG'][FUVind][FUV[:FUVcutoff]])
        aFUV2=np.trim_zeros(response['AREA_SG'][FUVind][FUV[FUVcutoff:]])
        aNUV=np.trim_zeros(response['AREA_SG'][NUVind])

        del FUV

        ###########
        # DN2PHOT #
        ###########
        d2pFUV=response['DN2PHOT_SG'][FUVind]
        d2pNUV=response['DN2PHOT_SG'][NUVind]

        ###############
        #Exposure Time#
        ###############
        if set_exptime:
                if 'EXPTIMEN' in hdr:
                        tnuv=hdr['EXPTIMEN']
                else:
                        tnuv=hdr['EXPTIME']
                if 'EXPTIMEF' in hdr:
                        tfuv=hdr['EXPTIMEF']
                else:
                        tfuv=hdr['EXPTIME']
        else:
                tfuv = 1.
                tnuv = 1.

        ############
        #Slit Width#
        ############
        wslit=np.pi/(180*3600*3)

        ##################################
        #Spectral Pixel Width [angstroms]#
        ##################################
        # ...and...
        ##############################
        #Spatial Pixel Size [radians]#
        ##############################
        #ITN26 is a little cryptic about this, but I swear this is right
        for key in indices:
                ehdr = fits.getheader(iris_file, ext = int(indices[key])) 
                if indices[key] == 1:
                        pixl = {key: ehdr['CDELT1']}
                        pixxy = {key: ehdr['CDELT2']*np.pi/(180*3600)}
                else:
                        pixl[key] = ehdr['CDELT1']
                        pixxy[key] = ehdr['CDELT2']*np.pi/(180*3600)

        #############
        # Constants #
        #############
        const={}
        for key in indices:
                if key=='fdFUV':
                        const[key]=d2pFUV/(pixxy[key]*pixl[key]*tfuv*wslit)
                elif key=='fdNUV':
                        const[key]=d2pNUV/(pixxy[key]*pixl[key]*tnuv*wslit)
                elif 'FUV' in hdr['TDET'+str(indices[key])]:
                        const[key]=d2pFUV/(pixxy[key]*pixl[key]*tfuv*wslit)
                else:
                        const[key]=d2pNUV/(pixxy[key]*pixl[key]*tnuv*wslit)

        ###########################################################
        # Wavelength Trimming and Radiometric Calibration Factors #
        ###########################################################    
        lamwin={} #LAMbda WINdow
        wvlns={}
        rcfs={} #Radiometric Calibration FactorS
        cdelt={}
        crval={}

        for key in indices:
                ehdr = fits.getheader(iris_file, ext = int(indices[key])) 
                if key=='fdFUV':
                        wvlns[key]=(np.arange(0, ehdr['NAXIS3'])-ehdr['CRPIX3']+1)*ehdr['CDELT3']+ehdr['CRVAL3']
                        lamwin[key]=np.arange(0, len(wvlns[key]))[(wvlns[key]>FUV1[0])&(wvlns[key]<FUV1[-1])]
                        lamwin[key]=lamwin[key][0:len(lamwin[key]):len(lamwin[key])-1]            
                        rcfs[key]=weno4(wvlns[key][lamwin[key][0]:lamwin[key][1]], FUV1, eFUV1/aFUV1)*const[key]
                        cdelt[key] = ehdr['CDELT3']
                        crval[key] = ehdr['CRVAL3']
                elif key=='fdNUV':
                        wvlns[key]=(np.arange(0, ehdr['NAXIS3'])-ehdr['CRPIX3']+1)*ehdr['CDELT3']+ehdr['CRVAL3']
                        lamwin[key]=np.arange(0, len(wvlns[key]))[(wvlns[key]>NUV[0])&(wvlns[key]<NUV[-1])]
                        lamwin[key]=lamwin[key][0:len(lamwin[key]):len(lamwin[key])-1]            
                        rcfs[key]=weno4(wvlns[key][lamwin[key][0]:lamwin[key][1]], NUV, eNUV/aNUV)*const[key]
                        cdelt[key] = ehdr['CDELT3']
                        crval[key] = ehdr['CRVAL3']

                elif hdr['TDET'+str(indices[key])]=='FUV1':
                        wvlns[key]=(np.arange(0, ehdr['NAXIS1'])-ehdr['CRPIX1']+1)*ehdr['CDELT1']+ehdr['CRVAL1']
                        lamwin[key]=np.arange(0, len(wvlns[key]))[(wvlns[key]>FUV1[0])&(wvlns[key]<FUV1[-1])]
                        lamwin[key]=lamwin[key][0:len(lamwin[key]):len(lamwin[key])-1]            
                        rcfs[key]=weno4(wvlns[key][lamwin[key][0]:lamwin[key][1]], FUV1, eFUV1/aFUV1)*const[key]
                        cdelt[key] = ehdr['CDELT1']
                        crval[key] = ehdr['CRVAL1']

                elif hdr['TDET'+str(indices[key])]=='FUV2':
                        wvlns[key]=(np.arange(0, ehdr['NAXIS1'])-ehdr['CRPIX1']+1)*ehdr['CDELT1']+ehdr['CRVAL1']
                        lamwin[key]=np.arange(0, len(wvlns[key]))[(wvlns[key]>FUV2[0])&(wvlns[key]<FUV2[-1])]
                        lamwin[key]=lamwin[key][0:len(lamwin[key]):len(lamwin[key])-1]            
                        rcfs[key]=weno4(wvlns[key][lamwin[key][0]:lamwin[key][1]], FUV2, eFUV2/aFUV2)*const[key]
                        cdelt[key] = ehdr['CDELT1']
                        crval[key] = ehdr['CRVAL1']

                elif hdr['TDET'+str(indices[key])]=='NUV':
                        wvlns[key]=(np.arange(0, ehdr['NAXIS1'])-ehdr['CRPIX1']+1)*ehdr['CDELT1']+ehdr['CRVAL1']
                        lamwin[key]=np.arange(0, len(wvlns[key]))[(wvlns[key]>NUV[0])&(wvlns[key]<NUV[-1])]
                        lamwin[key]=lamwin[key][0:len(lamwin[key]):len(lamwin[key])-1]            
                        rcfs[key]=weno4(wvlns[key][lamwin[key][0]:lamwin[key][1]], NUV, eNUV/aNUV)*const[key]
                        cdelt[key] = ehdr['CDELT1']
                        crval[key] = ehdr['CRVAL1']

                else:
                        raise ValueError("You have detectors that are not FUV1, FUV2, or NUV in your fits file.")

        for i in list(rcfs.keys()):
                rcfs[i]=rcfs[i].astype(np.float32)

        params = {"wvlns": wvlns, "lamwin": lamwin, "rcfs": rcfs, "cdeltw": cdelt, "crval": crval,
                  "wslit": wslit*3600.*180./np.pi, "cdelty": ehdr['CDELT2'], "fovx": fovx, "fovy": fovy}

        return params

def iris_eff_area(wvl, date=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%dT%H:%M:%S.%fZ'), iris_file = None):
    """
    Returns iris effective area for given wavelength region and date.

    If IRIS fits file name given, date is set to observation time found in file header.

    Parameters
    ----------
    wvl : `numpy.ndarray` 
        Wavelengths at which effective area is returned (assumed given in AA).
    date : `str` 
        Effective area date desired. Default is 'now'.
    iris_file : `str`
        Name of IRIS data file.

    Returns
    -------
    eff_area : `numpy.ndarray`
        Effective area for given wavelengths and date.
    """
    from irispreppy.radcal import iris_get_response as igr
    from weno4 import weno4
    from astropy.io import fits
    import astropy.units as u
    if iris_file is not None:
        hdr = fits.getheader(iris_file)
        if 'STARTOBS' not in hdr:
                begin=dt.datetime.strptime(hdr['DATE_OBS'], '%Y-%m-%dT%H:%M:%S.%f')
        else:
                begin=dt.datetime.strptime(hdr['STARTOBS'], '%Y-%m-%dT%H:%M:%S.%f')
        if 'ENDOBS' not in hdr:
                end=dt.datetime.strptime(hdr['DATE_END'], '%Y-%m-%dT%H:%M:%S.%f')
        else:
                end=dt.datetime.strptime(hdr['ENDOBS'], '%Y-%m-%dT%H:%M:%S.%f')
        date=dt.datetime.strftime((begin+((end-begin)/2)), '%Y-%m-%dT%H:%M:%S.%fZ')
        
    response=(igr.iris_get_response(date, quiet=False))[0]
    if response['NAME_SG'][0]==b'FUV' and response['NAME_SG'][1]==b'NUV':
        FUVind=0
        NUVind=1
    elif response['NAME_SG'][1]==b'FUV' and response['NAME_SG'][0]==b'NUV':
        FUVind=1
        NUVind=0
    else:
        print("[NAME_SG]="+str(response['NAME_SG']))
        raise RuntimeError("FUV and NUV cannot be found automatically. Please check ['NAME_SG'] from irisresponse above.")
    
    lam = response['LAMBDA']*u.nm
    lam = lam.to(u.AA).value
    if wvl[0] > 2000.:
        ind = NUVind
    else:
        ind = FUVind
    area = response['AREA_SG'][ind]
    eff_area = weno4(wvl, lam, area)
    
    return eff_area

def match_vdop_logT(vdem, resp, dvdop = 10):
    vr = np.arange(np.max([vdem.vdop.min().data,resp.vdop.min().data]),
                   np.min([vdem.vdop.max().data,resp.vdop.max().data])+dvdop,dvdop)
    vdem_red = vdem.isel(vdop=np.arange(np.size(vr)))
    for iii, ii in enumerate(vr):
        data = vdem.vdem.where(vdem.vdop <= ii)
        if iii>0:
            data = data.where(data.vdop > vr[iii-1]).sum(dim="vdop")
        else:
            data = data.sum(dim='vdop')
        vdem_red.vdem.loc[{"vdop": vdem_red.vdop.isel(vdop=iii).data}] = data
    vdem_red.coords["vdop"] = vr
    resp_interp = resp.interp(vdop=vr)
#
    logT_matched = resp_interp.logT.where(np.isin(np.float32(resp_interp.logT), 
                                          np.float32(vdem.logT))).dropna('logT')
    vdem_interp = vdem_red.sel(logT=logT_matched,method="nearest")
    resp_interp = resp_interp.sel(logT=logT_matched, method="nearest")
    vdem_interp.coords["logT"] = resp_interp.coords["logT"]
    return vdem_interp, resp_interp

def transform_iris_resp_units_old(
    resp_input: xr.Dataset,
    iris_file,
    line,
    new_units: str = "1e-27 cm5 DN / s",
) -> xr.Dataset:
    """
    Convert intensity units of a response function.

    It works from erg -> ph (erg*lambda/(h*c)), ph -> erg (or any other energy flux units).
    It allows also to go from sr or arsec to pixel size.

    Parameters
    ----------
    resp_input : `xarray.Dataset`
        Response function.
    iris_file : `str`
        IRIS fits file name, method uses parameters in fits header to set pixel size, effective area, etc.
    line      : `str` line name in IRIS fits file, e.g. 'Si IV 1394'
    new_units : `str`
        New units, it uses the format of str(astropy.units.XXX), by default "1e-27 cm5 DN / s"

    Returns
    -------
    resp_input : `xarray.Dataset`
        A new Dataset of the response function in new_units.
    """
    from weno4 import weno4
    from muse.utils.utils import add_history
    resp = resp_input.copy(deep=True)
    if "line" not in resp.dims:
        resp = resp.swap_dims({"channel": "line"})
    if "units" not in resp.SG_resp.attrs:
        # assumes that units are in 1.e-27 cm^5 erg / angstrom s sr 
        resp.SG_resp.attrs["units"] = str(1.e-27*u.cm**5*u.ergu / (u.angstrom*u.s*u.sr))

    iris = iris_radiometric(iris_file)

    units_conv = weno4(resp.wavelength, 
                       iris['wvlns'][line][iris['lamwin'][line][0]:iris['lamwin'][line][1]],iris['rcfs'][line] )

    if np.size(units_conv) > 1:
        units_ds = xr.DataArray(data=units_conv, dims=resp.wavelength.dims, coords={"wavelength": resp.wavelength})
        resp["SG_resp"] = resp.SG_resp / units_ds
#        wvl_ds = xr.DataArray(data = iris)
        resp["wavelength"] = (resp.wavelength-iris['crval'][line])/iris['cdeltw'][line]
    else:
        resp["SG_resp"] = resp.SG_resp.isel(line=0) / units_conv
        resp["wavelength"] = (resp.wavelength-iris['crval'][line])/iris['cdeltw'][line]

    resp.SG_resp.attrs["units"] = new_units

    add_history(resp, locals(), transform_iris_resp_units_old)
    return resp, iris['fovx'], iris['fovy'], iris['cdelty'], iris['cdeltw'] 

def iris_validation_plot(odate, 
                         iris_dir,
                         vdem_sel,
                         resp,
                         line = 'Si IV 1403',
                         ulim = 50., num_bins = 30, Log = False,
                         unit = 'DN/s',
                         code = 'Bifrost', Region = 'Quiet Sun',
                         hdr = None, exptime = 1.,
                         iris_alt = None,
                         save = False,
                         dir = './figs',
                         convolve = True,
                           ):
    arcsec2Mm = 0.729
    date = dt.datetime.strftime(odate,'%Y-%m-%dT%H:%M:%S')
    sdate = dt.datetime.strftime(odate,'%Y%m%d_%H%M')
    iris_file = glob.glob(os.path.join(iris_dir,f'iris_l2*{sdate}*raster*.fits'))[0]
    bf_data = iris_total_line(vdem_sel, resp, iris_file = iris_file, convolve = convolve) 
    fdate = dt.datetime.strftime(odate,'%Y-%m-%dT%H-%M-00')
    file = np.load(os.path.join(iris_dir,f'iris_fit_results_{fdate}.npz'))
    intensity = 'gaussian'
    if hdr is None:
        iris_flux = file['net_flux']
    else:
        exptime = hdr['exptime']
        if unit == 'DN/s':
            iris_flux = file['net_flux']/exptime # Not needed: Souvik says he has converted to DN/s
        else:
            iris_flux = file['net_flux']
    if iris_alt is not None:
        if unit == 'DN/s':
            iris_flux = iris_alt/exptime
        else:
            iris_flux = iris_alt
        intensity = 'moment'
    data = np.ravel(iris_flux)
    data[data < 1] = 1.
    low,high = np.nanpercentile(data, [3,99.5])
    hlow, hhigh = low,high
    extent = [0,bf_data['fovx'],0,bf_data['fovy']]
    #
    bf = bf_data['bf']
    if unit == 'DN':
        bf *= exptime
    low_bf,high_bf = np.nanpercentile(bf, [3,99.5])
    extent_bf = np.array([np.min(vdem_sel.x.to_numpy())/1.e8,np.max(vdem_sel.x.to_numpy())/1.e8,
                np.min(vdem_sel.y.to_numpy())/1.e8,np.max(vdem_sel.y.to_numpy())/1.e8])/arcsec2Mm

    fig,ax = plt.subplots(2,3, figsize = (16,10))
    if hdr is None:
        OBSID = ''
    else:
        OBSID = hdr['OBSID']
    if Log:
        data_mean = np.log10(np.nanmean(data))
        code_mean = np.log10(np.nanmean(bf))
        data = np.log10(data)
        bf = np.log10(bf)
        hlow = np.log10(hlow)
        hhigh = np.log10(hhigh)
        hlow_bf = np.log10(low_bf)
        hhigh_bf = np.log10(high_bf)
    else:
        hlow_bf = low_bf
        hhigh_bf = high_bf
        data_mean = np.nanmean(data)
        code_mean = np.nanmean(bf)
    ax[0][0].hist(data, bins=num_bins, range = (hlow,hhigh), 
              label=f'IRIS {OBSID} {date}', cumulative=True, histtype='step',
              weights=np.ones(len(data)) / len(data), color = 'blue')
    ax[0][0].axvline(x=data_mean, lw=1, ls='-', color = 'blue')

    ax[0][0].hist(bf, bins=num_bins, range = (hlow_bf,hhigh_bf), 
              label=f'{code} {Region}',cumulative=True, histtype='step', 
              weights=np.ones(len(bf)) / len(bf),color='tab:purple',ls='-.')
    ax[0][0].axvline(x=code_mean, color='tab:purple', lw=1, ls='-.')
    if Log:
        unit_txt = fr'$\log_{{10}}$([{unit}])'
    else:
        unit_txt = f'[{unit}]'
    code_txt = fr'{code} {line} Mean {code_mean:0.1f} {unit_txt}'
    ax[0][0].text((hhigh-hlow)*0.02+min([hlow,hlow_bf]), 0.94, code_txt, color = 'tab:purple')

    ax[0][0].set_xlabel(fr'{line} Intensity {unit_txt}')
    ax[0][0].set_ylabel('ECDF')
    ax[0][0].legend(loc='lower right')
    ax[0][0].axhline(y=0.25,color='black',ls='--')
    ax[0][0].axhline(y=0.75,color='black',ls='--')
    if Log:
        iris_txt = fr'IRIS {line} Mean {data_mean:0.1f} {unit}'
    else:
        iris_txt = fr'IRIS {line} Mean {data_mean:0.1f} {unit}'
    ax[0][0].text((hhigh-hlow)*0.02+min([hlow,hlow_bf]), 0.99, iris_txt, color = 'blue')

    low = min(low,low_bf)
    high = max(high, high_bf)
    if Log:
        im = ax[0][1].imshow(np.rot90(iris_flux, k = 1), 
                             norm = colors.LogNorm(vmin = low, vmax = high), 
                             cmap='irissji1400', 
                             extent = extent)
    else:
        im = ax[0][1].imshow(np.rot90(iris_flux, k = 1), 
                      vmin = low, vmax = high, cmap='irissji1400',
                      extent = extent)
    ax[0][1].set_xlabel('arcsec')
    ax[0][1].set_ylabel('arcsec')

    divider = make_axes_locatable(ax[0][1])
    cax = divider.append_axes('right', size=0.1, pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, extend ='both')
    cbar.ax.tick_params(direction='out')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    cbar.set_label(rf'IRIS {line} {unit}')

    if unit == 'DN':
        bf_data['zeroth'] *= 1.
    if Log:
        im = ax[0][2].imshow(bf_data['zeroth'].T, 
                             norm = colors.LogNorm(vmin = low, vmax = high), 
                             cmap='irissji1400', 
                             extent = extent_bf)
    else:
        im = ax[0][2].imshow(bf_data['zeroth'].T, 
                             vmin = low, vmax = high, 
                             cmap='irissji1400',
                             extent = extent_bf)
    ax[0][2].set_xlabel('arcsec')
    ax[0][2].set_ylabel('arcsec')

    divider = make_axes_locatable(ax[0][2])
    cax = divider.append_axes('right', size=0.1, pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, extend ='both')
    cbar.ax.tick_params(direction='out')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    cbar.set_label(rf'{code} {line} {unit}')

# Velocities
    iris_vel = file['core_shift']
    uzdata = np.ravel(iris_vel)
    low,high = np.nanpercentile(uzdata, [0,99.5])
    low,high = -1*ulim,ulim
    uzbf = bf_data['uzbf']
    low_bf,high_bf = np.nanpercentile(uzbf, [0,99.5])
    low_bf,high_bf = -1*ulim,ulim

    ax[1][0].hist(uzdata, bins=num_bins, range = (low,high), label=f'IRIS HOP 307 {date}', histtype='step',
               weights=np.ones(len(uzdata)) / len(uzdata), color = 'blue')
    ax[1][0].axvline(x=np.nanmean(uzdata), lw=1, ls='-', color = 'blue')

    ax[1][0].hist(uzbf, bins=num_bins, range = (low_bf,high_bf), label=f'{code} {Region}', histtype='step', 
               weights=np.ones(len(uzbf)) / len(uzbf),color='tab:purple',ls='-.')
    ax[1][0].axvline(x=np.mean(uzbf), color='tab:purple', lw=1, ls='-.')

    ax[1][0].set_xlabel(fr'{line} $u_z$ [km$^{{-1}}$]')
    ax[1][0].set_ylabel('ECDF')
    #ax[1][0].legend(loc='lower left')

    im = ax[1][1].imshow(np.rot90(iris_vel,k = 1), 
                      vmin = -1*ulim, vmax = ulim, cmap=cm.RdBu_r,
                      extent = extent)
    ax[1][1].set_xlabel('arcsec')
    ax[1][1].set_ylabel('arcsec')

    divider = make_axes_locatable(ax[1][1])
    cax = divider.append_axes('right', size=0.1, pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, extend ='both')
    cbar.ax.tick_params(direction='out')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    cbar.set_label(rf'IRIS {line} km/s')

    im = ax[1][2].imshow(bf_data['first'].T, vmin = -1*ulim, vmax = ulim, cmap=cm.RdBu_r,extent = extent_bf)
    ax[1][2].set_xlabel('arcsec')
    ax[1][2].set_ylabel('arcsec')

    divider = make_axes_locatable(ax[1][2])
    cax = divider.append_axes('right', size=0.1, pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, extend ='both')
    cbar.ax.tick_params(direction='out')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    cbar.set_label(rf'{code} {line} km/s')
#
    if save:
      if Log:
        savefile = os.path.join(dir,f'{code}_{'_'.join(line.split())}_Log_{intensity}_validation.png')
      else:
        savefile = os.path.join(dir,f'{code}_{'_'.join(line.split())}_{intensity}_validation.png')
      print(f'*** Saving {savefile}')
      plt.savefig(savefile)

def iris_total_line(vdem_sel, resp,
                    iris_file = None,
                    code = 'Bifrost',
                    line = 'Si IV 1403',
                    resolution_iris = 0.33,
                    dx_sim = 0.1, convolve = False, add_noise = True,
                    readout = 20., # e- readout noise
                    ):
    
    from scipy import signal
    arcsec2Mm = 0.729
    resp_dn, fovx, fovy, cdelty, cdeltw = transform_iris_resp_units(resp, iris_file, line)
    #spec_dn = (vdem_sel.vdem.sum(dim="vdop") * resp_dn.SG_resp.sel(vdop=0).sum(dim="SG_xpixel")).squeeze().sum(dim="logT")
    spec_dn = vdem_synthesis(vdem_sel, resp_dn, sum_over=['logT', 'vdop'])
    mom_dn = calculate_moments(spec_dn, moment_dim='SG_xpixel')
    zeroth =  mom_dn["0th"].sel(line='Si IV 1402.77').squeeze().to_numpy()
    if convolve:
        fwhm = 2.*np.sqrt(2*np.log(2))
        sptbin = resolution_iris*arcsec2Mm/dx_sim/fwhm
        gauss_kern = gauss_kernel(size=int(10*sptbin),sigma=sptbin)
        zeroth = signal.convolve2d(zeroth,gauss_kern,mode='same',boundary='wrap')
    if add_noise:
        DN_read = readout/10.
        img = np.random.rand(vdem_sel.x.size, vdem_sel.y.size)
        noise = np.random.normal(DN_read/2., DN_read, img.shape)
        noise[noise < 0.] = 0.
        zeroth = zeroth + noise 
    bf = zeroth.ravel() 
    #cdeltw_sim = (resp.SG_wvl.to_numpy()[0][1]-resp.SG_wvl.to_numpy()[0][0])
    #bf *= cdeltw_sim/cdeltw[line]  # correcting for difference wvl pixel size in sim and iris
    first = mom_dn["1st"].sel(line='Si IV 1402.77').squeeze().to_numpy()
    if code == 'Bifrost':
        first = -1*first # Left handed -> right handed reference system
    uzbf = first.ravel()
    second = mom_dn["2nd"].sel(line='Si IV 1402.77').squeeze().to_numpy()
    return {"zeroth":zeroth, "first":first, "second":second, "fovx":fovx, "fovy":fovy, "bf":bf, "uzbf":uzbf} 

def iris_intensity_moment(iris_file, 
                          line = "Si IV 1403",
                          linewvl = [1402,1404],
                          contwvl = [1399,1400]):
    from irispy.io import read_files
    import astroscrappy
    from astropy.visualization import quantity_support
    from sunpy.map import Map
    quantity_support()
    # read and clean raster data
    raster = read_files(iris_file)
    wvl = raster[line][0].axis_world_coords("wl")[0].to_value("angstrom")
    clean_raster = raster[line][0].data*0.
    for i in range(np.shape(clean_raster)[0]):
        raster_slit = raster[line][0][i]
        _, clean_raster[i,:,:] = astroscrappy.detect_cosmics(raster_slit.data)
    # compute moment
    iw0, iw1 = np.argmin(np.abs(wvl-linewvl[0])),np.argmin(np.abs(wvl-linewvl[1]))
    ic0, ic1 = np.argmin(np.abs(wvl-contwvl[0])),np.argmin(np.abs(wvl-contwvl[1]))
    itot = np.trapz(clean_raster[:,:,iw0:iw1], axis=2)
    icont = np.trapz(clean_raster[:,:,ic0:ic1], axis=2)
    return itot,icont

def gauss_kernel(size=3,sigma=1):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff_sq = (i-center)**2+(j-center)**2
          kernel[i,j]=np.exp(-diff_sq/(2*sigma**2))
    return kernel/np.sum(kernel)
    
def pixelate(arr, block_size):
    h, w = arr.shape

    # Trim array so dimensions divide evenly
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)

    arr = arr[:h_trim, :w_trim]

    # Reshape into blocks
    reshaped = arr.reshape(
        h_trim // block_size,
        block_size,
        w_trim // block_size,
        block_size
    )

    # Average each block
    small = reshaped.mean(axis=(1, 3))

    # Expand back to original resolution
    pixelated = np.repeat(
        np.repeat(small, block_size, axis=0),
        block_size,
        axis=1
    )

    return pixelated