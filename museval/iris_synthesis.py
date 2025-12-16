import os
import numpy as np
import xarray as xr
import br_py.tools as btr
from muse.synthesis.synthesis import vdem_synthesis
from irispreppy.radcal import iris_get_response as igr
import datetime as dt


def make_iris_vdem(simulation, snap,
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
    return vdem, bz0

def make_line_response(ionstr = ["si_4"],
                       wvlr = np.array([1393.75,1393.76]),
                       abundance   = "sun_coronal_2021_chianti",
                       lgtgmin     = 4.0, lgtgmax     = 6.0, lgtgstep    = 0.1,                  # in log10
                       uzmax       = 200.0, uzmin       = -200.0, uzstep      = 5.0,             # km/s
                       instr_width = 0.0, # km/s
                       ):
    import astropy.units as u
    from muse.instr.utils import create_eff_area_xarray
    from muse.instr.utils import chianti_gofnt_linelist, create_resp_func
    # Make sure the wavelength range is small to select a single line 
    # The VDEM could be expanded to include density dependence (and the response function)
    logT = xr.DataArray(np.arange(lgtgmin,lgtgmax, lgtgstep),dims='logT')
    pressure = xr.DataArray(np.array([3e15]), dims= 'pressure')
    vdop = np.arange(uzmin, uzmax, uzstep) * u.km / u.s
    # We assume no instrumental broadening. 
    effective_area_unity = create_eff_area_xarray(np.ones(2), [1393.,1394.5], [wvlr[0]])
    # finds in Chiantypy the line list with the properties listed above
    line_list = chianti_gofnt_linelist(
        temperature = 10**logT,
        pressure=pressure,
        abundance = abundance,
        wavelength_range = wvlr,
        ionList=ionstr,
        minimum_abundance=1e5,
    )
    # Generates the response function
    wvlmin=line_list.wvl.min().values - 0.9
    wvlmax=line_list.wvl.max().values + 0.9
    n = line_list.sizes['trans_index']

    resp = create_resp_func(
        line_list,
        vdop = vdop,
        instr_width=instr_width,  # ??? 2.4 pix -> ~13.5mA
        effective_area=effective_area_unity.eff_area,
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
#    resp["FOVX"] = iris['fovx']
#    resp["FOVY"] = iris['fovy']
#    resp["CDELTY"] = iris['cdelty']

    add_history(resp, locals(), transform_iris_resp_units)
    return resp, iris['fovx'], iris['fovy'], iris['cdelty']

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
                  "cdelty": ehdr['CDELT2'], "fovx": fovx, "fovy": fovy}

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
    from astropy.io import ascii
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

