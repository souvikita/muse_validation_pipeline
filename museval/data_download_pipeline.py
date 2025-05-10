import os
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
# from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch
import sunpy.coordinates  # NOQA
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import eispac
import eispac.net
from eispac.net.attrs import FileType
import glob
import numpy as np
import pdb
from aiapy.psf import deconvolve, psf
from astropy.visualization import ImageNormalize, LogStretch, time_support
from tqdm import tqdm
import astropy.io.ascii as ascii
import pdb

### depending on setup, this routine may not be needed
######################################################
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from parfive import SessionConfig
def create_session(*args,**keywords):
    from aiohttp import ClientSession, TCPConnector
    return ClientSession(connector=TCPConnector(ssl=False))
from parfive import Downloader
dl = Downloader(config=SessionConfig(aiohttp_session_generator=create_session))
######################################################

def is_complete(filepath, min_bytes=1024):
    """
    Return True if `filepath` exists and is at least `min_bytes` long.
    """
    return os.path.exists(filepath) and os.path.getsize(filepath) >= min_bytes

def extract_remote_str(row):
    """
    From an Astropy Table row, find the first column whose value looks like a file path or URL
    (endswith .fits, .h5, .data, .head, etc.), and return it as string.
    """
    for col in row.table.colnames:
        val = row[col]
        # convert bytes to str
        s = val.decode('utf-8') if isinstance(val, (bytes, bytearray)) else str(val)
        if s.lower().endswith(('.fits', '.fit', '.h5', '.hdf5', '.data', '.head', '.txt', '.csv')):
            return s
    # nothing matched
    raise KeyError(f"No remote file-like value in row; columns tried: {row.table.colnames}")


def files_to_retry(results, download_dir, min_bytes=1024):
    """
    Given a FidoResults `results` and the directory where files are stored,
    return a list of rows for which the local file is missing or too small.
    Determines file identity by inspecting row values.
    """
    retry = []
    for table in results:
        for row in table:
            remote_str = extract_remote_str(row)
            fname = os.path.basename(remote_str)
            local = os.path.join(download_dir, fname)
            if not is_complete(local, min_bytes=min_bytes):
                retry.append(row)
    return retry

e_mail = 'sbose925@gmail.com'
#Path where the EIS data would be stored if they don't already exist#
eis_data_path = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/'
obs_dates = ascii.read(eis_data_path+'Plage_observations_viggo.txt') #Plage_observations_viggo.txt is from Viggo 
obs_dates.add_index("date_begin_EIS")

for date in obs_dates["date_begin_EIS"]:
    date_begin_EIS = obs_dates.loc[date]["date_begin_EIS"]
    date_end_EIS = obs_dates.loc[date]["date_end_EIS"]
    print(f"*** Downloading AIA/EIS/HMI data for times {date_begin_EIS} - {date_end_EIS}")

    obs_data_dir = os.path.join(eis_data_path,"SDO_EIS_cutouts_"+date_begin_EIS)
    print(f'    Creating directory {obs_data_dir}')
    os.makedirs(obs_data_dir, exist_ok=True)
    print(f'    Searching for the EIS data for start - end time {date_begin_EIS} - {date_end_EIS}')
    results = Fido.search(a.Time(date_begin_EIS,date_end_EIS),
                      a.Instrument('EIS'),                       
                      a.Physobs.intensity,
                      a.Source('Hinode'),
                      a.Provider('NRL'),                      
                      a.Level('1.5'),
                      FileType('HDF5 data') | FileType('HDF5 header'),
                     )
    if results.file_num == 0:
        print(f'*** Slot, or engineering data: No spectral files found, Cycling')
        continue
    # files = Fido.fetch(results,path =eis_data_path+"SDO_EIS_cutouts_"+date_begin_EIS,downloader=dl)
    to_retry = files_to_retry(results, obs_data_dir, min_bytes=5*1024)
    if not to_retry:
        print(f'    All files already downloaded. Skipping download')
        
    else:
        print(f'    Downloading {len(to_retry)} missing or incomplete files...')
        for row in to_retry:
            Fido.fetch(row, path=obs_data_dir, downloader=dl)



    print('\n Reading EIS data and headers')
    downloaded_data_h5 = glob.glob(eis_data_path+"SDO_EIS_cutouts_"+date_begin_EIS+'/*.data.h5')
    downloaded_head_h5 = glob.glob(eis_data_path+"SDO_EIS_cutouts_"+date_begin_EIS+'/*.head.h5')
    data = eispac.read_cube(downloaded_data_h5[0])
    headers = eispac.read_cube(downloaded_head_h5[0])

    #----- Selecting the extent of the cutout--------- #
    print('\n Selecting the extent of the AIA cutout based on EIS data')
    bottom_left = SkyCoord(data.meta['extent_arcsec'][0]*u.arcsec, data.meta['extent_arcsec'][2]*u.arcsec, obstime=data.meta['mod_index']['date_obs'], observer="earth", frame="helioprojective")
    top_right = SkyCoord(data.meta['extent_arcsec'][1]*u.arcsec, data.meta['extent_arcsec'][3]*u.arcsec, obstime=data.meta['mod_index']['date_obs'], observer="earth", frame="helioprojective")
    cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)


    ##------- Query and order AIA full disk data -------###
    # print('\n Ordering 1 set of AIA and HMI full disk data.')
    aia = (a.jsoc.Series.aia_lev1_euv_12s & a.jsoc.Segment.image & (a.Wavelength(131*u.angstrom) | a.Wavelength(171*u.angstrom) | a.Wavelength(193*u.angstrom) | a.Wavelength(211*u.angstrom) | a.Wavelength(335*u.angstrom)))
    hmi = (a.jsoc.Series.hmi_m_45s & a.jsoc.Segment.magnetogram)
    query = Fido.search(
        a.Time(data.meta['mod_index']['date_obs'],data.meta['mod_index']['date_obs']),
        aia | hmi,
        a.jsoc.Notify(e_mail),
        # cutout, # Normally, including this will download only the cutouts but I would
        # need to deconvolve the data. So need to first download the full-disk data.
    )
    to_retry_aia = files_to_retry(query, obs_data_dir, min_bytes=5*1024)
    if not to_retry_aia:
        print(f'    All AIA/HMI files already downloaded. Skipping download')
        files_AIA_full_disk =[]
    else:
        print(f"    Fetching {len(to_retry_aia)} AIA/HMI files...")
        files_AIA_full_disk = []
        for row in to_retry_aia:
            files_AIA_full_disk += Fido.fetch(row, path=obs_data_dir, downloader=dl)
            
    # files_AIA_full_disk = Fido.fetch(query,path=eis_data_path+"SDO_EIS_cutouts_"+date_begin_EIS) #Will send 6/7 emails. You can ignore!

    ##------ Deconvolution of AIA images------##


    print('\n Reading saved AIA PSFs, deconvolving the downloaded full-disk AIA data and storing the cutouts')
    PSF_aia_team = np.load('/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/AIA_PSFs/aia_psfs_added.npz',allow_pickle=True)

    for aia_file in tqdm(range(len(files_AIA_full_disk)-1)):
        aia_full_disk = sunpy.map.Map(files_AIA_full_disk[aia_file])
        aia_full_disk_deconv = deconvolve(aia_full_disk, psf=PSF_aia_team[PSF_aia_team.files[aia_file]])##Assumes files are stored in the same sequence, i.e. 131, 171..
        aia_cutout = aia_full_disk_deconv.submap(bottom_left,top_right=top_right)
        wavelength_name = str(aia_full_disk.wavelength).split()[0]
        aia_cutout.save(eis_data_path+"SDO_EIS_cutouts_"+date_begin_EIS+'/cutout_AIA_'+wavelength_name+'.fits')


    print("\n *** Done!! Check the cutouts now!")