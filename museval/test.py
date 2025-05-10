#!/usr/bin/env python3
# Importing packages
import sys
import getopt
import os
import numpy as np
import glob
import shutil
import xarray as xr
from helita.sim import bifrost as br
import br_py.tools as btr
from muse.synthesis.synthesis import vdem_synthesis
from muse.utils.utils import read_response
from muse.transforms.transforms import muse_fov
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from iris_lmsalpy import extract_irisL2data as ei
from scipy.interpolate import CubicSpline as spl
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
import sunpy.map
import sunpy.visualization.colormaps as cm
from sunpy.time import parse_time
import scipy
import sys
import time
import bifrost_tools as bt
from bifrost_tools import lines_info as li
from bifrost_tools import atom_format as afmt
iline=li() # was li()
import museval

from dateutil.tz import gettz
from aiapy.calibrate.util import get_correction_table

import astropy.constants as const
import astropy.units as u
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord

from aiapy.psf import deconvolve, psf
from aiapy.response import Channel

from astropy.visualization import quantity_support
from astropy.visualization import ImageNormalize, LogStretch, time_support

import muse as muse
from muse.synthesis.synthesis import vdem_synthesis
from muse.instr.utils import create_eff_area_xarray
from muse.instr.utils import create_resp_func, create_resp_line_list, create_resp_func_ci

import eispac

import warnings
warnings.filterwarnings('ignore',category=UserWarning)
import calendar # to convert the month number to the month name

prog_name   = os.path.basename(sys.argv[0])
prog_ver    = 0.1
verbose     = True

help_message = '''
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   :: Compare synthetic observables from VDEM with AIA/EIS/HMI and IRIS data   ::
   ::                                                                          ::
   :: You should set up two environment variables: RESPONSE and VDEM           ::
   :: where AIA response functions and where simulation VDEMs are located.     ::
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    You can setup some parameters from command line:

    -d / --date    -  date of observation; format YYYY-MM-DD
    -r / --respdir -  response function directory
                      default is content of os.environ["RESPONSE"]
    -v / --vdemdir -  vdem directory
                      default is content of os.environ["VDEM"]
    -w / --workdir -  workdir
                      default is './'

    -h / --help     - Print this help msg
'''

#********* EIS function to check the availability of wavelengts******

def wavelength_in_cube(data_file, target_wave_str):
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

def main(argv = None):

    if argv is None:
        argv = sys.argv
    
    try:
      opts, args = getopt.getopt(argv[1:], "hd:r:v:w:", ["help","date=","respdir=","vdemdir=","workdir="])
    except getopt.GetoptError:
      print(help_message)
      sys.exit(2)

    resp_dir = os.environ['RESPONSE']
    vdem_dir = os.environ['VDEM']
    work_dir = './'
      
    # option processing
    for option, value in opts:
      if option in ("-h", "--help"):
        print (help_message)
        return 0
      if option in ("-d", "--date"):
        date = value
      if option in ("-r", "--respdir"):
        resp_dir = value
      if option in ("-v", "--vdemdir"):
        vdem_dir = float(value)
      if option in ("-w", "--workdir"):
        work_dir = value
    pass

    os.environ['RESPONSE'] = resp_dir
    aia_resp,date_obj = get_response(date)
    muse_AIA = aia_synthesis(aia_resp, work_dir, vdem_dir)
    muse_AIA

if __name__ == '__main__':
    __spec__ = None
    sys.exit(main())
