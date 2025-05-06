#!/usr/bin/env python3
# Importing packages
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
iline=li # was li()

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

    

if __name__ == '__main__':
    __spec__ = None
    os.environ['RESPONSE'] = '/Users/souvikb/MUSE_outputs/response'
    os.chdir('/Users/souvikb/MUSE_outputs/')
    vdem_dir = '/Users/souvikb/MUSE_outputs/'
    files = glob.glob(os.path.join(vdem_dir,'vdems/*'))
    print(f'*** Loading {files[0]} into vdem')
    vdem = xr.open_zarr(files[0]).compute()

    # User input for the date of observation #
    date = input("Enter the date of observation (YYYY-MM-DD): ")
    date_obj = Time(date,format='isot',scale='utc') # astropy time object
    # Extract the last two digits of the year
    last_two_digits = str(date_obj.datetime.year)[-2:]
    month_number = date_obj.datetime.month  # Extract the month number (e.g., 5 for May)
    month_name = calendar.month_name[month_number]  # Convert to full month name

    # Check if the response function for that year exists in your system. 
    # We agreed to use the same response function for all observations in a given year.
    response_pattern = os.path.join(os.environ['RESPONSE'], f'swapped_aia_resp_DN_*{last_two_digits}.zarr') # Just checks for the response function for the year
    matching_files = glob.glob(response_pattern)# checks if any file in the above directory matches the pattern
    # If the response function for that year exists, it will be used.   
    # If not, it will create a new response function for that year.
    if matching_files:
        print(f"Response function(s) found for the year: {last_two_digits}")
        response_file = matching_files[0] # because the glob returns a list while the read_response function takes a string. 
        response_all = read_response(response_file, logT=vdem.logT,gain=np.ones((4))).compute() # simply read the response functions
    else:
        print(f"Creating a new AIA response function for the year: {last_two_digits}")
        save_response = True
        use_QS_bands = True
        print(f"Response function date set to {parse_time(date).strftime('%d-%b-%Y')}")
        aia_goes_lines = ['AIA 94', 'AIA 131', 'AIA 171', 'AIA 193', 
                        'AIA 211', 'AIA 304','AIA 335', 
                        'GOES 15 1-8', 'GOES 15 0.5-4']
        # Channels (lines/bands) relevant for (in this case) MUSE QS validation
        if use_QS_bands:
            lines = aia_goes_lines[1:5]
            bands = [int(s) for s in " ".join(lines).split("AIA ")[1:5]]
        else:
            print(" You should set use_QS_bands to true unless you know what you are doing!!!")
            # sys.exit()
        
            #  Temperature limits, abundance, pressure, and pixel size
        #  NB note that available abundance files depend on Chianti version!
        units = 'DN'
        tmax=7.6
        tmin=4.4
        tbin=0.1
        abund = "sun_coronal_2021_chianti"
        abund = "sun_photospheric_2011_caffau"
        abund = "sun_photospheric_2021_asplund"
        pres = 3e15
        dx_pix=0.6
        dy_pix=0.6
        zarr_file = f'aia_resp_{units}_{parse_time(date).strftime("%y")}.zarr'
        for line,band in zip(lines,bands):
            print(f'*** Computing {units} response function for {line} date {parse_time(date).strftime("%b%Y")}')
            ch = Channel(band*u.angstrom)
            correction_table = get_correction_table("jsoc")
            resp_band = ch.wavelength_response(obstime=parse_time(date),correction_table=correction_table)
            eff_xr = create_eff_area_xarray(resp_band.value, ch.wavelength.value, [ch.channel.value])
            line_list = create_resp_line_list(eff_xr, temin=10**tmin, abundance=abund, eDensity=3e10, num_slits=1)
            line_list = line_list.sortby(line_list.resp_func)
            line_list = line_list.sel(trans_index = line_list.trans_index[::-1][:350].data)  
            ''' Important!! Viggo considers here only 350 lines.
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
        response_all = response_all.assign_coords(line = ("band",['AIA '+f'{int(s)}' for s in response_all.band.data]))
        response_all = response_all.compute()
        zarr_file = os.path.join(os.environ['RESPONSE'],zarr_file)
        response_all.to_zarr(zarr_file) # Saved to the .zarr file

    aia_resp = response_all.copy() # This is the response function to be used for the remaining of the year.

    vdem_cut = vdem.sel(logT=aia_resp.logT, method = "nearest")
    vdem_cut = vdem_cut.compute()
    # vdem_cut
    muse_AIA = vdem_synthesis(vdem_cut.sum(dim=["vdop"]), aia_resp, sum_over=["logT"]) #Synthesis AIA observations using the response function and VDEM
    # # print(muse_AIA)
    # # plt.figure(figsize=(10, 5))
    # muse_AIA.flux.sel(line='AIA 171').T.plot(cmap='inferno', norm=colors.PowerNorm(0.3))
    # plt.show()

    print('Creating historgrams from the synthesized AIA data')
    # The following are hard coded for now #
    code = 'MURaM'
    snapname = 'Plage'
    snap = 317600
    ## Creating the figure
    fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(5,10),sharex=True)
    ax.ravel()
    aialine = ['AIA 131','AIA 171','AIA 193','AIA 211']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    num_bins=60 ## to be fixed
    for idx_line, line in enumerate(aialine):
        synth_aiadata = muse_AIA.flux.sel(line=line).to_numpy()
        bins = np.linspace(muse_AIA.flux.sel(line=line).min(), muse_AIA.flux.sel(line=line).max(), num_bins)
        hist, bin_edges = np.histogram(synth_aiadata, bins=bins)
        ax[0].hist(synth_aiadata.ravel(), bins= bins,histtype='step', label=line,
            weights=np.ones(len(synth_aiadata.ravel())) / len(synth_aiadata.ravel()))
        # ax[0].set_xlim([1,15000])
        ax[0].legend(loc = 'upper right')
        ax[0].set_xscale('log')
        ax[0].set_ylabel('Fraction')
        
        ax[1].hist(synth_aiadata.ravel(), bins= bins,histtype='step', cumulative=True, label=line,
            weights=np.ones(len(synth_aiadata.ravel())) / len(synth_aiadata.ravel()))
        ax[1].set_xlabel(r'Intensity [DNs$^{-1}$]')
        ax[1].set_xscale('log')
        ax[1].set_ylabel('ECDF')
        ax[1].set_xlim([1,15000])
        ax[1].axvline(x=np.mean(synth_aiadata.ravel()),color=colors[idx_line],lw=1,ls='--')
        
    fig.suptitle(f'{code} simulation: {snapname} snap {snap:03d}',y=0.98)
    plt.tight_layout()
    plt.show()

    # time.sleep(2) # Wait for the user to realize that they have to input whether to save the histograms or not. hence 5 secs
    save_hist = input("Do you want to save the hisgtograms? (True/False): ")
    # Ensure the directory exists
    output_sims_dir = './pipeline_figs_sims'
    os.makedirs(output_sims_dir, exist_ok=True)  # Create the directory if it doesn't exist
    if save_hist == 'True':
        print('Saving the histograms')
        fig.savefig(os.path.join(output_sims_dir,f"{snapname}_{snap}_hist_vdem.png"),dpi=300,bbox_inches='tight')
        plt.close(fig)


    ### EIS and AIA data analysis
    Region = 'Plage'
    #### -------- Date to be modified by the users ------####
    data_basedir = os.path.join(os.environ['HOME'], 'MUSE_outputs', 'EIS_IRIS_QS_obs', 'Plage_datasets')
    obs_dates = ascii.read(data_basedir+'/plage_obs.txt') 
    obs_dates.add_index("date_begin_EIS")
    for date in obs_dates["date_begin_EIS"]:
        date_begin_EIS = obs_dates.loc[date]["date_begin_EIS"]
        cutouts_data_path = os.path.join(data_basedir,"SDO_EIS_cutouts_"+date_begin_EIS)
        cutout_files = sorted(glob.glob(os.path.join(cutouts_data_path, 'cutout_*.fits')))
        print('The cutout files are:\n' + '\n'.join(cutout_files))
        AIA_DATA = {} # Initialize the dictionary

        # Dynamically calculate figure size based on the number of subplots and aspect ratio
        for idx_line, line in enumerate(aialine):
            AIA_DATA[line] = sunpy.map.Map(cutout_files[idx_line])
        # Number of panels
        n_lines = len(aialine)

        # Get one map to measure aspect ratio
        sample_map = AIA_DATA[aialine[0]]
        ny, nx = sample_map.data.shape
        aspect = ny / nx  # height / width

        # Set figure size dynamically
        panel_width = 4  # good width per panel (in inches)
        fig_width = panel_width * n_lines
        fig_height = panel_width * aspect
        fig = plt.figure(figsize=(fig_width, fig_height))

        for idx_line, line in enumerate(aialine):
            AIA_DATA[line] = sunpy.map.Map(cutout_files[idx_line])
            exp_time = AIA_DATA[line].exposure_time
            data = AIA_DATA[line]/exp_time
            ax = fig.add_subplot(1, n_lines, idx_line + 1, projection=AIA_DATA[line].wcs)
            ##Plotting the AIA cutouts
            im = data.plot(axes=ax, vmin=0)
            # ax.set_title(data.wavelength.to_string('latex'), fontsize=10)
            ax.grid(False)
            lon = ax.coords[0]
            lat = ax.coords[1]
            lon.set_ticks_position('b')      # ticks on bottom
            lon.set_ticklabel_position('b')  # bottom

            lat.set_ticks_position('l')      # ticks on left
            lat.set_ticklabel_position('l')  # left
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(direction='out')
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('right')
            cbar.ax.yaxis.tick_right()
            cbar.set_label('DN/s')

        # fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        plt.tight_layout()
        plt.show()
        save_cutout = input("Do you want to save the cutouts? (True/False): ")
        # Change the working directory to the cutouts data path and ensure directory exists
        os.chdir(cutouts_data_path)
        output_dir = './pipeline_figs'
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        if save_cutout == 'True':
            print('Saving the cutouts')
            fig.savefig(os.path.join(output_dir,f"{date}_cutouts.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)
        else:
            print('Cutouts not saved')
            plt.close(fig)

        fig, ax = plt.subplots(2,2,figsize=(10,10))

    ## ------ Comparison between given AIA lines and synthesized AIA data ------ ##
        for i,line in enumerate(aialine):
            exp_time = AIA_DATA[line].exposure_time
            data = np.ravel(AIA_DATA[line].data/exp_time)
            bins = np.linspace(data.min(), data.max(), num_bins)
            hist, bin_edges = np.histogram(data, bins=bins)
            ax[i//2][i%2].hist(data, bins=bins, label=f'AIA {Region}',cumulative=True, histtype='step',
                weights=np.ones(len(data)) / len(data),color=colors[i])
            ax[i//2][i%2].axvline(x=np.mean(data).to_value(),color=colors[i],lw=1,ls='-')
            # 
            bf = muse_AIA.flux.sel(line=line).to_numpy()
            bins = np.linspace(bf.min(), bf.max(), num_bins)
            hist, bin_edges = np.histogram(bf.ravel(), bins=bins)
            ax[i//2][i%2].hist(bf.ravel(), bins=bins, label=f'{code} {Region}',cumulative=True, histtype='step', 
                    weights=np.ones(len(bf.ravel())) / len(bf.ravel()),color='tab:purple',ls='-.')
            ax[i//2][i%2].axvline(x=np.mean(bf.ravel()),color='tab:purple',lw=1,ls='-.')
            ax[i//2][i%2].set_xlabel(fr'{line} Intensity [DNs$^{-1}$]')
            if line == 'AIA 171':
                ax[i//2][i%2].set_xlim([1,10000])
            if line == 'AIA 193':
                ax[i//2][i%2].set_xlim([1,10000])
            if line == 'AIA 131':
                ax[i//2][i%2].set_xlim([1,1000])
            if line == 'AIA 211':
                ax[i//2][i%2].set_xlim([1,3000])
            ax[i//2][i%2].set_xscale('log')
            ax[i//2][i%2].set_ylabel('ECDF')
            ax[i//2][i%2].legend(loc='lower right')
            ax[i//2][i%2].axhline(y=0.25,color='black',ls='--')
            ax[i//2][i%2].axhline(y=0.75,color='black',ls='--')

        date = AIA_DATA[line].date.strftime('%Y-%m-%d')
        title = f"AIA {Region} {date} vs {code} {snapname} snap {snap:03d}"
        fig.suptitle(title,y=0.98)
        plt.tight_layout()
        plt.show()
        save_hist = input("Do you want to save the histograms? (True/False): ")
        # Change the working directory to the cutouts data path and ensure directory exists
        os.chdir(cutouts_data_path)
        output_dir = './pipeline_figs'  
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        if save_hist == 'True':
            print('Saving the histograms')
            fig.savefig(os.path.join(output_dir,f"{date}_comparison_hist.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)
        else:
            print('Histograms not saved')
            plt.close(fig)

        ## ------ EIS data -------------##
        # if __name__ == '__main__':
        #     eis_main()

        downloaded_data_h5 = glob.glob(cutouts_data_path+'/*.data.h5')
        downloaded_head_h5 = glob.glob(cutouts_data_path+'/*.head.h5')
        data = eispac.read_cube(downloaded_data_h5[0])
        headers = eispac.read_cube(downloaded_head_h5[0])
        
        eis_lines = [195.120,256.320, 284.160] # Fe XII and Fe XV
        target_template_name = ['fe_12_195_119.2c.template.h5', 'he_02_256_317.2c.template.h5','fe_15_284_160.2c.template.h5']
        eis_cmap = ['sdoaia193','sdoaia304','sohoeit284']
        for target_template, eis_spect in zip(target_template_name,eis_lines):
            if wavelength_in_cube(downloaded_data_h5[0], str(eis_spect)):
                data_cube = eispac.read_cube(downloaded_data_h5[0], eis_spect, apply_radcal=True)
                template_list = eispac.match_templates(data_cube)
                index = next(i for i, path in enumerate(template_list) if path.name == target_template)
                shutil.copy(template_list[index], cutouts_data_path)
                template = glob.glob(os.path.join(cutouts_data_path,target_template))#'fe_12*.template.h5'))
                template_filepath = template[0]
                # print(template_filepath)
                # read fit template
                tmplt = eispac.read_template(template_filepath)
                # Read spectral window into an EISCube
                data_cube = eispac.read_cube(downloaded_data_h5[0], tmplt.central_wave,apply_radcal=True)
                fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu='max')
                inten_map = fit_res.get_map(component=0, measurement='intensity')
                vel_map = fit_res.get_map(component=0, measurement='velocity')
                # wd_map = fit_res.get_map(component=0, measurement='width')
                    # Get one map to measure aspect ratio
                sample_map = inten_map  # or any map with the correct shape
                ny,nx = sample_map.data.shape
                aspect = ny / nx  # height / width

                panel_width = 4  # inches per panel (adjust as needed)
                fig_width = panel_width * 2  # two panels side by side
                fig_height = fig_width * aspect/8.  # maintain aspect             
                fig = plt.figure(figsize=(fig_width, fig_height)) 
                if target_template == 'fe_12_195_119.2c.template.h5':
                    ax1 = fig.add_subplot(121,projection=inten_map)
                    im=inten_map.plot(axes=ax1, cmap = eis_cmap[0])
                    fig.colorbar(im,extend='both',fraction=0.05,pad=0.05)
                    ax1.set_title(f'Intensity {eis_spect}'+ ' [erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]',size=10)
                    ax1.grid(False)
                elif target_template == 'he_02_256_317.2c.template.h5':
                    ax1 = fig.add_subplot(121,projection=inten_map)
                    im=inten_map.plot(axes=ax1, cmap = eis_cmap[1])
                    fig.colorbar(im,extend='both',fraction=0.05,pad=0.05)
                    ax1.set_title(f'Intensity {eis_spect}'+ ' [erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]',size=10)
                    ax1.grid(False)
                else:
                    ax1 = fig.add_subplot(121,projection=inten_map)
                    im=inten_map.plot(axes=ax1, cmap = eis_cmap[2])
                    fig.colorbar(im,fraction=0.05,pad=0.05)
                    ax1.set_title(f'Intensity {eis_spect}'+ ' [erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]',size=10)
                    ax1.grid(False)

                vel_map.plot_settings["norm"]=None
                ax2 = fig.add_subplot(122,projection=vel_map)
                im=vel_map.plot(axes=ax2,vmax=10,vmin=-10,cmap='coolwarm')
                fig.colorbar(im,extend ='both', fraction=0.046,pad=0.05)
                ax2.set_title(r'Velocity [kms$^{-1}$]',size=10)
                ax2.grid(False)
                plt.tight_layout()
                plt.show()
                save_eis = input("Do you want to save the EIS maps? (True/False): ")
                # Ensure the directory exists
                os.chdir(cutouts_data_path)
                output_dir = './pipeline_figs'
                os.makedirs(output_dir, exist_ok=True)
                # Create the directory if it doesn't exist
                if save_eis == 'True':
                    print('Saving the EIS maps')
                    fig.savefig(os.path.join(output_dir,f"{date}_EIS_{eis_spect}.png"),dpi=300,bbox_inches='tight')
                    plt.close(fig)
                else:
                    print('EIS maps not saved')
                    plt.close(fig)
            else:
                print(f"Skipping wavelength {eis_spect} Å: Not found in {downloaded_data_h5[0]}")
                # save_eis = input("Do you want to save the EIS maps? (True/False): ")

        ## Plotting the HMI LOS flux density cutouts
        # First grab the cutout region from the full disk HMI data
        bottom_left = SkyCoord(data.meta['extent_arcsec'][0]*u.arcsec, data.meta['extent_arcsec'][2]*u.arcsec, obstime=data.meta['mod_index']['date_obs'], observer="earth", frame="helioprojective")
        top_right = SkyCoord(data.meta['extent_arcsec'][1]*u.arcsec, data.meta['extent_arcsec'][3]*u.arcsec, obstime=data.meta['mod_index']['date_obs'], observer="earth", frame="helioprojective")

        ## Load the HMI data as sunpy map
        hmi_map = sunpy.map.Map(cutouts_data_path+'/*magnetogram.fits') ## This will  be rotated by 90 degrees as usual
        ## Load an AIA map to get the WCS
        aia_map_fdisk = sunpy.map.Map(cutouts_data_path+'/*.193.image_lev1.fits')
        out_hmi = hmi_map.reproject_to(aia_map_fdisk.wcs) ## actual reprojection from HMI to AIA based on WCS headers
        cutout_hmi_aligned = out_hmi.submap(bottom_left,top_right=top_right) ## this is the cutout 

        ## Now dynamically calculate figure size based on the number of subplots and aspect ratio
        ny, nx = cutout_hmi_aligned.data.shape
        aspect = ny / nx
        panel_width = 4
        fig_width = panel_width * 2
        fig_height = aspect*fig_width*0.9
        cutout_hmi_aligned.plot_settings["norm"]=None

        fig= plt.figure(figsize=(fig_width,fig_height))
        # ax.ravel()
        ax = fig.add_subplot(2, 2, 1, projection=cutout_hmi_aligned.wcs)
        im = cutout_hmi_aligned.plot(axes=ax, vmax=0.3*np.max(cutout_hmi_aligned.data), vmin=-0.3*np.max(cutout_hmi_aligned.data))
        ax.grid(False)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticks_position('b')      # ticks on bottom
        lon.set_ticklabel_position('b')  # bottom

        lat.set_ticks_position('l')      # ticks on left
        lat.set_ticklabel_position('l')  # left
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        cbar = fig.colorbar(im, cax=cax,extend='both')
        cbar.ax.tick_params(direction='out')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.tick_right()
        cbar.set_label(r'B$_{\mathrm{LOS}}$[G]')
        # ax.set_title()


        ax = fig.add_subplot(2, 2, 2, projection=cutout_hmi_aligned.wcs)
        abs_data = np.abs(cutout_hmi_aligned.data) 
        abs_map = sunpy.map.Map(abs_data, cutout_hmi_aligned.meta)
        im = abs_map.plot(axes=ax, clip_interval=(1, 99.99)*u.percent,cmap='YlGnBu')
        ax.grid(False)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_ticks_position('b')      # ticks on bottom
        lon.set_ticklabel_position('b')  # bottom

        lat.set_ticks_position('l')      # ticks on left
        lat.set_ticklabel_position(' ')  # left
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        cbar = fig.colorbar(im, cax=cax, extend ='both')
        cbar.ax.tick_params(direction='out')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.tick_right()
        cbar.set_label(r'|B$_{\mathrm{LOS}}$| [G]')

        #---- Histograms ------#
        bins = np.linspace(cutout_hmi_aligned.data.min(),cutout_hmi_aligned.data.max(),100)
        data_hmi = cutout_hmi_aligned.data
        ax = fig.add_subplot(2, 2, 3)
        ax.hist(data_hmi.ravel(),bins=bins, histtype='step',weights=np.ones(len(data_hmi.ravel()))/len(data_hmi.ravel()),log=True,color='blue')
        ax.set_ylabel('Fraction')
        ax.set_xlabel(r'B$_{\mathrm{LOS}}$[G]')
        ax.axvline(x=0,ls='-',color='black')
        
        bins = np.linspace(abs_data.min(),abs_data.max(),100)
        # data_hmi = cutout_hmi_aligned.data
        ax = fig.add_subplot(2, 2, 4)
        ax.hist(abs_data.ravel(),bins=bins, histtype='step',weights=np.ones(len(abs_data.ravel()))/len(abs_data.ravel()),log=True,color='blue')
        # ax.set_ylabel('Fraction')
        ax.set_xlabel(r'|B$_{\mathrm{LOS}}$| [G]')
        plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,wspace=0.3,hspace=0.2)
        ax.axvline(x=np.mean(abs_data),ls='--',color='black',label=f'$\mu$: {np.mean(abs_data):.2f} [G]')
        ax.legend(loc='best')
        plt.show()
        save_hmi = input("Do you want to save the HMI maps? (True/False): ")
        # Ensure the directory exists
        os.chdir(cutouts_data_path)
        output_dir = './pipeline_figs'
        os.makedirs(output_dir, exist_ok=True)
        # Create the directory if it doesn't exist 
        if save_hmi == 'True':
            print('Saving the HMI maps')
            fig.savefig(os.path.join(output_dir,f"{date}_HMI.png"),dpi=300,bbox_inches='tight')             
            plt.close(fig)
        else:
            print('HMI maps not saved')
            plt.close(fig)
        # plt.show()    

        ## Plot a fukll disk AIA map and draw the cutout region
        exp_time_fdisk = aia_map_fdisk.exposure_time
        data_fdisk = aia_map_fdisk/exp_time_fdisk
        fig = plt.figure()
        ax = fig.add_subplot(projection=data_fdisk)
        im = data_fdisk.plot(axes=ax, vmin=0)
        ax.grid(False)
        divider= make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(direction='out')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.yaxis.tick_right()
        cbar.set_label('DN/s')
        cutout_hmi_aligned.draw_extent(axes=ax)
        plt.show()
        save_aia_fdsik = input("Do you want to save the AIA maps? (True/False): ")
        # Ensure the directory exists
        os.chdir(cutouts_data_path)
        output_dir = './pipeline_figs'
        os.makedirs(output_dir, exist_ok=True)
        # Create the directory if it doesn't exist
        if save_aia_fdsik == 'True':
            print('Saving the full disk AIA maps')
            fig.savefig(os.path.join(output_dir,f"{date}_AIA_fdisk.png"),dpi=300,bbox_inches='tight')             
            plt.close(fig)
        else:
            print('AIA full disk maps not saved')
            plt.close(fig)