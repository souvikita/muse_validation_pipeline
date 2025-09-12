import os
import numpy as np
import glob
import shutil
import xarray as xr
from astropy.time import Time
import museval
from museval.utils import get_response, find_response, aia_synthesis, wavelength_in_cube
from museval.io import create_session, is_complete
from muse.synthesis.synthesis import transform_resp_units, vdem_synthesis
from matplotlib import colors
import matplotlib.pyplot as plt
from muse.instr.utils import convert_resp2muse_ciresp
import astropy.units as u
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy.time import Time
import sunpy.map
import sunpy.visualization.colormaps as cm
from sunpy.time import parse_time
from dateutil.tz import gettz
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import eispac
from museval.eis_calibration.eis_calib_2023 import calib_2023 as eis_calib_2023
from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings(
    "ignore",
    message="Failed to open Zarr store with consolidated metadata, but successfully read with non-consolidated metadata.*",
    category=RuntimeWarning
) #To ignore this  annoying Zarr warning

if __name__ == '__main__':
    __spec__ = None
    os.environ['RESPONSE'] = '/Users/souvikb/MUSE_outputs/response/'
    os.environ['text_files_path'] = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/HOP_307/'
    os.environ['eis_data_path'] = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/HOP_307/'
    os.environ['vdem_path'] = '/Users/souvikb/MUSE_outputs/vdems/'
    work_dir = '/Users/souvikb/MUSE_outputs/pipeline_figs_sims/'
    ##---- interactively select a VDEM based on your choice. A new dialogue box opens. -----##
    # import tkinter as tk
    # from tkinter import filedialog
    # root = tk.Tk()
    # root.withdraw()  # Hide the root window
    # vdem_file = filedialog.askdirectory(
    #     title="Select a VDEM .zarr file",
    #     initialdir=os.environ['vdem_path']
    # )
    vdem_file = os.path.join(os.environ['vdem_path'], 'vdems_bifrost_164_hion/vdem_164.zarr')
    print("\n Selected folder:", vdem_file)

    txt_files = glob.glob(os.path.join(os.environ['text_files_path'], "*.txt"))
    txt_files = sorted(txt_files)
    # txt_files
    MHD_code, snapname = [s.strip() for s in input("Enter the MHD code and snapname, separated by a comma: ").split(",")]

    for idx_txt, txt_file in tqdm(enumerate(txt_files)):
        obs_dates = ascii.read(txt_file)
        obs_dates.add_index("date_begin_EIS")
        print(f"\nProcessing observation dates from file: {txt_file}")
        for date in tqdm(obs_dates['date_begin_EIS']):
            date_sensitive = True
            unit = 'DN'
            date = Time(date,format='isot',scale='utc') # astropy time object
            print(f'Date Sensitive AIA {unit} Response from {date.strftime("%b %y")}')
            aia_resp = get_response(date = date, channels=[131,171,193,211],save_response=True, units=unit)
            muse_AIA = aia_synthesis(aia_resp, work_dir, vdem_file, swap_dims = False)
            print('Creating historgrams from the synthesized AIA data')
            fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(4,8),sharex=True)
            ax.ravel()
            colors_channel = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            channels = [131,171,193,211]
            num_bins=40
            for i, channel in enumerate([131,171,193,211]):
                synth_aiadata = muse_AIA.flux.sel(channel=channel).to_numpy()
                # bins = np.linspace(muse_AIA.flux.sel(channel=channel).min(), muse_AIA.flux.sel(channel=channel).max(), num_bins)
                min_val = max(muse_AIA.flux.sel(channel=channel).min(), 1e-1)  # avoid log(0)
                max_val = muse_AIA.flux.sel(channel=channel).max()
                bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
                hist, bin_edges = np.histogram(synth_aiadata, bins=bins)

                ax[0].hist(synth_aiadata.ravel(), bins=bins, histtype='step',color=colors_channel[i],
                        label=f'AIA {channel}',weights=np.ones(len(synth_aiadata.ravel())) / len(synth_aiadata.ravel()))

                ax[0].legend(loc = 'best')
                ax[0].set_xscale('log')
                ax[0].set_ylabel('Fraction')

                ax[1].hist(synth_aiadata.ravel(), bins=bins, histtype='step', color=colors_channel[i],cumulative=True,
                        weights=np.ones(len(synth_aiadata.ravel())) / len(synth_aiadata.ravel()))
                ax[1].set_xlabel(r'Intensity [DNs$^{-1}$]')
                ax[1].set_xscale('log')
                ax[1].set_ylabel('ECDF')
                ax[1].axvline(x=np.mean(synth_aiadata.ravel()),color=colors_channel[i],lw=0.5,ls='--',
                            label=f'$\mu_{{{channel}}}$ {np.mean(synth_aiadata.ravel()):.2f}')
                ax[1].set_xlim([1,muse_AIA.flux.max().to_numpy()*1.1])
                ax[1].legend(loc='best', fontsize='small')
            fig.suptitle(f'{MHD_code} simulation: {snapname}',y=0.98,horizontalalignment='center')
            plt.tight_layout()
            # plt.show()
            os.makedirs(f'{work_dir}/{MHD_code}_{snapname}_synth_histograms/', exist_ok=True)
            fig.savefig(f'{work_dir}/{MHD_code}_{snapname}_synth_histograms/histograms.png', dpi=300,bbox_inches='tight')
            plt.close(fig)

            date_begin_EIS = obs_dates.loc[date]["date_begin_EIS"]
            cutouts_data_path = os.path.join(os.environ['eis_data_path'], "SDO_EIS_cutouts_"+date_begin_EIS)
            cutout_files = sorted(glob.glob(os.path.join(cutouts_data_path, 'cutout_*.fits')))
            print('The cutout files are:\n' + '\n'.join(cutout_files))
            if not cutout_files:
                print(f"No cutout files found in {cutouts_data_path}")
                continue
            AIA_DATA = {} # Initialize the dictionary
            # Dynamically calculate figure size based on the number of subplots and aspect ratio
            for idx_line, line in enumerate(channels):
                AIA_DATA[line] = sunpy.map.Map(cutout_files[idx_line])
            # Number of panels
            n_lines = len(channels)
            # Get one map to measure aspect ratio
            sample_map = AIA_DATA[channels[0]]
            ny, nx = sample_map.data.shape
            aspect = ny / nx  # height / width

            # Set figure size dynamically
            panel_width = 4  # good width per panel (in inches)
            fig_width = panel_width * n_lines
            fig_height = panel_width * aspect
            fig = plt.figure(figsize=(fig_width, fig_height))

            for idx_line, line in enumerate(channels):
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
                cbar.set_label(r'DNs$^{-1}$')
            plt.tight_layout()
            # plt.show()
            os.chdir(cutouts_data_path)
            output_dir = './pipeline_figs'
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir,f"{date}_cutouts.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)

            ## ------ Comparison between given AIA lines and synthesized AIA data ------ ##
            region = 'plage'
            fig, ax = plt.subplots(2,2,figsize=(10,10))
            for i,channel in enumerate(channels):
                exp_time = AIA_DATA[channel].exposure_time
                data = np.ravel(AIA_DATA[channel].data/exp_time)
                bf = muse_AIA.flux.sel(channel=channel).to_numpy()
                combined = np.concatenate([data.value, bf.ravel()])
                min_val = max(combined.min(), 1e-1)  # avoiding log(0). This can be nasty otherwise
                max_val = combined.max()
                bins_new = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
                # bins_new = np.linspace(combined.min(), combined.max(), 60)
                hist, bin_edges = np.histogram(data.value, bins=bins_new)
                ax[i//2][i%2].hist(data.value, bins=bins_new, label=f'AIA {region}',cumulative=True, histtype='step',
                                weights=np.ones(len(data.value)) / len(data.value), color=colors_channel[i])
                ax[i//2][i%2].axvline(x=np.mean(data.value),color=colors_channel[i],lw=1,ls='-',
                                    label=f'$\mu_{{\mathrm{{obs}}}}$: {np.mean(data.value):.2f} [DNs$^{-1}$]')

                ax[i//2][i%2].hist(bf.ravel(), bins=bins_new, label=f'{MHD_code} {snapname}',cumulative=True, histtype='step',
                                weights=np.ones(len(bf.ravel())) / len(bf.ravel()),color='tab:purple',ls='-.',lw=2.)
                ax[i//2][i%2].axvline(x=np.mean(bf.ravel()),color='tab:purple',lw=1,ls='-.',label=f'$\mu_{{\mathrm{{syn}}}}$: {np.mean(bf.ravel()):.2f} [DNs$^{-1}$]')
                ax[i//2][i%2].set_xlabel(fr'{channel} Intensity [DNs$^{-1}$]')
                ax[i//2][i%2].set_xlim([1, max_val * 1.1])
                # if channel == 171:
                #     ax[i//2][i%2].set_xlim([1, 1e4])
                # elif channel == 193:
                #     ax[i//2][i%2].set_xlim([1, 1e4])
                # elif channel == 211:
                #     ax[i//2][i%2].set_xlim([1, 3e3])
                # elif channel == 131:
                #     ax[i//2][i%2].set_xlim([1, 1e3])
                ax[i//2][i%2].set_xscale('log')
                ax[i//2][i%2].set_ylabel('ECDF')
                ax[i//2][i%2].legend(loc='lower right')
                ax[i//2][i%2].axhline(y=0.25,color='black',ls='--')
                ax[i//2][i%2].axhline(y=0.75,color='black',ls='--')

            date = AIA_DATA[channel].date.strftime('%Y-%m-%d')
            title = f"AIA {region} {date} vs {MHD_code} {snapname}"
            fig.suptitle(title,y=0.98)
            fig.suptitle(title,y=0.98)
            plt.tight_layout()
            # plt.show()

            os.chdir(cutouts_data_path)
            output_dir = './pipeline_figs'
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir,f"{date}_{MHD_code}_{snapname}_comparison_hist.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)

            # date_begin_EIS = obs_dates.loc[date]["date_begin_EIS"]
            print(f'Processing EIS data for date: {date_begin_EIS}')
            cutouts_data_path = os.path.join(os.environ['eis_data_path'], "SDO_EIS_cutouts_"+date_begin_EIS)
            downloaded_data_h5 = glob.glob(cutouts_data_path+'/*.data.h5')
            downloaded_head_h5 = glob.glob(cutouts_data_path+'/*.head.h5')
            #check if the variables are empty or not before proceeding with the next steps
            if not downloaded_data_h5 or not downloaded_head_h5:
                print(f"No .data.h5 or .head.h5 files found in {cutouts_data_path}. Skipping EIS processing for this date.")
                continue
            else:
                data = eispac.read_cube(downloaded_data_h5[0])
                headers = eispac.read_cube(downloaded_head_h5[0])
                eis_lines = [195.120,197.860, 284.160] # Fe XII, Fe IX and Fe XV
                target_template_name = ['fe_12_195_119.2c.template.h5', 'fe_09_197_862.1c.template.h5','fe_15_284_160.2c.template.h5']
                eis_cmap = ['sdoaia193','Blues_r','sohoeit284']
                eis_list_data = [] # To check how many if not all lines listed above are available in the actual EIS data
                data_intensity_maps = [] # To store the intensity maps for all the available lines in the actual EIS data.
                for target_template, eis_spect in zip(target_template_name,eis_lines):
                    eis_list_data.append(wavelength_in_cube(downloaded_data_h5[0],str(eis_spect)))
                    if wavelength_in_cube(downloaded_data_h5[0], str(eis_spect)):
                        data_cube = eispac.read_cube(downloaded_data_h5[0], eis_spect, apply_radcal=True)
                        template_list = eispac.match_templates(data_cube)
                        index = next(i for i, path in enumerate(template_list) if path.name == target_template)
                        shutil.copy(template_list[index], cutouts_data_path)
                        template = glob.glob(os.path.join(cutouts_data_path,target_template))#'fe_12*.template.h5'))
                        template_filepath = template[0]
                        tmplt = eispac.read_template(template_filepath)
                        data_cube = eispac.read_cube(downloaded_data_h5[0], tmplt.central_wave,apply_radcal=True)
                        fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu='max')
                        inten_map = fit_res.get_map(component=0, measurement='intensity')
                        inten_map = eis_calib_2023(inten_map)
                        data_intensity_maps.append(inten_map.data) # Collecting the intensity arrays for plotting the histograms later
                        vel_map = fit_res.get_map(component=0, measurement='velocity')
                        # wd_map = fit_res.get_map(component=0, measurement='width')
                            # Get one map to measure aspect ratio
                        sample_map = inten_map  # or any map with the correct shape
                        ny, nx = sample_map.data.shape
                        aspect = ny / nx  # height / width
                        panel_width = 4  # inches per panel (adjust as needed)
                        panel_height = panel_width * aspect
                        fig_width = panel_width * 2.4  # extra space for colorbars
                        # fig_width = panel_width * 2  # two panels side by side
                        fig_height = panel_height  # two panels stacked vertically
                        # fig_height = panel_width * aspect/8.  # maintain aspect
                        fig = plt.figure(figsize=(fig_width, fig_height))
                        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.25)
                        ax1 = fig.add_subplot(gs[0], projection=inten_map)
                        if target_template == 'fe_12_195_119.2c.template.h5':
                            # ax1 = fig.add_subplot(121,projection=inten_map)
                            im=inten_map.plot(axes=ax1, cmap = eis_cmap[0])
                            cbar =fig.colorbar(im,extend='both',fraction=0.05,pad=0.05)
                            cbar.ax.set_ylabel(r'[erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')
                            ax1.set_title(f'Fe XII {eis_spect}',size=10)
                            ax1.grid(False)
                        elif target_template == 'fe_09_197_862.1c.template.h5':
                            # ax1 = fig.add_subplot(121,projection=inten_map)
                            im=inten_map.plot(axes=ax1, cmap = eis_cmap[1])
                            cbar=fig.colorbar(im,extend='both',fraction=0.05,pad=0.05)
                            cbar.ax.set_ylabel(r'[erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')
                            ax1.set_title(f'Fe IX {eis_spect}',size=10)
                            ax1.grid(False)
                        else:
                            # ax1 = fig.add_subplot(121,projection=inten_map)
                            im=inten_map.plot(axes=ax1, cmap = eis_cmap[2])
                            cbar=fig.colorbar(im,fraction=0.05,pad=0.05)
                            cbar.ax.set_ylabel(r'[erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')
                            ax1.set_title(f'Fe XV {eis_spect}',size=10)
                            ax1.grid(False)

                        vel_map.plot_settings["norm"]=None
                        ax2 = fig.add_subplot(gs[1],projection=vel_map)
                        im=vel_map.plot(axes=ax2,vmax=10,vmin=-10,cmap='coolwarm')
                        cbar=fig.colorbar(im,extend ='both', fraction=0.046,pad=0.05)
                        cbar.ax.set_ylabel(r'[kms$^{-1}$]')
                        ax2.set_title(r'Core shift',size=10)
                        ax2.grid(False)
                        plt.tight_layout()
                        # plt.show()
                        os.chdir(cutouts_data_path)
                        output_dir = './pipeline_figs'
                        os.makedirs(output_dir, exist_ok=True)
                        print('Saving the EIS maps')
                        fig.savefig(os.path.join(output_dir,f"{date}_EIS_{eis_spect}.png"),dpi=300,bbox_inches='tight')
                        plt.close(fig)
                    else:
                        print(f"Skipping wavelength {eis_spect} Å: Not found in {downloaded_data_h5[0]}")

             # Plotting the EIS intensity histograms
            n_panels_eis_hist = sum(eis_list_data)
            if n_panels_eis_hist == 0:
                print(f"No {eis_lines} found in the EIS data for date {date_begin_EIS}. Skipping histogram plotting.")
                continue
            fig, ax = plt.subplots(nrows=1, ncols=n_panels_eis_hist, figsize=(12, 4.))
            ax = np.atleast_1d(ax)
            eis_hist_bins = 60 # hard coded for now
            for i, (eis_spect, intensity_map) in enumerate(zip(eis_lines, data_intensity_maps)):
                if eis_list_data[i]:
                    # Get the intensity data for the current line
                    intensity_data = intensity_map.ravel()
                    min_int = max(intensity_data.min(), 1e-1)  # avoid log(0)
                    max_int = intensity_data.max()
                    # Create histogram bins
                    bins = np.logspace(np.log10(min_int), np.log10(max_int), eis_hist_bins)
                    # Plot histogram
                    ax[i].hist(intensity_data, bins=bins, histtype='step', cumulative=False,
                                weights=np.ones(len(intensity_data)) / len(intensity_data), color='blue')
                    ax[i].set_xscale('log')
                    ax[i].set_xlabel(r'Intensity [erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')
                    ax[i].set_ylabel('Fraction')
                    ax[i].set_title(f'EIS {eis_spect} {date_begin_EIS}', fontsize=10)
                    ax[i].axvline(x=np.mean(intensity_data), color='red', lw=1, ls='--',
                                label=f'$\mu_{{\mathrm{{EIS}}}}$: {np.mean(intensity_data):.2f} [erg s$^{-1}$ cm$^{-2}$ s$^{-1}$]')
                    ax[i].legend(loc='best', fontsize='small')

                    color='tab:green'
                    axs_twin=ax[i].twinx()
                    axs_twin.hist(intensity_data, bins=bins, histtype='step', cumulative=True,
                                  weights=np.ones(len(intensity_data)) / len(intensity_data), color='green', ls='-')
                    axs_twin.axhline(y=0.25, linestyle='dashed',color='green')
                    axs_twin.axhline(y=0.75, linestyle='dashed',color='green')
                    axs_twin.set_ylabel('ECDF',color='green')
                    axs_twin.tick_params(axis='y', labelcolor=color)
                else:
                    print(f"Skipping histogram for wavelength {eis_spect} Å: Not found in the EIS data.")

            plt.tight_layout()
            os.chdir(cutouts_data_path)
            output_dir = './pipeline_figs'
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir,f"{date}_EIS_intensity_histograms.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)

            print('Now plotting the HMI LOS flux density cutouts for the same date and region...')
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
            # plt.show()
            # save_hmi = input("Do you want to save the HMI maps? (True/False): ")
            # Ensure the directory exists
            os.chdir(cutouts_data_path)
            output_dir = './pipeline_figs'
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir,f"{date}_HMI.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)

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
            # plt.show()
            os.chdir(cutouts_data_path)
            output_dir = './pipeline_figs'
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir,f"{date}_AIA_fdisk.png"),dpi=300,bbox_inches='tight')
            plt.close(fig)
