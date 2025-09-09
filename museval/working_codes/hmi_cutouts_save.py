import eispac
import numpy as np
import sunpy.map
import sunpy.visualization.colormaps as cm
from sunpy.time import parse_time
from tqdm.auto import tqdm
 # import matplotlib.pyplot as plt  # Not used
import glob
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u

def read_eis_cutouts(cutouts_path):
    """
    Reads EIS cutout data from the specified path.

    Parameters:
    -----------
    cutouts_path : str
        Path to the directory containing EIS cutout data files.

    Returns:
    --------
    eis_data : list
        List of EIS data arrays.
    eis_headers : list
        List of EIS header information.
    """
    downloaded_data_h5 = glob.glob(cutouts_path + '/*.data.h5')
    downloaded_head_h5 = glob.glob(cutouts_path + '/*.head.h5')
    if downloaded_data_h5:
        eis_data = eispac.read_cube(downloaded_data_h5[0])
        eis_headers = eispac.read_cube(downloaded_head_h5[0])
    else:
        print(f"No EIS cutout data found in {cutouts_path}")
        return [], []

    return eis_data, eis_headers

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
    aia_map_fdisk = sunpy.map.Map(aia_files)
    out_hmi = hmi_map.reproject_to(aia_map_fdisk.wcs)
    for eis_data in eis_data_list:
        try:
            bottom_left = SkyCoord(
                eis_data.meta['extent_arcsec'][0]*u.arcsec,
                eis_data.meta['extent_arcsec'][2]*u.arcsec,
                obstime=eis_data.meta['mod_index']['date_obs'],
                observer="earth",
                frame="helioprojective"
            )
            top_right = SkyCoord(
                eis_data.meta['extent_arcsec'][1]*u.arcsec,
                eis_data.meta['extent_arcsec'][3]*u.arcsec,
                obstime=eis_data.meta['mod_index']['date_obs'],
                observer="earth",
                frame="helioprojective"
            )
            print(f"bottom_left: {bottom_left}")
            print(f"top_right: {top_right}")
            cutout_hmi_aligned = out_hmi.submap(bottom_left, top_right=top_right)
            print(f"cutout_hmi_aligned shape: {cutout_hmi_aligned.data.shape if hasattr(cutout_hmi_aligned, 'data') else 'N/A'}")
            if hasattr(cutout_hmi_aligned, 'data') and cutout_hmi_aligned.data.size == 0:
                print("Warning: Cutout is empty, not saving.")
                continue
            output_file = os.path.join(output_dir, f"Mag_Cutout_{eis_data.meta['mod_index']['date_obs']}_magnetogram.fits")
            cutout_hmi_aligned.save(output_file, overwrite=True)
            print(f"Saved HMI cutout to {output_file}")
        except Exception as e:
            print(f"Error processing EIS data: {e}")

def main():
    import os
    """
    Main function to save the HMI cutouts corresponding to the EIS data for the specified observation dates.
    """
    os.environ['text_files_path'] = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/HOP_307/'
    os.environ['eis_data_path'] = '/Users/souvikb/MUSE_outputs/EIS_IRIS_QS_obs/Plage_datasets/HOP_307/'
    os.environ['vdem_path'] = '/Users/souvikb/MUSE_outputs/vdems/'
    txt_files = glob.glob(os.path.join(os.environ['text_files_path'], "*.txt"))

    for idx_txt, txt_file in tqdm(enumerate(txt_files)):
        obs_dates = ascii.read(txt_file)
        obs_dates.add_index("date_begin_EIS")
        print(f"\nProcessing observation dates from file: {txt_file}")
        for date in obs_dates['date_begin_EIS']:
            date_begin_EIS = obs_dates.loc[date]["date_begin_EIS"]
            print(f'Processing EIS data for date: {date_begin_EIS}')
            cutouts_data_path = os.path.join(os.environ['eis_data_path'], "SDO_EIS_cutouts_"+date_begin_EIS)
            eis_data, eis_headers = read_eis_cutouts(cutouts_data_path)
            if not eis_data:
                print(f"No EIS data found for date: {date_begin_EIS}")
                continue
            save_hmi_c_outs(cutouts_data_path, cutouts_data_path, eis_data)

if __name__ == "__main__":
    main()  
    