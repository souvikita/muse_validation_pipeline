import os

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

######################################################

def files_to_retry(results, download_dir, min_bytes=1024):
    """
    Given a FidoResults `results` and the directory where files are stored,
    return a list of rows for which the local file is missing or too small.
    Determines file identity by inspecting row values.
    Using for EIS and AIA data download from Fido.
    """
    import os
    from sunpy.util import is_complete
    retry = []
    for table in results:
        for row in table:
            remote_str = extract_remote_str(row)
            fname = os.path.basename(remote_str)
            local = os.path.join(download_dir, fname)
            if not is_complete(local, min_bytes=min_bytes):
                retry.append(row)
    return retry
######################################################

def iris_download_file(url_mod, obs_data_dir):
    '''
    Download the JSON file from a given URL.'''
    import requests
    import os
    # os.makedirs(obs_data_dir, exist_ok=True)
    filename = url_mod.split("/")[-1]
    filepath = os.path.join(obs_data_dir, filename)

    response = requests.get(url_mod, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filepath
######################################################