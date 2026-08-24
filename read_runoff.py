import os
import re
import glob
import pandas as pd
import numpy as np
import xarray as xr


def read_runoff(catchment_no, discharge_folder):
    """
    Read Krycklan discharge data for a given catchment number.
    Returns daily runoff in mm/d as a DataFrame with DatetimeIndex.
    """
    # site-code segment between 'SVB' and '-C{catchment_no}' varies (e.g. '-VAB-C1', '_RIS-C1')
    pattern = os.path.join(discharge_folder, f'SITES_WB-SL-Q_SVB*-C{catchment_no}_*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No discharge file found for catchment {catchment_no} in {discharge_folder}')

    filepath = files[0]

    # Read header lines to extract catchment area and find data start
    area_km2 = None
    skip_rows = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            match = re.search(r'Catchment area\s+([\d.]+)\s*km2', line)
            if match:
                area_km2 = float(match.group(1))
            if line.strip() == '####':
                skip_rows = i + 1
                break

    if area_km2 is None:
        raise ValueError('Catchment area not found in file header')

    df = pd.read_csv(filepath, skiprows=skip_rows, parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')

    # Convert Q from m3/s to mm/d: * 86400 s/d * 1000 mm/m / (area_km2 * 1e6 m2/km2)
    df['Q'] = df['Q'] * 86400 * 1000 / (area_km2 * 1e6)

    df = df[['Q']].resample('D').mean()

    return df

def kge(obs, sim):
    """Kling-Gupta Efficiency. Returns NaN if fewer than 10 overlapping values."""
    df = pd.DataFrame({'obs': obs, 'sim': sim}).dropna()
    if len(df) < 10:
        return np.nan
    r     = np.corrcoef(df['obs'], df['sim'])[0, 1]
    alpha = df['sim'].std() / df['obs'].std()
    beta  = df['sim'].mean() / df['obs'].mean()
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def rmse(obs, sim):
    """Root Mean Square Error."""
    df = pd.DataFrame({'obs': obs, 'sim': sim}).dropna()
    if len(df) < 10:
        return np.nan
    return np.sqrt(((df['sim'] - df['obs']) ** 2).mean())

def mbe(obs, sim):
    """Mean Bias Error (positive = model overestimates)."""
    df = pd.DataFrame({'obs': obs, 'sim': sim}).dropna()
    if len(df) < 10:
        return np.nan
    return (df['sim'] - df['obs']).mean()


def extract_discharge_results_2D(nc_file_path):
    results = xr.open_dataset(nc_file_path)
    surface_runoff = results['bucket_surface_runoff'].mean(dim=['lat', 'lon']).to_pandas()
    netflow_to_ditch = results['deep_netflow_to_ditch'].mean(dim=['lat', 'lon']).to_pandas()
    netflow_to_lake = results['deep_netflow_to_lake'].mean(dim=['lat', 'lon']).to_pandas()
    return surface_runoff, netflow_to_ditch, netflow_to_lake