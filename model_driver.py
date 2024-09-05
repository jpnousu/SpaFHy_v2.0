# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:52:46 2019

@author: khaaahti
"""

import time
import numpy as np
import pandas as pd
from spafhy import SpaFHy
from iotools import initialize_parameters, read_FMI_weather, initialize_netcdf, initialize_netcdf_spinup, write_ncf, write_ncf_spinup
import matplotlib.pyplot as plt
import xarray as xr
import importlib

eps = np.finfo(float).eps

def driver(catchment, create_ncf=False, create_spinup=False, output=True, folder=''):
    """
    Model driver: sets up model, runs it and saves results to file (create_ncf==True)
    or return dictionary of results.
    """

    """ set up model """
    running_time = time.time()

    # load and process parameters
    pgen, pcpy, pbu, pds, cmask, simu_mask, ptop, gisinfo = preprocess_parameters(catchment, folder)

    flatten = True

    if flatten == True:
        pcpy, shapes, indices = flatten_grids(pcpy)
        pbu, _, _ = flatten_grids(pbu)
        pds, _, _ = flatten_grids(pds)
        ptop, _, _ = flatten_grids(ptop)

    # initialize SpaFHy
    spa = SpaFHy(pgen, pcpy, pbu, pds, ptop)

    # read forcing data
    forcing = preprocess_forcing(pgen, simu_mask)

    #if flatten == True:
    #    forcing = flatten_forcing_grids(forcing)

    Nsteps = len(forcing['date'])
    Nspin = (pd.to_datetime(pgen['spinup_end']) - pd.to_datetime(pgen['start_date'])).days + 1

    # results dictionary to accumulate simulation results
    # FOR ONE YEAR AT A TIME
    if create_ncf:
        save_interval = min(pgen['save_interval'], Nsteps - Nspin)
        results = _create_results(pgen, cmask, save_interval)
    else:
        save_interval = Nsteps - Nspin
        results = _create_results(pgen, cmask, Nsteps - Nspin)

    Nsaveresults = list(np.arange(Nspin, Nsteps, save_interval)[1:] - 1)

    # save parameters to results
    results = _append_results('parameters', pcpy['state'], results)
    results = _append_results('parameters', pbu, results)    
    results = _append_results('parameters', pcpy['loc'], results)
    
    if pgen['simtype'] == '2D':
        results = _append_results('parameters', pds, results)
    elif pgen['simtype'] == 'TOP':
        results = _append_results('parameters', ptop, results)


    if create_ncf:
        ncf, outputfile = initialize_netcdf(
                pgen=pgen,
                cmask=cmask,
                filepath=pgen['results_folder'],
                filename=pgen['ncf_file'],
                description=pgen['description'],
                gisinfo=gisinfo)

    if create_spinup:
        ncf_spinup, outputfile_spinup = initialize_netcdf_spinup(
                pgen=pgen,
                cmask=cmask,
                filepath=pgen['results_folder'],
                filename=pgen['ncf_file'],
                description=pgen['description'],
                gisinfo=gisinfo)

    print('*** Running model ***')

    if pgen['simtype'] == '2D':
            print('*** 2D run')
    elif pgen['simtype'] == 'TOP':
            print('*** TOPMODEL run')
    elif pgen['simtype'] == '1D':
            print('*** 1D run')
            
    if pgen['org_drain'] == True:
            print('*** Bucket organic layer drains according to Campbell 1985')
    else:
            print('*** Bucket organic layer as in Launiainen et al., 2019')
        


    interval = 0
    Nsaved = Nspin - 1

    for k in range(0, Nsteps):
        if pgen['simtype'] == '2D':
            deep_results, canopy_results, bucket_results = spa.run_timestep(forcing.isel(date=k))

            deep_results = reshape_to_grids(deep_results, grid_shape=cmask.shape)
            canopy_results = reshape_to_grids(canopy_results)
            bucket_results = reshape_to_grids(bucket_results)

        elif pgen['simtype'] == 'TOP':
            top_results, canopy_results, bucket_results = spa.run_timestep(forcing.isel(date=k))

            top_results = reshape_to_grids(deep_results)
            canopy_results = reshape_to_grids(canopy_results)
            bucket_results = reshape_to_grids(bucket_results)

        elif pgen['simtype'] == '1D':
            canopy_results, bucket_results = spa.run_timestep(forcing.isel(date=k))
            
            canopy_results = reshape_to_grids(canopy_results, shapes, indices)
            bucket_results = reshape_to_grids(bucket_results, shapes, indices)


        if (k >= Nspin):  # save results after spinup done
            if pgen['simtype'] == '2D':
                results = _append_results('deep', deep_results, results, k - Nsaved - 1)
            elif pgen['simtype'] == 'TOP':
                results = _append_results('top', top_results, results, k - Nsaved - 1)
            results = _append_results('canopy', canopy_results, results, k - Nsaved - 1)
            results = _append_results('bucket', bucket_results, results, k - Nsaved - 1)

            if k in Nsaveresults and create_ncf:
                interval += 1
                print('*** Writing results to netCDF4-file, subset %.0f/%.0f ***' % (interval, len(Nsaveresults)+1))
                # save forcing to results
                results = _append_results('forcing', forcing[dict(
                        date=slice(Nsaved + 1, k + 1))], results)
                write_ncf(results=results, ncf=ncf, steps=[Nsaved + 1 - Nspin, k + 1 - Nspin])
                Nsaved = k
    if (create_spinup) and (k == Nsteps - 1):
        write_ncf_spinup(results=results, pgen=pgen, ncf_spinup=ncf_spinup)
        print('*** Writing spinup to netCDF4-file')


    if create_ncf:
        interval += 1
        print('*** Writing results to netCDF4-file, subset %.0f/%.0f ***' % (interval, len(Nsaveresults)+1))
        # save forcing to results
        results = _append_results('forcing', forcing[dict(
                date=slice(Nsaved + 1, k + 1))], results)
        write_ncf(results=results, ncf=ncf, steps=[Nsaved + 1 - Nspin, k + 1 - Nspin])
        ncf.close()
        print('--- Running time %.2f seconds ---' % (time.time() - running_time))
        print('--- Results are in file: ' + outputfile + ' ---')
        if output:
            return outputfile
    else:
        print('--- Running time %.2f seconds ---' % (time.time() - running_time))
        if output:
            return results, spa, pcpy, pbu, ptop, cmask

def preprocess_parameters(catchment, folder=''):
    """
    Reading gisdata if applicable and preprocesses parameters
    """

    from iotools import read_bu_gisdata, read_ds_gisdata, read_cpy_gisdata, read_forcing_gisdata, read_top_gisdata, read_aux_gisdata
    from iotools import preprocess_budata, preprocess_dsdata, preprocess_cpydata, preprocess_topdata
    #from parameters_krycklan import root_properties, org_properties, deep_properties, parameters, root_properties, ptopmodel, auxiliary_grids

    parameters_module = importlib.import_module(f'parameters_{catchment}')

    # Initialize parameters based on the catchment
    initialize_parameters(catchment, folder)

    root_properties = parameters_module.root_properties
    org_properties = parameters_module.org_properties
    deep_properties = parameters_module.deep_properties
    parameters = parameters_module.parameters
    ptopmodel = parameters_module.ptopmodel
    auxiliary_grids = parameters_module.auxiliary_grids

    pgen, pcpy, pbu, pspd = parameters(folder)
    ptop = ptopmodel()
    aux = auxiliary_grids()
    
    # Checking which of the parameters are given as .asc grids
    spatial_pbu = {}
    for key in pbu:
        if isinstance(pbu[key], float):
            spatial_pbu[key] = False
        if isinstance(pbu[key], str):
            spatial_pbu[key] = True

    spatial_pcpy = {}
    for key in pcpy['state']:
        if isinstance(pcpy['state'][key], float):
            spatial_pcpy[key] = False
        if isinstance(pcpy['state'][key], str):
            spatial_pcpy[key] = True

    spatial_pspd = {}
    for key in pspd:
        if isinstance(pspd[key], float):
            spatial_pspd[key] = False
        if isinstance(pspd[key], str):
            spatial_pspd[key] = True

    spatial_ptop = {}
    for key in ptop:
        if isinstance(ptop[key], float):
            spatial_ptop[key] = False
        if isinstance(ptop[key], str):
            spatial_ptop[key] = True

    spatial_aux = {}
    for key in aux:
        if isinstance(aux[key], float):
            spatial_aux[key] = False
        if isinstance(aux[key], str):
            spatial_aux[key] = True

    orgp = org_properties()
    rootp = root_properties()
    deepp = deep_properties()
    gisdata = {}

    gisdata.update(read_aux_gisdata(pgen['gis_folder'], spatial_aux))

    if pgen['spatial_soil']:
        gisdata.update(read_bu_gisdata(pgen['gis_folder'], spatial_pbu=spatial_pbu, mask=pgen['mask']))
        
    if pgen['spatial_cpy']:
        gisdata.update(read_cpy_gisdata(pgen['gis_folder'], spatial_pcpy=spatial_pcpy, mask=pgen['mask']))

    if pgen['spatial_forcing']:
        gisdata.update(read_forcing_gisdata(pgen['gis_folder'], mask=pgen['mask']))
        pgen.update({'forcing_id': gisdata['forcing_id']})

    if pgen['simtype'] == 'TOP':
        gisdata.update(read_top_gisdata(pgen['gis_folder'], spatial_ptop, mask=pgen['mask']))

    if pgen['simtype'] == '2D':
        gisdata.update(read_ds_gisdata(pgen['gis_folder'], spatial_pspd))

    if (pgen['spatial_cpy'] == False and
        pgen['spatial_soil'] == False and
        pgen['spatial_forcing'] == False):
        gisdata = {'cmask': np.ones((1,1))}

    # masking the gisdata according to pgen['mask']
    if pgen['mask'] is not None:
        for key in gisdata:
            if key not in ['xllcorner', 'yllcorner', 'dxy', 'cmask', 'cmask_bi', 'streams', 'lakes']:
                if pgen['mask'] == 'cmask':
                    mask = gisdata['cmask_bi'].copy()
                    gisdata[key] = gisdata[key] * mask
                if isinstance(pgen['mask'], (int, float)):
                    catchment = pgen['mask']
                    mask = gisdata['cmask'].copy()
                    mask[mask != pgen['mask']] = np.nan
                    mask[mask == pgen['mask']] = 1.0
                    gisdata[key] = gisdata[key] * mask
                if pgen['mask'] == 'streams':
                    if pgen['simtype'] != '2D': # making sure streams are not masked if 2D run
                        mask = gisdata['streams'].copy()
                        mask = np.where(mask == 1.0, np.nan, 1.0)
                        gisdata[key] = gisdata[key] * mask
                if pgen['mask'] == 'streams/lakes':
                    if pgen['simtype'] != '2D': # making sure streams are not masked if 2D run
                        mask = gisdata['streams'].copy()
                        mask2 = gisdata['lakes'].copy()
                        mask = np.where(mask == 1.0, np.nan, 1.0)
                        mask2 = np.where(mask2 == 1.0, np.nan, 1.0)
                        mask[~np.isfinite(mask2)] = np.nan
                        gisdata[key] = gisdata[key] * mask
                if pgen['mask'] == 'lakes':
                    mask = gisdata['lakes'].copy()
                    mask = np.where(mask == 1.0, np.nan, 1.0)
                    gisdata[key] = gisdata[key] * mask                        
                if pgen['mask'] == 'cmask/streams':
                    mask = gisdata['cmask_bi'].copy()
                    gisdata[key] = gisdata[key] * mask
                    if pgen['simtype'] != '2D': # making sure streams are not maksed if 2D run       
                        mask2 = gisdata['streams'].copy()
                        mask2 = np.where(mask2 == 1.0, np.nan, 1.0)
                        mask[~np.isfinite(mask2)] = np.nan
                        gisdata[key] = gisdata[key] * mask

    simu_mask = mask.copy()

    budata = preprocess_budata(pbu, spatial_pbu, orgp, rootp, gisdata, pgen['spatial_soil'])

    cpydata = preprocess_cpydata(pcpy, spatial_pcpy, gisdata, pgen['spatial_cpy'])

    if pgen['simtype'] == 'TOP':
        ptop = preprocess_topdata(ptop, spatial_ptop, gisdata, spatial=True)

    if pgen['simtype'] == '2D':
        dsdata = preprocess_dsdata(pspd, spatial_pspd, deepp, gisdata, pgen['spatial_soil'])
    else:
        dsdata = pspd.copy() # dummy

    gisinfo = {}
    gisinfo['xllcorner'] = gisdata['xllcorner']
    gisinfo['yllcorner'] = gisdata['yllcorner']
    gisinfo['dxy'] = gisdata['dxy']

    # overwrites the state variables with the last timestep of spinup file (if given)
    try:
        spinup = xr.open_dataset(pgen['spinup_file'])
        cpydata['w'] = np.array(spinup['canopy_water_storage'][-1]) * 1e-3
        cpydata['swe'] = np.array(spinup['canopy_water_storage'][-1]) * 1e-3
        soildata['top_storage'] = np.array(spinup['bucket_water_storage_top'][-1]) * 1e-3
        soildata['root_storage'] = np.array(spinup['bucket_water_storage_root'][-1]) * 1e-3
        if pgen['simtype'] == '2D':
            soildata['ground_water_level'] = np.array(spinup['soil_ground_water_level'][-1])
        elif pgen['simtype'] == 'TOP':
            ptop['so'] = np.array(spinup['top_saturation_deficit'][-1])

        print('*** State variables assigned from ', pgen['spinup_file'],  '***')
    except:
        print('*** State variables assigned from parameters.py ***')

    return pgen, cpydata, budata, dsdata, gisdata['cmask'], simu_mask, ptop, gisinfo

def preprocess_forcing(pgen, simu_mask, flatten=False):
    """
    Reads forcing file(s) based on indices in pgen['forcing_id']
    Creates xarray dataset of forcing data
    """

    import xarray as xr

    indices = np.array(pgen['forcing_id'],ndmin=2)

    variables = ['doy',
                 'air_temperature',
                 'vapor_pressure_deficit',
                 'global_radiation',
                 'par',
                 'precipitation',
                 'CO2',
                 'relative_humidity',
                 'wind_speed']

    dims = ['date','i','j']
    dates = pd.date_range(pgen['start_date'], pgen['end_date']).tolist()
    empty_array = np.ones((len(dates),) + np.shape(indices)) * np.nan

    ddict = {var: (dims, empty_array.copy()) for var in variables}

    for index in np.unique(indices):
        if np.isnan(index):
            break
        fp = pgen['forcing_file'].replace('[forcing_id]',str(int(index)))
        df = read_FMI_weather(pgen['start_date'],
                              pgen['end_date'],
                              sourcefile=fp)
        ix = np.where(indices==index)
        for var in variables:
            ddict[var][1][:,ix[0],ix[1]] = np.matmul(
                    df[var].values.reshape(len(dates),1),np.ones((1,len(ix[0]))))
    ds = xr.Dataset(ddict, coords={'date': dates})

    # Multiply each variable by the simulation mask
    #for var in variables:
    #    ds[var] = ds[var] * simu_mask

    if ds.dims['i'] == 1 and ds.dims['j'] == 1:
        ds = ds.squeeze(dim=['i', 'j'])    


    return ds

def _create_results(pgen, cmask, Nsteps):
    """
    Creates results dictionary to accumulate simulation results
    """
    i, j = np.shape(cmask)

    results = {}

    for var in pgen['variables']:

        var_shape = []
        var_name = var[0]

        if var_name.split('_')[0] != 'parameters':
            var_shape.append(Nsteps)
        if (var_name.split('_')[0] != 'forcing' or
            pgen['spatial_forcing'] == True):
            if (var_name.split('_')[0] != 'top'):
                var_shape.append(i)
                var_shape.append(j)
            if (var_name.split('_')[0] == 'top' and var_name.split('_')[1] == 'local'):
                var_shape.append(i)
                var_shape.append(j)

        results[var_name] = np.full(var_shape, np.NAN)

    return results


def _append_results(group, step_results, results, step=None):
    """
    Adds results from each simulation steps to temporary results dictionary
    """

    results_keys = results.keys()

    for key in results_keys:
        var = key.split('_',1)[-1]

        if var in step_results and key.split('_',1)[0] == group:
            if group == 'forcing':
                res = step_results[var].values
            else:
                res = step_results[var]
            if step==None:
                results[key] = res
            else:
                results[key][step] = res
    return results

'''
def flatten_grids(in_dict):
    """
    Flattens 2D arrays in the input dictionary and removes np.nan values,
    handling nested dictionaries and preserving the original structure.
    
    Parameters:
    in_dict (dict): A dictionary that may contain nested dictionaries and numpy arrays.
    
    Returns:
    dict: A dictionary with flattened and filtered arrays, preserving the nested structure.
    """
    def process_dict(d):
        # Create a new dictionary to hold the flattened and filtered values
        out_dict = {}

        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                out_dict[key] = process_dict(value)
            elif isinstance(value, np.ndarray):
                if value.ndim == 2:  # 2D array
                    # Flatten and filter np.nan values
                    flattened = value.flatten()
                    filtered = flattened[~np.isnan(flattened)]
                    out_dict[key] = filtered
                elif value.ndim == 1:  # 1D array
                    # Filter np.nan values
                    out_dict[key] = value[~np.isnan(value)]
                else:
                    # Keep other data types as is
                    out_dict[key] = value
            else:
                # Keep non-numpy types as is
                out_dict[key] = value

        return out_dict

    return process_dict(in_dict)

'''

def flatten_grids(in_dict):
    """
    Flattens 2D arrays in the input dictionary and removes np.nan values,
    while returning the original grid shape and indices for reshaping.
    
    Parameters:
    in_dict (dict): A dictionary that may contain nested dictionaries and numpy arrays.
    
    Returns:
    tuple: A tuple containing:
        - A dictionary with flattened and filtered arrays.
        - A tuple with the original shape of the first 2D array found.
        - A numpy array with indices of non-nan values of the first 2D array.
    """
    flattened_dict = {}
    shapes = None  # To store the shape of the first 2D array
    indices = None  # To store indices of non-nan values

    for key, value in in_dict.items():
        if isinstance(value, dict):
            # Process nested dictionaries by calling the function recursively
            nested_flattened, nested_shapes, nested_indices = flatten_grids(value)
            flattened_dict[key] = nested_flattened
            
            # Only set `shapes` if it hasn't been set yet
            if shapes is None and nested_shapes is not None:
                shapes = nested_shapes

            # Only set `indices` if it hasn't been set yet
            if indices is None and nested_indices is not None:
                indices = nested_indices

        elif isinstance(value, np.ndarray):
            if value.ndim == 2:
                # Save the shape of the first 2D array found
                if shapes is None:
                    shapes = value.shape
                if indices is None:
                    indices = np.argwhere(~np.isnan(value))

                # Flatten the 2D array and remove np.nan values
                flattened = value.flatten()
                mask = ~np.isnan(flattened)
                filtered = flattened[mask]
                
                # Save filtered array
                flattened_dict[key] = filtered
            
            elif value.ndim == 1:
                # Remove np.nan values from 1D arrays
                mask = ~np.isnan(value)
                filtered = value[mask]
                
                # Save the filtered array and its indices
                flattened_dict[key] = filtered
            else:
                # For non-1D/2D arrays, just store them as-is
                flattened_dict[key] = value

        else:
            # For non-numpy elements, just store them as-is
            flattened_dict[key] = value
        
    return flattened_dict, shapes, indices

def flatten_forcing_grids(ds):
    """
    Flattens the i, j dimensions of variables in the input xarray dataset into a single dimension
    and removes np.nan values, preserving the original structure of the dataset.
    
    Parameters:
    ds (xarray.Dataset): An xarray dataset with variables that have dimensions time, i, j.
    
    Returns:
    xarray.Dataset: A dataset with flattened and filtered variables, preserving the original structure.
    """
    def process_variable(var):
        # Flatten the i, j dimensions into a single dimension
        flattened = var.stack(space=('i', 'j')).values
        
        # Filter out np.nan values
        filtered = flattened[~np.isnan(flattened)]
        
        return filtered

    # Create a new dataset to hold the flattened and filtered variables
    out_ds = xr.Dataset()

    for var_name, var in ds.data_vars.items():
        if {'i', 'j'}.issubset(var.dims):
            # Process variables with i, j dimensions
            out_ds[var_name] = (['time', 'space'], process_variable(var))
        else:
            # Keep other variables unchanged
            out_ds[var_name] = var

    return out_ds



'''
def reshape_to_grids(in_dict, grid_shape):
    """
    Reshapes 1D arrays in the input dictionary back to the specified grid shape,
    reinserting np.nan where appropriate. If reshaping is not possible, 
    the array is kept as is or treated as a single value if it has only one element.
    
    Parameters:
    in_dict (dict): A dictionary containing 1D numpy arrays (post-computation results).
    grid_shape (tuple): The shape of the 2D grid to reshape to.
    
    Returns:
    dict: A dictionary with reshaped arrays or kept as 1D if reshaping isn't possible.
    """
    reshaped_results = {}
    
    # Calculate the number of elements required for the grid shape
    total_elements = np.prod(grid_shape)

    for key, arr in in_dict.items():
        # If the array length matches the grid's total number of elements, reshape it
        if len(arr) == total_elements:
            reshaped_arr = np.full(grid_shape, np.nan)  # Create an array of np.nan
            reshaped_arr.flat[:len(arr)] = arr  # Place the computed results
            reshaped_results[key] = reshaped_arr
        elif len(arr) == 1:
            # If the array has only one element, treat it as a single value
            reshaped_results[key] = arr[0]
        else:
            # Keep the array as is if reshaping isn't possible
            reshaped_results[key] = arr

    return reshaped_results
'''

def reshape_to_grids(flattened_dict, shapes, indices):
    """
    Reshapes 1D arrays back to their original 2D shapes using the provided
    original shapes and indices. If reshaping is not possible, the array is kept as is.
    
    Parameters:
    flattened_dict (dict): A dictionary with flattened arrays from simulation results.
    shapes (dict): A dictionary with original shapes of the 2D arrays (from flatten_grids).
    indices (dict): A dictionary with indices of non-nan values (from flatten_grids).
    
    Returns:
    dict: A dictionary with reshaped arrays or kept as 1D if reshaping isn't possible.
    """
    reshaped_results = {}

    for key, data in flattened_dict.items():
            
        # Create an array of np.nan with the target shape
        reshaped_arr = np.full(shapes, np.nan)
            
        # Place the data into the reshaped array at the correct indices
        reshaped_arr[indices[:, 0], indices[:, 1]] = data
            
        # Store the reshaped array in the results dictionary
        reshaped_results[key] = reshaped_arr

    return reshaped_results

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='parameter folder', type=str)

    args = parser.parse_args()

    outputfile = driver(create_ncf=True, folder=args.folder)

    print(outputfile)