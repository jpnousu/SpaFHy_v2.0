# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:52:46 2019

@author: khaaahti
"""

import time
import numpy as np
import pandas as pd
from spafhy import SpaFHy
from iotools import initialize_parameters, initialize_netcdf, initialize_netcdf_spinup, write_ncf, write_ncf_spinup
import matplotlib.pyplot as plt
import xarray as xr
import importlib
import pprint
import os
from multiprocessing import Pool, cpu_count

eps = np.finfo(float).eps

def worker(catch, catchment, create_ncf, create_spinup, output, folder):
    print(f'Simulating catchment: {catch}')
    driver(catchment, catch, create_ncf=create_ncf, create_spinup=create_spinup, output=output, folder=folder)

def parallel_driver(catchment, catchment_no, create_ncf=False, create_spinup=False, output=True, folder=''):
    # Create a Pool with the desired number of processes
    size_setpool = min(cpu_count(), len(catchment_no))
    with Pool(processes=size_setpool) as pool:
        # Prepare arguments for each catchment
        args = [(catch, catchment, create_ncf, create_spinup, output, folder) for catch in catchment_no]
        pool.starmap(worker, args)

    print('**** FINISHED ALL RUNS ****')


def driver(catchment, catchment_no, create_ncf=False, create_spinup=False, output=True, folder=''):
    """
    Model driver: sets up model, runs it and saves results to file (create_ncf==True)
    or return dictionary of results.
    """

    """ set up model """
    running_time = time.time()

    parameters_module = importlib.import_module(f'parameters_{catchment}')
    parameters = parameters_module.parameters
    pgen, _, _, _ = parameters(folder)
    pgen['mask'] = catchment_no

    # load and process parameters
    pgen, pcpy, pbu, pds, cmask, ptop, gisinfo = preprocess_parameters(pgen, catchment, folder)

    # new directory for results files
    results_folder = create_simulation_folder(pgen)
    pgen['results_folder'] = results_folder
    results_file = os.path.join(results_folder, pgen['ncf_file'])
    pgen['ncf_file'] = results_file

    # load and process forcing data
    forcing = preprocess_forcing(pgen)
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

    # flatten arrays
    flatten = False
    if flatten == True:
        rows = pcpy['state']['LAI_conif'].shape[0]
        cols = pcpy['state']['LAI_conif'].shape[1]
        pcpy = flatten_2d_arrays(pcpy)
        pbu = flatten_2d_arrays(pbu)
        pds = flatten_2d_arrays(pds)
        ptop = flatten_2d_arrays(ptop)

    # this here temp so that we save params
    dir_path = pgen['results_folder']
    #with open(ptfile, 'w') as file:
    #    pprint.pprint(pcpy, stream=file, indent=4, width=100)
    # Loop through each dictionary and save it
    dicts = {'pgen': pgen, 'pcpy': pcpy, 'pbu': pbu, 'pds': pds, 'ptop': ptop}
    for dict_name, dict_data in dicts.items():
        file_path = os.path.join(dir_path, f'{dict_name}.txt')
        with open(file_path, 'w') as file:
            pprint.pprint(dict_data, stream=file, indent=4, width=100)
    
    # initialize SpaFHy
    spa = SpaFHy(pgen, pcpy, pbu, pds, ptop)

    for k in range(0, Nsteps):
        if pgen['simtype'] == '2D':
            deep_results, canopy_results, bucket_results = spa.run_timestep(forcing.isel(date=k))
        elif pgen['simtype'] == 'TOP':
            top_results, canopy_results, bucket_results = spa.run_timestep(forcing.isel(date=k))
        elif pgen['simtype'] == '1D':
            canopy_results, bucket_results = spa.run_timestep(forcing.isel(date=k))

        # here should reshape the results arrays
        if flatten == True:
            canopy_results = reshape_1d_to_2d(canopy_results, rows=rows, cols=cols)
            bucket_results = reshape_1d_to_2d(bucket_results, rows=rows, cols=cols)

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

def preprocess_parameters(pgen, catchment, folder=''):
    """
    Reading gisdata if applicable and preprocesses parameters
    """

    from iotools import read_bu_gisdata, read_ds_gisdata, read_cpy_gisdata, read_top_gisdata, read_aux_gisdata
    from iotools import preprocess_budata, preprocess_dsdata, preprocess_cpydata, preprocess_topdata

    parameters_module = importlib.import_module(f'parameters_{catchment}')

    # Initialize parameters based on the catchment
    initialize_parameters(catchment, folder)

    root_properties = parameters_module.root_properties
    org_properties = parameters_module.org_properties
    deep_properties = parameters_module.deep_properties
    parameters = parameters_module.parameters
    ptopmodel = parameters_module.ptopmodel
    auxiliary_grids = parameters_module.auxiliary_grids

    _, pcpy, pbu, pspd = parameters(folder)
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
                if isinstance(pgen['mask'], (int, float, np.int32, np.int64, np.float64)):
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

    return pgen, cpydata, budata, dsdata, gisdata['cmask'], ptop, gisinfo


def preprocess_forcing(pgen):
    """
    Reads forcing file(s) based on indices in pgen['forcing_id']
    Creates xarray dataset of forcing data
    """

    from iotools import read_FMI_weather
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

def flatten_2d_arrays(d):
    new_dict = {}
    for key, value in d.items():
        # If the value is a dictionary, recursively apply the function and add to the new dictionary
        if isinstance(value, dict):
            new_dict[key] = flatten_2d_arrays(value)
        # If the value is a 2D array, flatten it and add to the new dictionary
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            new_dict[key] = value.flatten()
        # Otherwise, just add the value as it is
        else:
            new_dict[key] = value
    return new_dict


def reshape_1d_to_2d(results_dict, rows, cols):
    reshaped_dict = {}
    
    for key, value in results_dict.items():
        if isinstance(value, (list, np.ndarray)) and len(value) == rows * cols:
            reshaped_dict[key] = np.array(value).reshape(rows, cols)
        else:
            reshaped_dict[key] = value  # Leave floats or other values unchanged
    
    return reshaped_dict

def create_simulation_folder(pgen):
    # Get the results folder path
    results_folder = pgen['results_folder']
    
    # Get the description, simtype and mask from pgen
    description = pgen.get('description', 'default_description')
    simtype = pgen.get('simtype', 'default_simtype')
    mask = pgen.get('mask', 'default_mask')
    
    # Get the current timestamp
    timestamp = time.strftime('%Y%m%d%H%M')
    
    # Create the subfolder name
    simulation_folder = f"{description}_{mask}_{simtype}_{timestamp}"
    
    # Create the full path for the new subfolder
    simulation_folder_path = os.path.join(results_folder, simulation_folder)
    
    # Create the subfolder
    os.makedirs(simulation_folder_path, exist_ok=True)
    
    return simulation_folder_path



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='parameter folder', type=str)

    args = parser.parse_args()

    outputfile = driver(create_ncf=True, folder=args.folder)

    print(outputfile)