import argparse
import os
import importlib
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from model_driver import driver
from read_runoff import extract_discharge_results_2D, kge, mbe, read_runoff, rmse
from scripts.create_soil_params import create_soil_params

load_dotenv()

MINIMIZE_METRICS = {'rmse_tot', 'mbe_tot', 'rmse_sbsrf', 'mbe_sbsrf'}


def resolve_model_root():
    return Path(__file__).resolve().parents[1] / 'SpaFHy_RUNS' / 'krycklan'


def read_glue_results(glue_results_csv, soil_type):
    glue_results_csv = Path(glue_results_csv).expanduser().resolve()
    if not glue_results_csv.exists():
        raise FileNotFoundError(f'GLUE results file not found: {glue_results_csv}')

    summary = pd.read_csv(glue_results_csv)
    required_columns = {'sample_id', f'kmax_{soil_type}', f'kmin_{soil_type}', f'f_{soil_type}'}
    missing = required_columns - set(summary.columns)
    if missing:
        raise ValueError(f'Missing required columns in {glue_results_csv}: {sorted(missing)}')

    return summary, glue_results_csv


def select_best_rows(summary, n_best, metric_name):
    if metric_name not in summary.columns:
        raise ValueError(f'Metric column not found in GLUE results: {metric_name}')

    ascending = metric_name in MINIMIZE_METRICS
    ranked = summary.dropna(subset=[metric_name]).sort_values(metric_name, ascending=ascending)
    return ranked.head(n_best).copy()


def build_parameter_dict(row, soil_type):
    kmax = float(row[f'kmax_{soil_type}'])
    kmin = float(row[f'kmin_{soil_type}'])
    f = float(row[f'f_{soil_type}'])

    return {
        'kmax_values': {soil_type: kmax},
        'kmin_values': {soil_type: kmin},
        'f_values': {soil_type: f},
    }


def save_run_outputs(output_ncf_path, q_df, params):
    """Save calibration params and runoff comparison to results folder."""
    out_dir = Path(output_ncf_path).parent

    params_df = pd.DataFrame.from_dict(params, orient='index', columns=['value'])
    params_df.index.name = 'parameter'
    params_df.to_csv(out_dir / 'calibration_params.txt')
    q_df.to_csv(out_dir / 'runoff.txt')

    return out_dir


def run_best_row(row, folder, catchment, catchment_no, q_obs, soil_type):
    param_sets = build_parameter_dict(row, soil_type)

    psoil = create_soil_params(
        kmax_values=param_sets['kmax_values'],
        kmin_values=param_sets['kmin_values'],
        f_values=param_sets['f_values'],
        write=False,
        verbose=False,
    )

    parameters_module = importlib.import_module('parameters_krycklan')
    pgen, _, _, _ = parameters_module.parameters(folder, soil_params=psoil)
    dates = pd.date_range(pgen['start_date'], pgen['end_date'])
    spinup_steps = (pd.to_datetime(pgen['spinup_end']) - pd.to_datetime(pgen['start_date'])).days + 1
    sim_dates = dates[spinup_steps:]

    # Run model with ncf output (like calibration script)
    outputfile = driver(
        catchment=catchment,
        catchment_no=catchment_no,
        create_ncf=True,
        create_spinup=False,
        output=True,
        folder=folder,
        psoil=psoil,
        save_outputs=True,
    )

    # Handle both string and list return types
    if isinstance(outputfile, (list, tuple)):
        output_path = outputfile[0]
    else:
        output_path = outputfile

    # Extract discharge from netcdf
    surface_runoff, netflow_to_ditch, netflow_to_lake = extract_discharge_results_2D(output_path)
    q_sim_tot = surface_runoff + netflow_to_ditch + netflow_to_lake
    q_sim_sbsrf = netflow_to_ditch + netflow_to_lake

    # Build dataframe with spinup trimming using pd.concat like calibration script
    q_df = pd.concat(
        {
            'q_obs': q_obs['Q'],
            'q_sim': pd.Series(q_sim_tot, index=sim_dates),
            'q_sim_srf': pd.Series(surface_runoff, index=sim_dates),
            'q_sim_sbsrf': pd.Series(q_sim_sbsrf, index=sim_dates),
        },
        axis=1,
        join='inner',
    ).dropna()
    q_df.index.name = 'time'

    params = {
        f'kmax_{soil_type}': float(row[f'kmax_{soil_type}']),
        f'kmin_{soil_type}': float(row[f'kmin_{soil_type}']),
        f'f_{soil_type}': float(row[f'f_{soil_type}']),
        'kge_tot': kge(q_df['q_obs'], q_df['q_sim']),
        'rmse_tot': rmse(q_df['q_obs'], q_df['q_sim']),
        'mbe_tot': mbe(q_df['q_obs'], q_df['q_sim']),
        'kge_sbsrf': kge(q_df['q_obs'], q_df['q_sim_sbsrf']),
        'rmse_sbsrf': rmse(q_df['q_obs'], q_df['q_sim_sbsrf']),
        'mbe_sbsrf': mbe(q_df['q_obs'], q_df['q_sim_sbsrf']),
    }

    return save_run_outputs(output_path, q_df, params)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run SpaFHy simulations for the best GLUE parameter sets and save outputs like the calibration script.'
    )
    parser.add_argument('glue_results_csv', help='Path to glue_results.csv produced by run_glue_krycklan.py')
    parser.add_argument('n_best', type=int, help='Number of best simulations to run')
    parser.add_argument(
        '--metric',
        default='rmse_tot',
        help='Metric column used to rank GLUE results (default: rmse_tot)',
    )
    parser.add_argument(
        '--catchment-no',
        type=int,
        default=2,
        help='Catchment number to simulate (default: 2)',
    )
    parser.add_argument(
        '--soil-type',
        default='Medium',
        help='Soil type to extract from the GLUE result columns (default: Medium)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder = str(resolve_model_root())
    runoff_folder = str(Path(folder) / 'obs' / 'DISCHARGE')

    q_obs = read_runoff(args.catchment_no, runoff_folder)
    glue_results, glue_results_path = read_glue_results(args.glue_results_csv, args.soil_type)

    selected = select_best_rows(glue_results, args.n_best, args.metric)
    print(f'Reading GLUE results from: {glue_results_path}')
    print(f'Selecting top {len(selected)} rows by {args.metric}')

    out_dirs = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        print(f'\nRunning best simulation {rank}/{len(selected)}')
        out_dir = run_best_row(
            row=row,
            folder=folder,
            catchment='krycklan',
            catchment_no=args.catchment_no,
            q_obs=q_obs,
            soil_type=args.soil_type,
        )
        out_dirs.append(out_dir)
        print(f'  Saved outputs to {out_dir}')

    print('\n--- Finished best-parameter runs ---')
    for out_dir in out_dirs:
        print(f'  {out_dir}')