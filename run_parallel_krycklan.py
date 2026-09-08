#!/usr/bin/env python3

"""Run SpaFHy Krycklan simulations from the command line.

This is a reusable entry point for local runs and HPC batch jobs.
Use --catchment-no for one catchment, --catchment-nos for a subset,
or --all-catchments for the full Krycklan set used in the notebook.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


DEFAULT_CATCHMENT_NOS = np.array([
    141, 142, 143,
    151, 152, 153,
    161, 162, 163, 164,
    20, 21, 22, 8, 10,
    1, 2, 4, 5, 6, 7, 9, 12, 13, 3,
], dtype=int)


def resolve_spafhy_folder(spafhy_folder_arg: str | None) -> Path:
    spafhy_folder = spafhy_folder_arg or os.getenv('SPAFHY_FOLDER')
    if spafhy_folder is None:
        return Path(__file__).resolve().parent
    return Path(spafhy_folder).expanduser().resolve()


def resolve_project_folder(project_folder_arg: str | None) -> Path:
    project_folder = project_folder_arg or os.getenv('PROJECT_FOLDER')
    if project_folder is None:
        raise EnvironmentError('PROJECT_FOLDER is not set. Pass --project-folder or export PROJECT_FOLDER.')
    return Path(project_folder).expanduser().resolve()


def build_catchment_selection(args):
    if args.all_catchments:
        return DEFAULT_CATCHMENT_NOS
    if args.catchment_no is not None:
        return args.catchment_no
    if args.catchment_nos:
        if len(args.catchment_nos) == 1:
            return int(args.catchment_nos[0])
        return np.array(args.catchment_nos, dtype=int)
    return DEFAULT_CATCHMENT_NOS


def parse_args():
    parser = argparse.ArgumentParser(description='Run SpaFHy Krycklan simulations.')
    parser.add_argument('--catchment', default='krycklan', help='Catchment name, defaults to krycklan.')
    parser.add_argument('--project-folder', help='Folder that contains the catchment run data.')
    parser.add_argument('--spafhy-folder', help='Folder that contains the SpaFHy source code.')
    parser.add_argument('--n-workers', type=int, help='Cap the multiprocessing pool size.')

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--catchment-no', type=int, help='Run one catchment number.')
    selection.add_argument('--catchment-nos', nargs='+', type=int, help='Run a subset of catchments.')
    selection.add_argument('--all-catchments', action='store_true', help='Run the full Krycklan catchment list.')

    parser.add_argument('--create-ncf', dest='create_ncf', action='store_true', default=True, help='Write netCDF output (default).')
    parser.add_argument('--no-ncf', dest='create_ncf', action='store_false', help='Disable netCDF output.')
    parser.add_argument('--create-spinup', action='store_true', help='Also write the spinup netCDF file.')
    parser.add_argument('--output', dest='output', action='store_true', default=True, help='Return the output file path (default).')
    parser.add_argument('--no-output', dest='output', action='store_false', help='Do not return the output file path.')

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    spafhy_folder = resolve_spafhy_folder(args.spafhy_folder)
    project_folder = resolve_project_folder(args.project_folder)
    catchment_nos = build_catchment_selection(args)

    if args.n_workers is not None:
        os.environ['SPAFHY_N_WORKERS'] = str(args.n_workers)

    os.chdir(spafhy_folder)
    sys.path.insert(0, str(spafhy_folder))

    from model_driver import parallel_driver

    folder = project_folder / args.catchment
    folder.mkdir(parents=True, exist_ok=True)

    print(f'Running catchment: {args.catchment}')
    print(f'Source folder: {spafhy_folder}')
    print(f'Project folder: {project_folder}')
    print(f'Catchment selection: {catchment_nos}')

    outputfile = parallel_driver(
        args.catchment,
        catchment_no=catchment_nos,
        create_ncf=args.create_ncf,
        create_spinup=args.create_spinup,
        output=args.output,
        folder=str(folder),
    )

    if args.output:
        print('Output files:')
        for item in outputfile:
            print(f'  {item}')


if __name__ == '__main__':
    main()