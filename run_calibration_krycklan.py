import os
import itertools
import numpy as np
from scripts.create_soil_params import create_soil_params
from pathlib import Path
from model_driver import parallel_driver

# Load .env from the same directory as this script
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

if __name__ == '__main__':
    io_path = str(Path(os.getenv('PROJECT_FOLDER')))
    folder = os.path.join(io_path, 'krycklan')  # io repo

    # f values for calibration
    f_min, f_max, f_step = 1.0, 6.0, 3.0
    f_range = np.arange(f_max, f_min, -f_step)
    print(f"f_range: {f_range}")

    # kmax values for calibration
    # currently tests: could be kmax 1e-6 to 1e-3 and f 1.0 to 8.0
    kmax_min, kmax_max, kmax_n = 1e-5, 1e-3, 2
    kmax_range = np.logspace(np.log10(kmax_min), np.log10(kmax_max), kmax_n)
    print(f"kmax_range: {kmax_range}")

    # all possible combinations of kmax and f values
    combinations = list(itertools.product(kmax_range, f_range))
    print(f"Total combinations: {len(combinations)}")

    for kmax, f in combinations:
        kmax_values = {'Medium': kmax}
        f_values = {'Medium': f}
        # Create parameters_krycklan_soil.py file with the given kmax and f values
        result = create_soil_params(
            kmax_values,
            f_values,
            write=True,
            verbose=False)
        # Run the model
        outputfile = parallel_driver(catchment='krycklan', catchment_no=2, create_ncf=True, create_spinup=False, output=True, folder=folder)