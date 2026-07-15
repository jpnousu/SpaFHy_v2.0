import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def discretized_depths(max_depth):
    depth_breaks = []

    if max_depth > 0:
        upper = min(max_depth, 1.0)
        depth_breaks.extend(np.arange(0.1, upper + 1e-9, 0.1).tolist())
    if max_depth > 1.0:
        upper = min(max_depth, 2.0)
        depth_breaks.extend(np.arange(1.2, upper + 1e-9, 0.2).tolist())
    if max_depth > 2.0:
        upper = min(max_depth, 4.0)
        depth_breaks.extend(np.arange(2.5, upper + 1e-9, 0.5).tolist())
    if max_depth > 4.0:
        upper = max_depth
        depth_breaks.extend(np.arange(5.0, upper + 1e-9, 1.0).tolist())

    if not depth_breaks or depth_breaks[-1] < max_depth:
        depth_breaks.append(max_depth)

    return np.round(np.array(depth_breaks), 3)


def round_sig(value, sig_figs=3):
    if value == 0:
        return 0.0
    return float(f"{value:.{sig_figs - 1}e}")


def expand_pf_layers(pf_dict, n_layers):
    """Expand pF parameters so each list has one value per deep layer."""
    expanded = {}
    for key, values in pf_dict.items():
        if not isinstance(values, list):
            expanded[key] = [values] * n_layers
            continue

        if len(values) == n_layers:
            expanded[key] = values
        elif len(values) == 0:
            expanded[key] = []
        elif all(v == values[0] for v in values):
            expanded[key] = [values[0]] * n_layers
        elif len(values) < n_layers:
            expanded[key] = values + [values[-1]] * (n_layers - len(values))
        else:
            expanded[key] = values[:n_layers]

    return expanded


def fmt_number(v):
    """Format numbers for readable Python literals."""
    if isinstance(v, int):
        return str(v)

    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e3):
        s = f"{v:.3E}"
        mantissa, exp = s.split("E")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mantissa}E{exp_int:+03d}"

    return repr(float(v))


def fmt_list(values, indent="", compact=False):
    if compact and len(values) > 0 and all(v == values[0] for v in values):
        return f"[{fmt_number(values[0])}] * {len(values)}"
    return "[" + ", ".join(fmt_number(v) for v in values) + "]"


def deep_properties_function_text(deepp_dict):
    lines = []
    lines.append("def deep_properties():")
    lines.append("    \"\"\"")
    lines.append("    Properties of soil profiles generated from exponential conductivity parameters.")
    lines.append("    Note z is elevation of lower boundary of layer (soil surface at 0.0).")
    lines.append("    \"\"\"")
    lines.append("    deepp = {")

    for soil_type, profile in deepp_dict.items():
        lines.append(f"        {soil_type!r}: {{")
        lines.append(f"            'deep_id': {profile['deep_id']},")
        lines.append(f"            'deep_z': {fmt_list(profile['deep_z'])},")

        pf = profile.get('pF', {})
        lines.append("            'pF': {")
        lines.append(f"                'ThetaS': {fmt_list(pf.get('ThetaS', []), compact=True)},")
        lines.append(f"                'ThetaR': {fmt_list(pf.get('ThetaR', []), compact=True)},")
        lines.append(f"                'alpha': {fmt_list(pf.get('alpha', []), compact=True)},")
        lines.append(f"                'n': {fmt_list(pf.get('n', []), compact=True)},")
        lines.append("            },")

        lines.append(f"            'deep_ksat': {fmt_list(profile['deep_ksat'])},")
        lines.append("        },")

    lines.append("    }")
    lines.append("    return deepp")
    return "\n".join(lines)


def org_properties_function_text(orgp_dict):
    lines = []
    lines.append("def org_properties():")
    lines.append("    \"\"\"")
    lines.append("    swedish_soilmap")
    lines.append("    \"\"\"")
    lines.append("    orgp = {")

    for soil_type, params in orgp_dict.items():
        lines.append(f"        {soil_type!r}: {{")
        lines.append(f"            'org_id': {params['org_id']},")
        lines.append(f"            'org_depth': {params['org_depth']},")
        lines.append(f"            'org_poros': {params['org_poros']},")
        lines.append(f"            'org_fc': {params['org_fc']},")
        lines.append(f"            'org_rw': {params['org_rw']},")
        lines.append(f"            'org_ksat': {params['org_ksat']},")
        lines.append(f"            'org_beta': {params['org_beta']},")
        lines.append("        },")

    lines.append("    }")
    lines.append("    return orgp")
    return "\n".join(lines)


def root_properties_function_text(rootp_dict):
    lines = []
    lines.append("def root_properties():")
    lines.append("    \"\"\"")
    lines.append("    swedish_soilmap")
    lines.append("    \"\"\"")
    lines.append("    rootp = {")

    for soil_type, params in rootp_dict.items():
        lines.append(f"        {soil_type!r}: {{")
        lines.append(f"            'root_id': {params['root_id']},")
        lines.append(f"            'root_depth': {params['root_depth']},")
        lines.append(f"            'root_poros': {params['root_poros']},")
        lines.append(f"            'root_fc': {params['root_fc']},")
        lines.append(f"            'root_wp': {params['root_wp']},")
        lines.append(f"            'root_wr': {params['root_wr']},")
        lines.append(f"            'root_ksat': {params['root_ksat']},")
        lines.append(f"            'root_alpha': {params['root_alpha']},")
        lines.append(f"            'root_n': {params['root_n']},")
        lines.append(f"            'root_beta': {params['root_beta']},")
        lines.append("        },")

    lines.append("    }")
    lines.append("    return rootp")
    return "\n".join(lines)


def wrc(pF: Dict, theta: np.ndarray = None, psi: np.ndarray = None, draw_pF: bool = False):
    """
    vanGenuchten-Mualem soil water retention model

    References:
        Schaap and van Genuchten (2005). Vadose Zone 5:27-34
        van Genuchten, (1980). Soil Science Society of America Journal 44:892-898

    Args:
        pF (dict):
            ThetaS (float|array): saturated water content [m3 m-3]
            ThetaR (float|array): residual water content [m3 m-3]
            alpha (float|array): air entry suction [cm-1]
            n (float|array): pore size distribution [-]
        theta (float|array): vol. water content [m3 m-3]
        psi (float|array): water potential [m]
        draw_pF (bool): Draw pF-curve.
    Returns:
        y (float|array): water potential [m] or vol. water content [m3 m-3], or None.
        fc (float): field capacity [m3 m-3] at psi = -1 m (top layer).
        wp (float): wilting point [m3 m-3] at psi = -150 m (top layer).
    """
    eps = np.finfo(float).eps

    Ts = np.array(pF['ThetaS'])
    Tr = np.array(pF['ThetaR'])
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    def theta_psi(x):
        x = np.minimum(x, Ts)
        x = np.maximum(x, Tr)
        s = (Ts - Tr) / ((x - Tr) + eps)
        Psi = -1e-2 / alfa * (s ** (1.0 / m) - 1.0) ** (1.0 / n)
        Psi[np.isnan(Psi)] = 0.0
        return Psi

    def psi_theta(x):
        x = 100 * np.minimum(x, 0)
        Th = Tr + (Ts - Tr) / (1 + abs(alfa * x) ** n) ** m
        return Th

    # Always compute fc and wp from the top (first) layer parameters
    fc = float(psi_theta(-1.0).flat[0])
    wp = float(psi_theta(-150.0).flat[0])

    y = None
    if theta is not None:
        y = theta_psi(theta)
    elif psi is not None:
        y = psi_theta(psi)

    if draw_pF:
        Ts = Ts[0]
        Tr = Tr[0]
        alfa = alfa[0]
        n = n[0]
        xx = -np.logspace(-4, 5, 100)
        yy = psi_theta(xx)

        fig = plt.figure(99)
        fig.suptitle('vanGenuchten-Mualem WRC', fontsize=16)
        ttext = (r'$\theta_s=$' + str(Ts) + r', $\theta_r=$' + str(Tr)
                 + r', $\alpha=$' + str(alfa) + ',n=' + str(n))
        plt.title(ttext, fontsize=14)
        plt.semilogx(-xx, yy, 'g-')
        plt.semilogx(1, fc, 'ro', 150, wp, 'ro')
        plt.text(1, 1.1 * fc, 'FC'), plt.text(150, 1.2 * wp, 'WP')
        plt.ylabel(r'$\theta$  $(m^3m^{-3})$', fontsize=14)
        plt.xlabel(r'$\psi$ $(m)$', fontsize=14)
        plt.ylim(0.8 * Tr, min(1, 1.1 * Ts))
        y = None

    return y, fc, wp
