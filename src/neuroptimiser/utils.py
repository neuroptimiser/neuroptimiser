"""Utility functions and constants for the Neuroptimiser framework."""
__author__ = "Jorge M. Cruz-Duarte"
__email__ = "jorge.cruz-duarte@univ-lille.fr"
__all__ = [
    "from_yaml", "get_core_params", "process_parameters",
    "reset_all_processes",
    "tro2s", "trs2o",
    "get_arch_matrix",
    "get_2d_sys", "get_izhikevich_sys",
    "IZHIKEVICH_MODELS_KIND", "DYN_MODELS_KIND", "ADJ_MAT_OPTIONS"
]

from copy import deepcopy

import numpy as np
import yaml

# Constants for Izhikevich models and dynamic systems
IZHIKEVICH_MODELS_KIND = ["RS", "IB", "CH", "FS", "TC", "TCn", "RZ", "LTS", "random"]

# Dynamic systems kinds for 2D systems
DYN_MODELS_KIND = ["saddle", "attractor", "repeller", "source", "sink"]

# Options for adjacency matrix generation
ADJ_MAT_OPTIONS = [
    "one-way-ring", "1dr", "ring",
    "two-way-ring", "2dr", "bidirectional-ring",
    "fully-connected", "all", "full",
    "random", "rand",
]


def reset_all_processes(*processes) -> None:
    """Reset all provided processes to their initial state.

    Arguments:
        *processes: Variable number of process instances to reset.
    Returns:
    None
    """
    for proc in processes:
        if hasattr(proc, "reset"):
            proc.reset()


def tro2s(x: np.ndarray | float, lb: np.ndarray | float, ub: np.ndarray | float) -> np.ndarray | float:
    """Transform a value from the original scale to a normalized scale.

    Arguments:
        x: Value or array of values to transform.
        lb: Lower bound of the original scale.
        ub: Upper bound of the original scale.
    Returns:
    Normalized value or array of values in the range [-1, 1].
    """
    return 2 * (x - lb) / (ub - lb) - 1


def trs2o(x: np.ndarray | float, lb: np.ndarray | float, ub: np.ndarray | float) -> np.ndarray | float:
    """Transform a value from a normalized scale back to the original scale.

    Arguments:
        x: Normalized value or array of values in the range [-1, 1].
        lb: Lower bound of the original scale.
        ub: Upper bound of the original scale.
    Returns:
    Value or array of values transformed back to the original scale.
    """
    return (x + 1) / 2 * (ub - lb) + lb


def get_arch_matrix(length, topology: str = "ring", num_neighbours: int = None) -> np.ndarray:
    """Generate an adjacency matrix for a given topology.

    Arguments:
        length: Number of nodes in the network.
        topology: Type of network topology (e.g., "ring", "fully-connected", "random").
        num_neighbours: Number of neighbours for random topology (if applicable).
    Returns:
    A square adjacency matrix representing the specified topology.
    """
    base_matrix = np.eye(length, length)

    if length in (1, 2):
        return base_matrix

    if topology in ("one-way-ring", "1dr", "ring"):
        # 1d ring topology
        return np.roll(base_matrix, -1, 1)
    elif topology in ("two-way-ring", "2dr", "bidirectional-ring"):
        return np.roll(base_matrix, -1, 1) + np.roll(base_matrix, 1, 1)
    elif topology in ("fully-connected", "all", "full"):
        return np.ones((length, length)) - base_matrix
    elif topology in ("random", "rand"):
        if 0 < num_neighbours < length:
            # Randomly select the neighbours preserving the diagonal in zeros
            matrix = np.zeros((length, length))
            for i in range(length):
                matrix[i, np.random.choice(np.delete(np.arange(length), i, 0), num_neighbours, replace=False)] = 1
            return matrix
        else:
            raise ValueError(f"Invalid number of neighbours: {num_neighbours}")
    else:
        raise NotImplementedError("Topology not implemented yet (?)")


def get_2d_sys(kind="sink", trA_max=1.5, detA_max=3.0, eps=1e-6) -> np.ndarray:
    """Generate a 2D dynamic system matrix based on the specified kind.

    Arguments:
        kind: Type of dynamic system ("random", "saddle", "attractor", "repeller", "source", "sink", or "centre").
        trA_max: Maximum trace value for the system matrix.
        detA_max: Maximum determinant value for the system matrix.
        eps: Small value to avoid division by zero or negative values.
    Returns:
    A 2x2 numpy array representing the system matrix.
    """
    if kind == "random":
        _kind = np.random.choice(DYN_MODELS_KIND)
        return get_2d_sys(_kind, trA_max=trA_max, detA_max=detA_max, eps=eps)
    elif kind == "saddle":
        detA = np.random.uniform(-detA_max, eps)

        a = 2.0 * np.random.uniform(eps, trA_max) - 1.0
        d = detA / a

        b = 2.0 * np.random.uniform(eps, trA_max) - 1.0
        c = 0.0
    else:
        abs_trA = np.random.uniform(eps, trA_max)
        trA = abs_trA if kind in ["repeller", "source"] else -abs_trA

        trAsq4 = (trA ** 2) / 4
        if kind in ["attractor", "repeller"]:
            # discriminant = trA^2 - 4 (trA^2/4 - delta) = 4 delta > 0 (node)
            delta = np.random.uniform(eps, trAsq4 - eps)
        elif kind in ["source", "sink"]:
            # discriminant = trA^2 - 4 (trA^2/4 - delta) = 4 delta < 0 (spiral)
            delta = np.random.uniform(-trA_max, -eps)
        else:  # Centre
            delta = 0.0

        detA = trAsq4 - delta

        a = 2.0 * np.random.uniform(eps, trA_max) - 1.0
        b = 2.0 * np.random.uniform(eps, trA_max) - 1.0

        d = trA - a
        c = (a * d - detA) / b

    return np.array([[a, b], [c, d]])


def get_izhikevich_sys(kind="RS", scale=0.1) -> dict:
    """Get the parameters for an Izhikevich neuron model.

    Arguments:
        kind: Type of Izhikevich model (e.g., "RS", "IB", "CH", "FS", "TC", "TCn", "RZ", "LTS", or "random").
        scale: Scale factor for random perturbation of parameters (default is 0.1).
    Returns:
    A dictionary containing the parameters of the Izhikevich model.
    """
    if kind == "random":
        kind = np.random.choice(IZHIKEVICH_MODELS_KIND) + "r"
        return get_izhikevich_sys(kind)
    else:
        # Default parameters for Izhikevich model
        a = 0.02
        b = 0.2
        c = -65
        d = 8.0
        I = 0.1
        vmin = -80.  # [V]
        vmax = 30.
        umin = -20.  # [V]
        umax = 0.
        Lt = 1.0

        match kind:
            case "IB":
                c = -55
                d = 4.0
            case "CH":
                c = -50
                d = 2.0
            case "FS":
                a = 0.1
                d = 2.0
            case "TC":
                a = 0.02
                b = 0.25
                d = 0.05
                I = 0.0
            case "TCn":
                a = 0.02
                b = 0.25
                d = 0.05
                I = -10.0
            case "RZ":
                a = 0.1
                b = 0.26
                d = 2.0
            case "LTS":
                a = 0.2
                b = 0.25
                d = 2.0
            case _:
                pass  # RS

        coeffs = {
            "a": a, "b": b, "c": c, "d": d, "I": I,
            "vmin": vmin, "vmax": vmax, "umin": umin, "umax": umax, "Lt": Lt,
        }
        if kind[-1] == "r":
            for key in ["a", "b", "c", "d", "I"]:
                value = coeffs[key]
                new_value = value + np.random.randn() * abs(value) * scale
                coeffs[key] = new_value
        return coeffs

def get_core_params(core_cfg, **kwargs):
    _num_steps = int(kwargs.get("num_steps", 500))

    # Default core parameters
    cdp_alpha       = float(core_cfg.get("alpha", 1.0))
    cdp_dt          = float(core_cfg.get("dt", 0.01))
    cdp_max_steps   = core_cfg.get("max_steps", None)
    if cdp_max_steps is None:
        cdp_max_steps = _num_steps
    cdp_noise_std   = core_cfg.get("noise_std", (0.0, 0.3))
    cdp_ref_mode    = core_cfg.get("ref_mode", "pg")
    cdp_is_bounded  = bool(core_cfg.get("is_bounded", True))

    _hl_selector_params = core_cfg.get("selector", {})

    threshold_cfg   = core_cfg.get("threshold", {})
    cdp_thr_mode    = threshold_cfg.get("thr_mode", "diff_pg")
    cdp_thr_alpha   = float(threshold_cfg.get("thr_alpha", 2.0))
    cdp_thr_min     = float(threshold_cfg.get("thr_min", 1e-6))
    cdp_thr_max     = float(threshold_cfg.get("thr_max", 2.0))
    cdp_thr_k       = float(threshold_cfg.get("thr_k", 0.05))

    spiking_cfg     = core_cfg.get("spiking", {})
    cdp_spk_cond    = spiking_cfg.get("spk_cond", "l2")
    cdp_spk_alpha   = float(spiking_cfg.get("spk_alpha", 0.25))
    cdp_spk_q_ord   = spiking_cfg.get("spk_q_ord", 2)
    cdp_spk_weights = spiking_cfg.get("spk_weights", [1.0, 1.0])

    hd_operator_cfg = core_cfg.get("hd_operator", {})
    cdp_name        = hd_operator_cfg.get("name", "linear")
    cdp_coeffs      = hd_operator_cfg.get("coeffs", "sink")
    cdp_approx      = hd_operator_cfg.get("approx", "rk4")

    hs_operator_cfg = core_cfg.get("hs_operator", {})
    cdp_hs_operator = hs_operator_cfg.get("name", "differential")
    cdp_hs_variant  = hs_operator_cfg.get("variant", "current-to-rand")

    selector_cfg    = core_cfg.get("selector", {})
    cdp_sel_mode    = selector_cfg.get("sel_mode", "greedy")

    # Core parameters dictionary
    _core_params = {
        "alpha":        cdp_alpha,
        "dt":           cdp_dt,
        "max_steps":    cdp_max_steps,
        "noise_std":    cdp_noise_std,
        "ref_mode":     cdp_ref_mode,
        "is_bounded":   cdp_is_bounded,
        "name":         cdp_name,
        "coeffs":       cdp_coeffs,
        "approx":       cdp_approx,
        "thr_mode":     cdp_thr_mode,
        "thr_alpha":    cdp_thr_alpha,
        "thr_min":      cdp_thr_min,
        "thr_max":      cdp_thr_max,
        "thr_k":        cdp_thr_k,
        "spk_cond":     cdp_spk_cond,
        "spk_alpha":    cdp_spk_alpha,
        "spk_q_ord":    cdp_spk_q_ord,
        "spk_weights":  cdp_spk_weights,
        "hs_operator":  cdp_hs_operator,
        "hs_variant":   cdp_hs_variant,
        "sel_mode":     cdp_sel_mode,
    }

    return _core_params, _hl_selector_params


def process_parameters(_experiment_config):
    # Extract nested config
    problems_cfg        = _experiment_config.get("metadata", {})
    optimiser_cfg       = _experiment_config.get("optimiser", {})
    full_core_cfg       = _experiment_config.get("core_parameters", {})
    default_core_cfg    = full_core_cfg.get("default", {})

    # Problem setup
    name                = problems_cfg.get("name", "unnamed experiment")
    description         = problems_cfg.get("description", "")
    debug_mode          = bool(problems_cfg.get("debug_mode", False))
    suite_name          = problems_cfg.get("suite_name", "bbob")
    func_indices        = problems_cfg.get("func_indices", None)
    num_dimensions      = problems_cfg.get("num_dimensions", [2])
    instances           = problems_cfg.get("instances", [1])
    budget_multiplier   = int(problems_cfg.get("budget_multiplier", 1))

    # Optimiser setup
    num_steps           = int(float(optimiser_cfg.get("num_steps", 500)))
    num_agents          = int(optimiser_cfg.get("num_agents", 30))
    spiking_core        = optimiser_cfg.get("spiking_core", "TwoDimSpikingCore")

    neighbourhood_cfg   = optimiser_cfg.get("neighbourhood", {})
    num_neighbours      = int(neighbourhood_cfg.get("num_neighbours", 10))
    neuron_topology     = neighbourhood_cfg.get("neuron_topology", "2dr")
    unit_topology       = neighbourhood_cfg.get("unit_topology", "random")

    # Generate the config parameters
    _config_params = {
        "num_iterations":   num_steps,
        "num_agents":       num_agents,
        "spiking_core":     spiking_core,
        "num_neighbours":   num_neighbours,
        "neuron_topology":  neuron_topology,
        "unit_topology":    unit_topology,
    }

    # Generate the suite config
    _experiment_params = {
        "name":             name,
        "description":      description,
        "debug_mode":       debug_mode,
        "suite_name":       suite_name,
        "budget_multiplier":    budget_multiplier,
        "func_indices":     func_indices,
        "instance":         instances,
        "dimensions":       num_dimensions,
    }

    # Identify the custom core parameters and process them
    custom_core_keys = [k for k in full_core_cfg.keys() if k.startswith("core_")]
    custom_core_weights = np.array([float(full_core_cfg[k].get("weight", -1))
                                    for k in custom_core_keys])
    num_custom_cores = len(custom_core_keys)

    if np.any(custom_core_weights == -1):
        raise ValueError("Custom core parameters must have a weight specified.")

    sum_core_weights = np.sum(custom_core_weights)
    custom_core_weights = custom_core_weights / sum_core_weights

    # From num_agents decide which has a config
    _core_params = []
    if num_custom_cores > 0:
        for custom_core_key in np.random.choice(
                a=custom_core_keys, size=num_agents, p=custom_core_weights):
            # Get the two dicts
            dcp = deepcopy(default_core_cfg)  # default core parameters
            ccp = full_core_cfg.get(custom_core_key, {})  # custom core parameters

            # Override the default one
            dcp.update(ccp)

            # Process the parameters
            unit_core_parameters, unit_sel_parameters = get_core_params(dcp, num_steps=num_steps)

            _core_params.append(unit_core_parameters)
            # print(f"{custom_core_key} with {ccp}")
    else:
        for _ in range(num_agents):
            dcp = deepcopy(default_core_cfg)
            unit_core_parameters, unit_sel_parameters = get_core_params(dcp, num_steps=num_steps)
            _core_params.append(unit_core_parameters)

    # Return the relevant dicts
    return _experiment_params, _config_params, _core_params

def from_yaml(yaml_path: str, disp: bool = False) -> tuple[dict, dict, dict]:
    """Load a YAML file and return its contents as a dictionary.

    Arguments:
        yaml_path: Path to the YAML file.
        disp: If True, print the loaded configuration for debugging purposes.

    Returns:
    A dictionary containing the contents of the YAML file.
    """

    if not yaml_path.endswith(".yaml") and not yaml_path.endswith(".yml"):
        raise ValueError("The file must have a .yaml or .yml extension.")
    with open(yaml_path, 'r') as file:
        experiment_config = yaml.safe_load(file)

    experiment_params, config_params, core_params = process_parameters(experiment_config)

    if disp:
        print(f"\nLoaded experiment configuration from YAML: {yaml_path}")
        print("=" * 40)
        print("Experiment parameters:")
        print("-" * 40)
        for key, value in experiment_params.items():
            print(f"  {key}: {value}")
        print("\nConfig parameters:")
        print("-" * 40)
        for key, value in config_params.items():
            print(f"  {key}: {value}")
        print("\nDefault Core parameters:")
        print("-" * 40)
        for key, value in core_params[0].items():
            print(f"  {key}: {value}")
        print("\nFull core parameters for all units:")
        print("-" * 40)
        for i, cp in enumerate(core_params):
            print(f" NHU {i+1}:")
            for key, value in cp.items():
                print(f"    {key}: {value}")
        print("=" * 40)

    return experiment_params, config_params, core_params