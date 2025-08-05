import argparse
import collections
import gc
import csv

import sys

sys.setrecursionlimit(5000)

from neuroptimiser import NeurOptimiser
from datetime import datetime
from copy import deepcopy
import numpy as np
import cocoex
import cocopp
import yaml
import time
import os


def to_coco_str(x):
    if x is None:
        return ""
    elif isinstance(x, list):
        if len(x) > 1:
            return ",".join(map(str, x))
        else:
            return str(x[0])
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, str):
        return str(x)
    else:
        raise ValueError("Invalid type for COCO string conversion.")


def get_core_params(core_cfg, **kwargs):
    _num_steps = int(kwargs.get("num_steps", 500))

    # Default core parameters
    cdp_alpha = float(core_cfg.get("alpha", 1.0))
    cdp_dt = float(core_cfg.get("dt", 0.01))
    cdp_max_steps = core_cfg.get("max_steps", None)
    if cdp_max_steps is None:
        cdp_max_steps = _num_steps
    cdp_noise_std = core_cfg.get("noise_std", (0.0, 0.3))
    cdp_ref_mode = core_cfg.get("ref_mode", "pg")
    cdp_is_bounded = bool(core_cfg.get("is_bounded", True))

    _selector_params = core_cfg.get("selector", {})

    threshold_cfg = core_cfg.get("threshold", {})
    cdp_thr_mode = threshold_cfg.get("thr_mode", "diff_pg")
    cdp_thr_alpha = float(threshold_cfg.get("thr_alpha", 2.0))
    cdp_thr_min = float(threshold_cfg.get("thr_min", 1e-6))
    cdp_thr_max = float(threshold_cfg.get("thr_max", 2.0))
    cdp_thr_k = float(threshold_cfg.get("thr_k", 0.05))

    spiking_cfg = core_cfg.get("spiking", {})
    cdp_spk_cond = spiking_cfg.get("spk_cond", "l2")
    cdp_spk_alpha = float(spiking_cfg.get("spk_alpha", 0.25))

    hd_operator_cfg = core_cfg.get("hd_operator", {})
    cdp_name = hd_operator_cfg.get("name", "linear")
    cdp_coeffs = hd_operator_cfg.get("coeffs", "sink")
    cdp_approx = hd_operator_cfg.get("approx", "rk4")

    hs_operator_cfg = core_cfg.get("hs_operator", {})
    cdp_hs_operator = hs_operator_cfg.get("name", "differential")
    cdp_hs_variant = hs_operator_cfg.get("variant", "current-to-rand")

    # Core parameters dictionary
    _core_params = {
        "alpha": cdp_alpha,
        "dt": cdp_dt,
        "max_steps": cdp_max_steps,
        "noise_std": cdp_noise_std,
        "ref_mode": cdp_ref_mode,
        "is_bounded": cdp_is_bounded,
        "name": cdp_name,
        "coeffs": cdp_coeffs,
        "approx": cdp_approx,
        "thr_mode": cdp_thr_mode,
        "thr_alpha": cdp_thr_alpha,
        "thr_min": cdp_thr_min,
        "thr_max": cdp_thr_max,
        "thr_k": cdp_thr_k,
        "spk_cond": cdp_spk_cond,
        "spk_alpha": cdp_spk_alpha,
        "hs_operator": cdp_hs_operator,
        "hs_variant": cdp_hs_variant,
    }

    return _core_params, _selector_params


def process_parameters(_experiment_config):
    # Extract nested config
    problems_cfg = _experiment_config.get("metadata", {})
    optimiser_cfg = _experiment_config.get("optimiser", {})
    full_core_cfg = _experiment_config.get("core_parameters", {})
    default_core_cfg = full_core_cfg.get("default", {})

    # Problem setup
    name = problems_cfg.get("name", "unnamed experiment")
    description = problems_cfg.get("description", "")
    debug_mode = bool(problems_cfg.get("debug_mode", False))
    suite_name = problems_cfg.get("suite_name", "bbob")
    func_indices = problems_cfg.get("func_indices", None)
    num_dimensions = problems_cfg.get("num_dimensions", [2])
    instances = problems_cfg.get("instances", [1])
    budget_multiplier = int(problems_cfg.get("budget_multiplier", 1))

    # Optimiser setup
    num_steps = int(float(optimiser_cfg.get("num_steps", 500)))
    num_agents = int(optimiser_cfg.get("num_agents", 30))
    spiking_core = optimiser_cfg.get("spiking_core", "TwoDimSpikingCore")

    neighbourhood_cfg = optimiser_cfg.get("neighbourhood", {})
    num_neighbours = int(neighbourhood_cfg.get("num_neighbours", 10))
    neuron_topology = neighbourhood_cfg.get("neuron_topology", "2dr")
    unit_topology = neighbourhood_cfg.get("unit_topology", "random")

    # Generate the config parameters
    _config_params = {
        "num_iterations": num_steps,
        "num_agents": num_agents,
        "spiking_core": spiking_core,
        "num_neighbours": num_neighbours,
        "neuron_topology": neuron_topology,
        "unit_topology": unit_topology,
    }

    # Generate the suite config
    _experiment_params = {
        "name": name,
        "description": description,
        "debug_mode": debug_mode,
        "suite_name": suite_name,
        "budget_multiplier": budget_multiplier,
        "func_indices": func_indices,
        "instance": instances,
        "dimensions": num_dimensions,
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
            unit_core_parameters, _ = get_core_params(dcp, num_steps=num_steps)

            _core_params.append(unit_core_parameters)
            # print(f"{custom_core_key} with {ccp}")
    else:
        for _ in range(num_agents):
            dcp = deepcopy(default_core_cfg)
            unit_core_parameters, _ = get_core_params(dcp, num_steps=num_steps)
            _core_params.append(unit_core_parameters)

    # Return the relevant dicts
    return _experiment_params, _config_params, _core_params


def main():
    # %% Load configuration from YAML
    parser = argparse.ArgumentParser(
        description="Run NeurOptimiser experiments using COCO."
    )
    parser.add_argument("config",
                        type=str,
                        help="Path to the YAML config file")
    parser.add_argument("number_batches",
                        type=int,
                        default=1,
                        help="Number of batches to run")
    parser.add_argument("current_batch",
                        type=int,
                        default=1,
                        help="Current batch number to execute")
    args = parser.parse_args()

    # %% Load configuration from YAML
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, args.config)
    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)

    number_batches = args.number_batches
    current_batch = args.current_batch

    # %% Extract configurations
    experiment_params, config_params, core_params = process_parameters(experiment_config)
    print("[haki] Experiment parameters:", experiment_params)
    print("[haki] Optimiser configuration:", config_params)
    print(f"[haki] Loaded {len(core_params)} core configurations")

    # %% Prepare the COCO problem suite
    function_indices_str = to_coco_str(experiment_params["func_indices"])
    instances_str = to_coco_str(experiment_params["instance"])
    dimensions_str = to_coco_str(experiment_params["dimensions"])

    # Load the COCO suite
    suite = cocoex.Suite(
        suite_name=experiment_params["suite_name"],
        suite_instance=f"instances: {instances_str}",
        suite_options=f"function_indices: {function_indices_str} dimensions: {dimensions_str}",
    )

    # Set up the output folder and observer
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_str = ('_batch{:0' + str(len(str(number_batches - 1))) + '}of{}').format(
        current_batch, number_batches - 1) if number_batches > 1 else ''
    output_folder = (  # f"{date_str}_"
        f"{experiment_params['name']}_"
        f"{dimensions_str.replace(',', '')}D_"
        f"{function_indices_str.replace(',', '')}F_"
        # f"{experiment_params['budget_multiplier']}B_"
        f"on_{experiment_params['suite_name']}"
        f"{batch_str}"
        # f"{instances_str.replace(',', '-')}I_"
        # f"{config_params['num_agents']}P_"
        # f"{config_params['num_iterations']}S"
    )
    folder_path = os.path.join(output_folder)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    print(f"[haki] Output folder name: {folder_path}")

    observer_name = cocoex.default_observers()[experiment_params["suite_name"]]

    observer = cocoex.Observer(
        name=observer_name,
        options=f"result_folder: {folder_path} "
                f"algorithm_name: {experiment_params['name']}",
    )

    # Set up the experiment repeater
    repeater = cocoex.ExperimentRepeater(
        budget_multiplier=experiment_params["budget_multiplier"],
        max_sweeps=1,
    )

    # Set up the batcher
    batcher = cocoex.BatchScheduler(number_batches, current_batch)
    minimal_print = cocoex.utilities.MiniPrint()
    timings = collections.defaultdict(list)  # key is the dimension
    final_conditions = collections.defaultdict(list)  # key is (id_fun, dimension, id_inst)
    cocoex.utilities.write_setting(locals(), [observer.result_folder, 'parameters.pydat'])

    # %% Run the experiment
    print(f"[haki] Starting experiment {experiment_params['name']}...")
    time0 = time.time()
    while not repeater.done():  # while budget is left and successes are few
        for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
            if not batcher.is_in_batch(problem) or repeater.done(problem):
                continue  # skip this problem

            # %% Initialise the solver
            optimiser = NeurOptimiser(config_params, core_params)

            problem.observe_with(observer)  # generate data for cocopp
            problem(problem.dimension * [0])  # for better comparability

            num_iterations = problem.dimension * config_params["num_iterations"]
            # num_iterations = config_params["num_iterations"]

            time_start = time.time()
            xopt, fxopt = optimiser.solve(
                obj_func=problem,
                exp_name=experiment_params["name"],
                num_iterations=num_iterations,
                debug_mode=experiment_params["debug_mode"],
            )
            time_end = time.time()

            elapsed_time = time_end - time_start
            evals_used = problem.evaluations
            if evals_used == 1:
                # Estimate it
                evals_used = num_iterations * config_params["num_agents"]

            # problem(xopt)  # make sure the returned solution is evaluated

            if repeater._sweeps == 1:  # only for the first sweep
                timings[problem.dimension].append(elapsed_time / evals_used)
            repeater.track(problem)  # track evaluations and final_target_hit
            minimal_print(problem)  # show progress

            final_conditions[problem.id_triple].append(repr([
                evals_used, elapsed_time, fxopt]))

            csv_file_path = os.path.join(observer.result_folder, 'final_conditions.csv')
            write_header = not os.path.exists(csv_file_path)
            with open(csv_file_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["function_id", "dimension", "instance", "evals_used", "elapsed_time", "fxopt"])
                for entry in final_conditions[problem.id_triple]:
                    values = eval(entry)
                    writer.writerow([problem.id_triple[0], problem.id_triple[1], problem.id_triple[2]] + values)

            del optimiser  # free memory
            problem.free()
            # timings.clear()
            final_conditions.clear()
            gc.collect()

    print("\nTiming summary over all functions without repetitions:\n"
          "  dimension  median time [seconds/evaluation]\n"
          "  -------------------------------------")
    for dimension in sorted(timings):
        ts = sorted(timings[dimension])
        print("    {:3}       {:.1e}".format(dimension, (ts[len(ts) // 2] + ts[-1 - len(ts) // 2]) / 2))
    print("  -------------------------------------")

    if number_batches > 1:
        print("\n*** Batch {} of {} batches finished in {}."
              " Make sure to run *all* batches (0..{}) ***".format(
            current_batch, number_batches - 1,
            cocoex.utilities.ascetime(time.time() - time0), number_batches - 1))
    else:
        print("\n*** Full experiment done in %s ***"
              % cocoex.utilities.ascetime(time.time() - time0))
    print("    Data written into {}".format(observer.result_folder))


if __name__ == "__main__":
    main()
