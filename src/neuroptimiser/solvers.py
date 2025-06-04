from abc import ABC, abstractmethod
from datetime import datetime
from copy import deepcopy
from types import NoneType
from typing import List, Dict, Tuple

from schema import And, Optional, Schema, SchemaError

import numpy as np
from neuroptimiser.core.models import (
    NeuroHeuristicUnit, TensorContractionLayer, NeighbourhoodManager, HighLevelSelection,
    SPK_CORE_OPTIONS
)
from neuroptimiser.utils import (
    trs2o, get_arch_matrix, ADJ_MAT_OPTIONS, DYN_MODELS_KIND
)

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.monitor.process import Monitor

class AbstractSolver(ABC):
    """
    Abstract base class for any solver in the Neuro Optimiser framework.
    This class defines the basic structure and methods that any solver should implement.
    It includes methods for setting up the problem, validating configuration parameters,
    and running the optimisation process.
    """
    _INVALID_VALUE = 1e9
    _experiment_name: str
    _num_dimensions: int
    _num_iterations: int
    _num_agents: int
    _num_neighbours: int
    _search_space: np.ndarray
    _obj_func: callable
    _original_obj_func: callable

    results: dict = None
    _config_params: dict = dict()
    _base_params_schema: dict = dict()
    _additional_params_schema: dict = dict()
    _params_schema: dict = dict()

    def __init__(self) -> None:
        super().__init__()

    @property
    def config_params(self) -> dict:
        return self._config_params

    @config_params.setter
    def config_params(self, new_config_params: dict) -> None:
        self._set_config_params(new_config_params)

    @staticmethod
    def _rescale_problem(function, search_space=None):
        if search_space is None:
            return function
        else:
            lb = search_space[:, 0]
            ub = search_space[:, 1]

            def scaled_problem(x):
                x_real = trs2o(x, lb, ub)
                return function(x_real)

            return scaled_problem

    @staticmethod
    def _scale_position(position, search_space):
        lb = search_space[:, 0]
        ub = search_space[:, 1]
        return trs2o(position, lb, ub)

    def get_default_config(self) -> dict:
        """
        Returns the default configuration parameters for the solver.
        """

        config = self.validate_config_params(dict())
        config["search_space"] = config["search_space"][0,:].reshape(1, 2)
        return config

    def _set_config_params(self, config_params: dict) -> None:
        if config_params is None:
            config_params = dict()
        self._config_params = self.validate_config_params(config_params)

    def reset_random_seed(self, seed: int = None) -> None:
        """
        Resets the random seed for reproducibility.
        """
        if seed is None:
            seed = self._config_params["seed"]
        np.random.seed(seed)

    def _get_initial_positions(self) -> np.ndarray:
        initial_positions = np.random.uniform(
            low=-1.0, high=1.0, size=
            (self._num_agents, self._num_dimensions))
        return initial_positions

    def _get_position_masks(self) -> np.ndarray:
        position_masks = np.eye(self._num_dimensions)
        return position_masks

    @staticmethod
    def generate_experiment_name(prefix="Exp") -> str:
        """
        Generates a unique experiment name based on the current date and time.
        """
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment_name = f"{prefix}_{formatted_time}"
        return experiment_name

    def __call__(self, problem, exp_name: str = None, num_iterations: int = None,):
        return self.solve(problem, exp_name, num_iterations)

    def _check_config_params(self, obj_func, exp_name, num_iterations, search_space):
        if num_iterations is not None:
            self._config_params["num_iterations"] = num_iterations

        # If problem has not been provided, raise an error
        if obj_func is None:
            raise ValueError("obj_func must be provided")

        elif not callable(obj_func):
            raise ValueError("obj_func must be a callable function")

        if hasattr(obj_func, "bounds") or hasattr(obj_func, "lower_bounds"):
            if hasattr(obj_func, "bounds"):
                boundaries = np.array([obj_func.bounds.lb, obj_func.bounds.ub]).T # we assume that obj_func is an IOH problem
            else:
                boundaries = np.array([obj_func.lower_bounds, obj_func.upper_bounds]).T # we assume that obj_func is a COCOEX problem
            self._config_params["search_space"] = boundaries
            self._config_params["num_dimensions"] = boundaries.shape[0]

        elif search_space is not None:
            self._config_params["search_space"] = search_space
        else:
            # Warn the user that the search space is not provided and use the default search space
            # print(f"Search space is not provided. Using the default search space: {self._config_params['search_space']}")
            # Use default
            pass

        self._original_obj_func = obj_func # Original function (no scaled)
        self._obj_func = self._rescale_problem(
            obj_func, self._config_params["search_space"])
        self._config_params["function"] = self._obj_func

        # Generate the experiment name
        self._experiment_name = self.generate_experiment_name(exp_name if exp_name is not None else "Exp")

        # Validate the configuration parameters
        self._set_config_params(self._config_params)

        # Extract the configuration parameters
        self._num_dimensions    = self._config_params["num_dimensions"]
        self._num_iterations    = self._config_params["num_iterations"]
        self._num_agents        = self._config_params["num_agents"]
        self._num_neighbours    = self._config_params["num_neighbours"]
        self._search_space      = self._config_params["search_space"]

    @staticmethod
    def validate_params(schema: dict, params: dict) -> dict:
        """
        Validates the parameters against the provided schema.

        Args:
            schema (dict): The schema to validate against.
            params (dict): The parameters to validate.

        Returns:
            The validated parameters.

        """
        schema = Schema(schema)
        try:
            params = schema.validate(params)
        except SchemaError as e:
            raise ValueError(f"Invalid parameters: {e}")
        return params

    def validate_config_params(self, config_params: dict) -> dict:
        """
        Validates the configuration parameters for the solver.

        Args:
            config_params (dict): The configuration parameters to validate.
        """
        self._base_params_schema = {
            Optional("search_space", default=np.array([[-1, 1]])):
                And(np.ndarray, lambda n: n.shape[1] > 1,
                    error="search_space must be an array with at least 2 columns"),
            Optional("function", default=None):
                And(lambda n: callable(n) or n is None,
                    error="function must be a callable function"),
            Optional("num_dimensions", default=2):
                And(int, lambda n: n > 1,
                    error="num_dimensions must be greater than 1"),
            Optional("num_iterations", default=300):
                And(int, lambda n: n > 0,
                    error="num_iterations must be greater than 0"),
            Optional("num_agents", default=10):
                And(int, lambda n: n > 1,
                    error="num_agents must be greater than 1"),
            Optional("num_neighbours", default=1):
                And(int, lambda n: n > 0,
                    error="num_neighbours must be greater than 0"),
            Optional("seed", default=69):
                And(int, lambda n: n > 0,
                    error="seed must be greater than 0"),
        }
        self._params_schema = self._base_params_schema
        self._params_schema.update(self._additional_params_schema)

        config_params = self.validate_params(schema=self._params_schema, params=config_params)

        # Validation for the search space
        if config_params["search_space"].shape[0] == 1:
            # If only one row is provided, then it is assumed that the search space is the same for all dimensions
            config_params["search_space"] = np.tile(config_params["search_space"], (config_params["num_dimensions"], 1))

        elif config_params["search_space"].shape[0] != config_params["num_dimensions"]:
            raise ValueError("search_space must have the same number of rows as num_dimensions")

        return config_params

    @abstractmethod
    def solve(self, obj_func, exp_name: str, num_iterations: int = None,
              search_space: np.ndarray = None) -> tuple[float, np.ndarray]:
        """
        Abstract method to solve the optimisation problem.
        """
        pass

class NeurOptimiser(AbstractSolver):
    """
    Neuro Optimiser is the main class for the Neuro Optimiser framework.
    It inherits from the AbstractSolver class and implements the solve method, and sets up the model for the optimisation process.

    Args:
        config_params (dict): Configuration parameters for the Neuro Optimiser.
        core_params (dict|list[dict]): Parameters for the spiking core of the nheuristic units.
        selector_params (dict): Parameters for the high-level selection unit.
    """
    def __init__(self,
                 config_params: dict = None,
                 core_params: dict|list[dict] = None,
                 selector_params: dict = None
                 ) -> None:

        super().__init__()

        # Add the additional parameters for the Neuro Optimiser
        self.debug_mode = False
        self._profiler = None
        self._additional_params_schema = {
            Optional("spiking_core", default="TwoDimSpikingCore"):
                And(str, lambda n: n in SPK_CORE_OPTIONS,
                    error=f"spiking_core must be one of {SPK_CORE_OPTIONS}"),
            Optional("unit_topology", default="2dr"):
                And(str, lambda n: n in ADJ_MAT_OPTIONS,
                    error=f"unit_topology must be one of {ADJ_MAT_OPTIONS}"),
            Optional("neuron_topology", default="2dr"):
                And(str, lambda n: n in ADJ_MAT_OPTIONS,
                    error=f"neuron_topology must be one of {ADJ_MAT_OPTIONS}"),
            Optional("core_params", default=dict()):
                And(dict, error="core_params must be a dictionary"),
        }
        self._set_config_params(config_params)

        _strs_ext_models_A = [x + "_all" for x in DYN_MODELS_KIND] + ["random", "random_all"]
        _strs_models_A = DYN_MODELS_KIND + _strs_ext_models_A

        # Validate the configuration parameters for neurons
        core_params_schema = {
            Optional("alpha", default=1.0):
                And(lambda n: isinstance(n, (float, int)),
                    error="alpha must be a number"),
            Optional("dt", default=0.01):
                And(lambda n: isinstance(n, (float, int)), lambda n: n > 0.0,
                    error="dt must be a number greater than 0"),
            Optional("max_steps", default=100):
                And(lambda n: isinstance(n, (float, int)), lambda n: n > 1.0,
                    error="max_steps must be a number greater than 1"),
            Optional("noise_std", default=0.1):
                And(lambda n: isinstance(n, (float, int, tuple, list)),
                    error="noise_std must be a number greater than or equal to 0"),
            Optional("seed", default=None):
                And(lambda n: isinstance(n, (int, NoneType)),
                    error="seed must be an integer or None"),
            Optional("name", default="linear"):
                And(lambda n: n in ["linear", "izhikevich"],
                    error="system_params must either 'linear' or 'izhikevich'"),
            Optional("coeffs", default="random"):
                And(lambda n: isinstance(n, (str, list)),
                    error="coeffs must be a string or a list with the coefficients or the type of model"),
            Optional("approx", default="rk4"):
                And(lambda n: n in ["rk4", "euler"] or n is None,
                    error="approx must be either 'rk4', 'euler' or None"),
            Optional("ref_mode", default="pg"):
                And(lambda n: n in [None, "p", "g", "pg", "pgn"],
                    error="ref_mode must be either None, 'p', 'g', 'pg' or 'pgn'"),
            Optional("thr_mode", default="diff_pg"):
                And(lambda n: n in ["fixed", "adaptive_time", "adaptive_stag", "diff_pg", "diff_pref", "random"],
                    error="thr_mode must be either 'fixed', 'adaptive_time', 'adaptive_stag', 'diff_pg', 'diff_pref', 'random'"),
            Optional("thr_alpha", default=1.0):
                And(lambda n: isinstance(n, (float, int)) and n > 0.0,
                    error="thr_alpha must be a positive float or int"),
            Optional("thr_min", default=1e-6):
                And(lambda n: isinstance(n, (float, int)) and n >= 0.0,
                    error="thr_min must be a non-negative float or int"),
            Optional("thr_max", default=1.0):
                And(lambda n: isinstance(n, (float, int)) and n > 0.0,
                    error="thr_max must be a positive float or int"),
            Optional("thr_k", default=0.05):
                And(lambda n: isinstance(n, (float, int)) and n >= 0.0,
                    error="thr_k must be a non-negative float or int"),
            Optional("spk_cond", default="fixed"):
                And(lambda n: n in ["fixed", "l1", "l2", "l2_gen", "random", "adaptive", "stable"],
                    error="spk_cond must be either 'fixed', 'l1', 'l2', 'l2_gen', 'random', 'adaptive', and 'stable'"),
            Optional("spk_alpha", default=0.25):
                And(lambda n: isinstance(n, (float, int)) and n >= 0.0,
                    error="spk_alpha must be a non-negative float or int"),
            Optional("hs_operator", default="fixed"):
                And(lambda n: n in ["fixed", "directional", "differential", "random"],
                    error="hs_operator must be either 'fixed', 'directional', 'differential', 'random'"),
            Optional("hs_variant", default="fixed"):
                And(lambda n: isinstance(n, str),
                    error="hs_variant must be a string"),
            Optional("is_bounded", default=False):
                And(lambda n: isinstance(n, bool),
                    error="is_bounded must be a boolean"),
        }

        if core_params is None:
            repeated_core_params = [{}] * self._config_params['num_agents']
        elif isinstance(core_params, dict):
            repeated_core_params = [core_params] * self._config_params['num_agents']
        elif isinstance(core_params, list):
            if len(core_params) == self._config_params['num_agents']:
                repeated_core_params = core_params
            elif len(core_params) == 1:
                repeated_core_params = core_params * self._config_params['num_agents']
            else:
                raise ValueError(
                    "core_params must be a list of the same length as the number of agents, a single dict, or a list with one dictionary to be repeated")
        else:
            raise ValueError("core_params must be either None, a dictionary, or a list of dictionaries")

        self.core_params = [
            self.validate_params(core_params_schema, sub_core_params)
            for sub_core_params in repeated_core_params
        ]

        # TODO: Verify the selector_params
        if selector_params is None:
            selector_params = {}
        self.selector_params = selector_params

    def get_default_params(self) -> tuple[list[dict], dict]:
        """
        Returns the default parameters for the Neuro Optimiser.

        Returns:
            tuple[list[dict], dict]:
            A tuple containing:
            - A list of dictionaries, each representing the parameters of a Neuromorphic Heuristic Unit (NHU).
            - A dictionary for the high-level selector parameters (currently not implemented).

        Example:
            >>> solver = NeurOptimiser()
            >>> core_params, selector_params = solver.get_default_params()
            >>> print(core_params)
            >>> # Output:
            >>> # [{'noise_std': 0.1, 'thr_k': 0.05, 'thr_min': 1e-06, 'coeffs': 'random', 'thr_alpha': 1.0, 'name': 'linear', 'is_bounded': False, 'hs_operator': 'fixed', 'max_steps': 100, 'spk_alpha': 0.25, 'spk_cond': 'fixed', 'thr_max': 1.0, 'ref_mode': 'pg', 'thr_mode': 'diff_pg', 'seed': None, 'hs_variant': 'fixed', 'approx': 'rk4', 'alpha': 1.0, 'dt': 0.01}]
            >>> print(selector_params)
            >>> # Output:
            >>> # {}
        """
        return [self.core_params[0]], self.selector_params


    def _build_model(self):
        if self.debug_mode:
            print("[neuropt:log] Parameters are set up.")

        # 0. SETTING UP THE PARAMETERS
        # 0.1. Generate the initial positions for each nheuristic unit
        initial_positions = self._get_initial_positions()

        # 0.2. Define the topology for the spiking neural networks
        neuralnet_adjacency = get_arch_matrix(
            length=self._num_agents,
            topology=self.config_params["neuron_topology"],
            num_neighbours=self._num_neighbours
        )

        # 0.3. Define the topology for the neighbourhood manager
        neighbourhood_adjacency = get_arch_matrix(
            length=self._num_agents,
            topology=self.config_params["unit_topology"],
            num_neighbours=self._num_neighbours
        )

        if self.debug_mode:
            print("[neuropt:log] Initial positions and topologies are set up.")

        # 1. SETTING UP THE BUILDING BLOCKS
        # 1.1. Instantiate the tensor contraction layer for the spiking neural networks
        self.neuralnet = TensorContractionLayer(
            weights=neuralnet_adjacency,
            shape=(self._num_agents, self._num_dimensions)
        )

        # 1.2. Instantiate the neighbourhood manager for the spiking neural networks
        self.neighbourhood = NeighbourhoodManager(
            weights=neighbourhood_adjacency,
            shape=(self._num_agents, self._num_dimensions)
        )

        # 1.3. Instantiate the high-level selection unit
        self.selection = HighLevelSelection(
            num_dimensions=self._num_dimensions,
            num_agents=self._num_agents
        )

        if self.debug_mode:
            print(
                "[neuropt:log] Tensor contraction layer, neighbourhood manager, and high-level selection unit are created.")

        # 1.4. Create the population of nheuritic units
        self.nhus = []
        for agent_id in range(self._num_agents):
            agent_config = deepcopy(self.config_params)
            agent_config["num_neighbours"] = self.neighbourhood.p_out.shape[0]
            agent_config["core_params"] = self.core_params[agent_id]
            agent_config["selector_params"] = self.selector_params
            agent_config["core_params"]["init_position"] = initial_positions[agent_id, :]
            agent_config["core_params"]["seed"] = agent_id + 69

            unit = NeuroHeuristicUnit(
                agent_id=agent_id,
                name=f"nhu_{agent_id}",
                # SpikingCore=self.config_params["spiking_core"],
                **agent_config
            )
            self.nhus.append(unit)

        if self.debug_mode:
            print("[neuropt:log] Population of nheuristic units is created.")

        # 2. SETTING UP THE CONNECTIONS
        # 2.1. Connect the nheuristic units to the spiking neural networks
        for unit in self.nhus:
            # Wire spiking signals
            unit.s_out.connect(self.neuralnet.s_in)
            self.neuralnet.a_out.connect(unit.a_in)

            # Wire the neighbourhood manager
            unit.p_out.connect(self.neighbourhood.p_in)
            unit.fp_out.connect(self.neighbourhood.fp_in)

            self.neighbourhood.p_out.connect(unit.pn_in)
            self.neighbourhood.fp_out.connect(unit.fpn_in)

            # Wire the high-level selection unit
            unit.p_out.connect(self.selection.p_in)
            unit.fp_out.connect(self.selection.fp_in)

            self.selection.g_out.connect(unit.g_in)
            self.selection.fg_out.connect(unit.fg_in)

        if self.debug_mode:
            print("[neuropt:log] Connections between nheuristic units and auxiliary processes are established.")

        # 3. SETTING UP THE RUN CONFIGURATION
        self.rcfg = Loihi2SimCfg(
            select_tag='floating_pt',
            select_sub_proc_model=True,
        )

    def _reset_processes(self):
        # Reset all processes if the model exists
        for unit in self.nhus:
            if hasattr(unit, "reset"):
                unit.reset()
        if hasattr(self.selection, "reset"):
            self.selection.reset()

    def _stop_processes(self):
        # Stop processes
        for unit in self.nhus:
            unit.stop()
        self.selection.stop()
        self.neuralnet.stop()
        self.neighbourhood.stop()

    def _run_debug_mode(self):
        monitor_p = Monitor()
        monitor_g = Monitor()
        monitor_fp = Monitor()
        monitor_fg = Monitor()
        monitor_S = Monitor()

        monitors_v1 = [Monitor() for _ in range(self._num_agents)]
        monitors_v2 = [Monitor() for _ in range(self._num_agents)]

        monitor_p.probe(
            target=self.selection.p, num_steps=self._num_iterations)
        monitor_g.probe(
            target=self.selection.g, num_steps=self._num_iterations)
        monitor_fp.probe(
            target=self.selection.fp, num_steps=self._num_iterations)
        monitor_fg.probe(
            target=self.selection.fg, num_steps=self._num_iterations)
        monitor_S.probe(
            target=self.neuralnet.s_matrix,
            num_steps=self._num_iterations)

        for i in range(self._num_agents):
            monitors_v1[i].probe(
                target=self.nhus[i].v1, num_steps=self._num_iterations)
            monitors_v2[i].probe(
                target=self.nhus[i].v2, num_steps=self._num_iterations)

        print("[neuropt:log] Monitors are set up.")

        print(f"[neuropt:log] Starting simulation with {self._num_iterations} iterations...")

        # STEP 4: Run the simulation
        t = 0
        current_fg = None
        for t in range(self._num_iterations):
            self.selection.run(condition=RunSteps(num_steps=1), run_cfg=self.rcfg)

            current_fg = self.selection.fg.get()[0]
            if t % (self._num_iterations // 10) == 0 or t == self._num_iterations - 1:
                print(f"... step: {t}, best fitness: {current_fg}")

        print("[neuropt:log] Simulation completed. Fetching monitor data...",
              end=" ")
        # STEP 5: Read data
        p_data = monitor_p.get_data()
        g_data = monitor_g.get_data()

        fp_data = monitor_fp.get_data()
        fg_data = monitor_fg.get_data()

        v1_data = [x.get_data() for x in monitors_v1]
        v2_data = [x.get_data() for x in monitors_v2]

        s_data = monitor_S.get_data()

        # Prepare data
        p = [x["p"] for x in p_data.values()][0]
        g = [x["g"] for x in g_data.values()][0]
        fp = [x["fp"] for x in fp_data.values()][0]
        fg = [x["fg"] for x in fg_data.values()][0]

        v1 = [[x["v1"] for x in x.values()][0] for x in v1_data]
        v2 = [[x["v2"] for x in x.values()][0] for x in v2_data]

        spikes = [x["s_matrix"] for x in s_data.values()][0]

        monitor_p.stop()
        monitor_g.stop()
        monitor_fp.stop()
        monitor_fg.stop()
        monitor_S.stop()

        for i in range(self._num_agents):
            monitors_v1[i].stop()
            monitors_v2[i].stop()

        print("done")

        self._stop_processes()

        return p, fp, g, fg, v1, v2, spikes

    def _run_production_mode(self):
        # Run the simulation for all steps at once
        self.selection.run(condition=RunSteps(num_steps=self._num_iterations), run_cfg=self.rcfg)

        # STEP 5: Read data
        p = self.selection.p.get()
        fp = self.selection.fp.get()
        g = self.selection.g.get()
        fg = self.selection.fg.get()
        v1 = []
        v2 = []
        spikes = []

        self._stop_processes()

        return p, fp, g, fg, v1, v2, spikes

    def _process_positions(self, p, g):
        # Convert positions to the real search space
        g_original = [self._scale_position(gx, self._search_space) for gx in g]
        p_original = [self._scale_position(px, self._search_space) for px in p]

        return g_original, p_original

    def solve(self, obj_func,
              exp_name: str = None,
              num_iterations: int = None,
              search_space: np.ndarray = None,
              debug_mode: bool = False) -> Tuple[float, np.ndarray]:
        """
        Solve the optimisation problem using the Neuro Optimiser framework.

        Args:
            obj_func (callable): The objective function to be optimised.
            exp_name (str, optional): Name of the experiment. Defaults to None.
            num_iterations (int, optional): Number of iterations for the optimisation process. Defaults to None.
            search_space (np.ndarray, optional): Search space for the optimisation. Defaults to None.
            debug_mode (bool, optional): If True, enables debug mode with additional logging and monitoring. Defaults to False.
        Returns:
            Tuple[float, np.ndarray]: A tuple containing the best position found and its corresponding fitness value.
        """

        if debug_mode:
            print("[neuropt:log] Debug mode is enabled. Monitoring will be activated.")

        # Step 1: Set up the model
        self._check_config_params(obj_func, exp_name, num_iterations, search_space)
        self.debug_mode = debug_mode

        # Build the model only if it hasn't been built already
        self._build_model()

        if self.debug_mode:
            p, fp, g, fg, v1, v2, spikes = self._run_debug_mode()
        else:
            p, fp, g, fg, v1, v2, spikes = self._run_production_mode()

        g_original, p_original = self._process_positions(p, g)

        best_position   = g_original[-1]
        best_fitness    = fg[-1]

        # Prepare the results dictionary
        self.results = {
            "experiment_name": self._experiment_name,
            "best_position": best_position,
            "best_fitness": best_fitness,
            "p": p_original,
            "fp": fp,
            "g": g_original,
            "fg": fg,
            "v1": v1,
            "v2": v2,
            "s": spikes
        }

        return best_position, best_fitness
