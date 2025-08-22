"""Neuroptimiser Core Models

This module contains the core process models for the Neuroptimiser framework based on Lava.
"""
__author__ = "Jorge M. Cruz-Duarte"
__email__ = "jorge.cruz-duarte@univ-lille.fr"
__version__ = "1.0.0"
__all__ = ["AbstractPerturbationNHeuristicModel", "PyTwoDimSpikingCoreModel",
           "PySelectorModel", "PyHighLevelSelectionModel",
           "SubNeuroHeuristicUnitModel", "PyTensorContractionLayerModel",
           "PyNeighbourhoodManagerModel", "PySpikingHandlerModel",
           "PyPositionReceiverModel", "PyPositionReceiverModel"]

from collections.abc import Callable

import numpy as np
from functools import partial
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires

from neuroptimiser.core.processes import (
    TwoDimSpikingCore,
    Selector, HighLevelSelection,
    NeuroHeuristicUnit,
    TensorContractionLayer, NeighbourhoodManager,
    SpikingHandler, PositionSender, PositionReceiver
)

_INVALID_VALUE = 1e9
_EPS = 1e-12
_POS_FLOAT_ = np.float32
_FIT_FLOAT_ = np.float32
SPK_CORE_OPTIONS = ["TwoDimSpikingCore"]  # To be extended with more options

class AbstractPerturbationNHeuristicModel(PyLoihiProcessModel):
    """Abstract model for a perturbation-based nheuristic process model

    This model serves as a base for implementing various perturbation-based process models.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.AbstractSpikingCore`:
        Abstract process that implements a spiking core for perturbation-based nheuristics.
    """

    # Inports
    s_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, bool)
    p_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)
    g_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fg_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)
    xn_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fxn_in: PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    # Variables
    x:      np.ndarray      = LavaPyType(np.ndarray, float)

    # Outports
    s_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, bool)
    x_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)

    def __init__(self, proc_params):
        """Initialises the process model with the given parameters.

        Arguments
        ---------
            proc_params : dict
                A dictionary containing the parameters for the process model. It should include:
                    - ``seed``: int, random seed for reproducibility
                    - ``num_dimensions``: int, number of dimensions for the perturbation
                    - ``num_neighbours``: int, number of neighbours for the perturbation
        """
        super().__init__(proc_params)

        self.initialised       = False
        self.seed              = proc_params["seed"]
        self.num_dimensions    = proc_params['num_dimensions']
        self.num_neighbours    = proc_params['num_neighbours']
        self.shape             = (self.num_dimensions,)

        self.p                 = np.zeros(self.shape).astype(float)
        self.threshold         = np.zeros(self.shape).astype(float)
        self.self_fire         = np.zeros(self.shape).astype(bool)
        self.compulsory_fire   = np.zeros(self.shape).astype(bool)

        np.random.default_rng(seed=int(self.seed))

HS_DEFAULT_PARAMS = {
    "fixed": {
        "jitter_scale": 0.0,            # add small noise after selecting point
    },
    "random": {
        "normal_std": 1.0,
        "uniform_low": -1.0,
        "uniform_high": 1.0,
    },
    "directional": {
        "alpha_scale": 1.0,           # scaling factor for the direction
        "jitter_scale": 0.0,          # add small noise after selecting point
    },
    "differential": {
        "F": None,                    # actual F value, if not None, it # overrides _get_F and other F-related params lost their effect
        "F_low": 0.0,                 # overrides _get_F range if provided
        "F_high": 2.5,
        "jitter_scale": 0.0,         # add small noise after selecting point
    },
    "swarm": {
        "phi1": 1.45,                 # cognitive weight
        "phi2": 1.55,                 # social weight
        "phi3": 1.50,                 # neighbourhood weight
        "w": 0.67,                    # inertia-like term
        "jitter_scale": 0.0,          # add small noise after selecting point
    },
    # //////////////////////// WORK IN PROGRESS /////////////////////////
    # "cma": {
    #     "sigma_init": 0.3,     # initial global step-size
    #     "eta_m": 0.25,         # mean drift rate towards guidance
    #     "c1": 0.2,             # rank-1 covariance update weight
    #     "c_sigma": 0.3,        # step-size adaptation rate
    #     "d_sigma": 1.0,        # damping (kept simple)
    #     "min_sigma": 1e-4,
    #     "max_sigma": 3.0,
    #     "target_success": 0.2, # (1+1) success rule target
    #     "forget_rate": 0.1,    # slight forgetting on failure
    #     "jitter_scale": 0.0    # optional tiny Gaussian noise on output
    # },
    # //////////////////////// WORK IN PROGRESS /////////////////////////
}

@implements(proc=TwoDimSpikingCore, protocol=LoihiProtocol)
@requires(CPU)
class PyTwoDimSpikingCoreModel(AbstractPerturbationNHeuristicModel):
    """Two-dimensional spiking core model for perturbation-based nheuristics

    This model implements a two-dimensional spiking core for perturbation-based nheuristics, allowing for various dynamic systems and approximation methods.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.TwoDimSpikingCore`:
        Process that implements a two-dimensional spiking core for perturbation-based nheuristics.
    """

    # Variables
    v1       : np.ndarray = LavaPyType(np.ndarray, float)
    v2       : np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        """Initialises the two-dimensional spiking core model with the given parameters.

        Arguments
        ---------
            proc_params : dict
                A dictionary containing the parameters for the process model. It should include:
                    - ``alpha``: float, scaling factor for the perturbation
                    - ``max_steps``: int, maximum number of steps for the dynamic system
                    - ``noise_std``: float, standard deviation of the noise added to the perturbation
                    - ``name``: str, name of the dynamic system (e.g., "linear", "izhikevich")
                    - ``approx``: str, approximation method for the dynamic system (e.g., "euler", "rk4")
                    - ``dt``: float, time step for the approximation
                    - ``ref_mode``: str, reference model for the perturbation (e.g., "p", "g", "pg", "pgn")
                    - ``thr_mode``: str, threshold mode for the spiking condition (e.g., "fixed", "adaptive_time", "adaptive_stag", "diff_pg", "diff_pref", "random")
                    - ``thr_alpha``: float, scaling factor for the threshold
                    - ``thr_min``: float, minimum threshold value
                    - ``thr_max``: float, maximum threshold value
                    - ``thr_k``: float, scaling factor for the threshold
                    - ``spk_cond``: str, spiking condition (e.g., "fixed", "l1", "l2", "l2_gen", "random", "adaptive", "stable")
                    - ``spk_alpha``: float, scaling factor for the spiking condition
                    - ``hs_operator``: str, heuristic search operator (i.e., "fixed", "random", "directional", "differential")
                    - ``hs_variant``: str, variant of the heuristic search operator. The available variants when
                        - ``hs_operator``="fixed" are "gbest", "center", "pgbest", "pbest"|"fixed" (default).
                        - ``hs_operator``="random" are "uniform" and "normal" (default).
                        - ``hs_operator``="directional" are "pbest", "gbest"|"fixed" (default), and "center".
                        - ``hs_operator``="differential" are "rand", "current", "pbest", "best", "current-to-rand", "best-to-rand", "pbest-to-rand", "current-to-best" (default), "current-to-pbest", "rand-to-best", and "rand-to-pbest".
                        - ``hs_operator``="swarm" are "cognitive", "social", "cogsocial" (default), and "neighbourhood".
                    - ``is_bounded``: bool, whether the perturbation is bounded (default: True)

        """
        super().__init__(proc_params)
        self.alpha      = proc_params['alpha']
        self.max_steps  = proc_params['max_steps']
        self.noise_std  = proc_params['noise_std']

        self.system     = proc_params['name']
        self.approx     = proc_params['approx']
        self.dt         = proc_params['dt']
        self.ref_model  = proc_params['ref_mode']

        self.thr_mode   = proc_params['thr_mode']
        self.thr_alpha  = proc_params['thr_alpha']
        self.thr_min    = proc_params['thr_min']
        self.thr_max    = proc_params['thr_max']
        self.thr_k      = proc_params['thr_k']

        self.spk_cond   = proc_params['spk_cond']
        self.spk_alpha  = proc_params['spk_alpha']

        self.hs_operator= proc_params['hs_operator']
        self.hs_variant = proc_params['hs_variant']

        self.is_bounded = proc_params.get("is_bounded", True)

        # Initialise the dynamic model
        if self.system == "linear":
            self.models_coeffs   = proc_params['coeffs_values']
            self.stable     = [np.trace(model_A) < 0 for model_A in self.models_coeffs]
            model = self._two_dim_linear_system
        elif self.system == "izhikevich":
            self.models_coeffs   = proc_params['coeffs_values']
            self.stable = [False for _ in self.models_coeffs]
            model = self._izhikevich_system
        else:
            raise ValueError(f"Unknown system: {self.system}")

        self.models = []
        for dim in range(self.num_dimensions):
            _model_per_dim = partial(model, dim=dim)
            self.models.append(_model_per_dim)

        self.neighbour_indices_cache = []
        self._buffer_new_v_array = np.empty((self.num_dimensions, 2), dtype=float)

        # Read the parameters for the heuristic search operator
        hs_user_params = proc_params.get("hs_params", {})
        self._hs_params = dict(HS_DEFAULT_PARAMS.get(self.hs_operator, {}))
        self._hs_params.update(hs_user_params)

        # Configure the heuristic search operator
        match self.hs_operator:
            case "fixed":
                self._apply_hs = self._hs_fixed
            case "random":
                self._apply_hs = self._hs_random
            case "directional":
                self._apply_hs = self._hs_directional
            case "differential":
                self._apply_hs = self._hs_differential

                self._get_F = self._hs_params["F"]
                if self._get_F is None:
                    self._get_F = partial(self._sample_F,
                        F_low=self._hs_params["F_low"], F_high=self._hs_params["F_high"])

                if self.num_neighbours > 3:
                    for dim in range(self.num_dimensions):
                        triplet = np.random.choice(
                            range(self.num_neighbours), 3, replace=False)
                        self.neighbour_indices_cache.append(triplet)
            case "swarm":
                self._apply_hs = self._hs_swarm
            case "cma":
                self._apply_hs = self._hs_cma


                self._cma_m = np.zeros((self.num_dimensions, 2)).astype(float)  # mean
                self._cma_C = np.stack([np.eye(2) for _ in range(self.num_dimensions)]).astype(float) # covariance
                self._cma_sigma = np.full(self.num_dimensions, self._hs_params["sigma_init"], dtype=float)  # step-size
                self._cma_initialised = np.zeros(self.num_dimensions, dtype=bool)

                self._cma_pc = np.zeros((self.num_dimensions, 2)).astype(float)  # evolution path for covariance
                self._cma_ps = np.zeros((self.num_dimensions, 2)).astype(float)  # evolution path for sigma

            case _:
                raise ValueError(f"Unknown hs_operator: {self.hs_operator}")

        # print(self._hs_params)

        # Initialise the approximation method
        if self.approx == "euler":
            self.approx_method = self._euler_approximation
        elif self.approx == "rk4":
            self.approx_method = self._rk4_approximation
        else:
            raise ValueError(f"Unknown approximation method: {self.approx}")

        self._spike_condition   = self._init_spike_condition()

        self.v1_pbest           = np.zeros(self.shape).astype(float)
        self.v1_gbest           = np.zeros(self.shape).astype(float)
        self.v2_pbest           = np.random.uniform(-1.0, 1.0, size=self.shape).astype(float)
        self.v2_gbest           = np.random.uniform(-1.0, 1.0, size=self.shape).astype(float)

        self._base_threshold    = self.thr_alpha * np.ones_like(self.v1)
        self._threshold_fn      = self._init_threshold_fn()

        self.v_bounds           = np.zeros((2,self.shape[0])).astype(float)
        self.x_ref              = np.zeros(self.shape).astype(float)
        self.xn                 = np.zeros(
            shape=(self.num_neighbours, self.num_dimensions)).astype(float)
        self.vn                 = np.zeros(
            shape=(self.num_neighbours, self.num_dimensions)).astype(float)
        self.ref_point          = np.zeros(self.shape).astype(float)
        self.prev_p             = np.zeros(self.shape).astype(float)
        self.stag_count         = 0

    # TRANSFORMATION METHODS
    def _linear_transform(self, x):
        """Linear transformation of the input variable to the range [-1, 1]."""
        var = self.alpha * (x - self.x_ref)
        return var

    def _linear_transform_inv(self, var):
        """Inverse linear transformation of the variable from the range [-1, 1] to the original range."""
        return var / self.alpha + self.x_ref

    def _check_vbounds(self):
        """Check and update the bounds of the variables v1 and v2."""
        # Update the bounds
        self.v_bounds[0,:] = self._linear_transform(-1.0*np.ones(self.shape))
        self.v_bounds[1,:] = self._linear_transform(np.ones(self.shape))

        # Check the bounds
        self.v1 = np.clip(self.v1, self.v_bounds[0,:], self.v_bounds[1,:])
        self.v2 = np.clip(self.v2, self.v_bounds[0,:], self.v_bounds[1,:])

    def _send_to_ports(self):
        """Send the current state of the process to the output ports."""
        # Inverse transformation to send
        self.x  = self._linear_transform_inv(self.v1)

        # Send the new position
        self.s_out.send(self.self_fire)
        self.x_out.send(self.x)

    # APPROXIMATION METHODS
    def _euler_approximation(self, model, var, **kwargs):
        """Euler approximation of the system of equations."""
        return var + model(var) * self.dt

    def _rk4_approximation(self, model, var, **kwargs):
        """Runge-Kutta 4th order approximation of the system of equations."""
        k1 = model(var)
        k2 = model(var + k1/2 * self.dt)
        k3 = model(var + k2/2 * self.dt)
        k4 = model(var + k3 * self.dt)

        return var + (k1 + 2*k2 + 2*k3 + k4) / 6 * self.dt

    # 2D MODELS
    def _izhikevich_system(self, upsilon, dim, **kwargs):
        """Izhikevich model for two-dimensional spiking neurons."""
        # Read coefficients
        coeffs = self.models_coeffs[dim] if len(self.stable) > 1 else self.models_coeffs[0]

        # Characteristic values
        Lv = coeffs["vmax"] - coeffs["vmin"]
        Lu = coeffs["umax"] - coeffs["umin"]
        Lt = coeffs["Lt"]

        # Assumming upsilon is dimensionless
        v_, u_ = upsilon

        # Get the dimensional variables from (-1,1)
        v = Lv * (v_ + 1) / 2 + coeffs["vmin"]
        u = Lu * (u_ + 1) / 2 + coeffs["umin"]

        # Add the noise
        a_ = coeffs["a"]
        b_ = coeffs["b"]
        I_ = coeffs["I"] #+ np.random.uniform(0, 30)

        # Compute the model with these variables
        dv = 0.04 * v ** 2 + 5 * v + 140 - u + I_
        du = a_ * (b_ * v - u)

        # Nondimensionalise the variables
        dv_ = 2 * Lt * dv / Lv
        du_ = 2 * Lt * du / Lu

        return np.array([dv_, du_])

    def _two_dim_linear_system(self, upsilon, dim, **kwargs):
        """Two-dimensional linear system of equations."""
        system_matrix = self.models_coeffs[dim] \
            if len(self.stable) > 1 else self.models_coeffs[0]
        deriv_upsilon = system_matrix @ upsilon
        return deriv_upsilon

    def _hs_fixed(self, dim, var):
        """Fixed heuristic search operator for perturbation-based nheuristics.

        Variants (set via self.hs_variant):
            - "gbest"  : uses the global best (gbest) position
            - "center" : uses the center (0, 0) as the reference point
            - "pgbest" : uses the average of personal best (pbest) and global best (gbest)
            - "pbest"|"fixed" : uses the personal best (pbest) position
        """
        match self.hs_variant:
            case "gbest":
                new_var_1 = self.v1_gbest[dim]
                new_var_2 = self.v2_gbest[dim]
            case "center":
                new_var_1 = 0.0
                new_var_2 = 0.0
            case "pgbest":
                new_var_1 = (self.v1_pbest[dim] + self.v1_gbest[dim]) / 2.0
                new_var_2 = (self.v2_pbest[dim] + self.v2_gbest[dim]) / 2.0
            case _: #"pbest" | "fixed":
                new_var_1 = self.v1_pbest[dim]
                new_var_2 = self.v2_pbest[dim]

        new_var = np.array([new_var_1, new_var_2]) + np.random.normal(
            scale=self._hs_params["jitter_scale"], size= 2)
        return new_var

    def _hs_random(self, dim, var):
        """Random heuristic search operator for perturbation-based nheuristics.

        Variants (set via self.hs_variant):
            - "uniform" : generates random values from a uniform distribution
            - "normal"  : generates random values from a normal distribution
        """
        match self.hs_variant:
            case "uniform":
                new_var_1 = np.random.uniform(
                    low=self._hs_params["uniform_low"],
                    high=self._hs_params["uniform_high"])
                new_var_2 = np.random.uniform(
                    low=self._hs_params["uniform_low"],
                    high=self._hs_params["uniform_high"])
            case _: #"normal":
                new_var_1 = np.random.normal(
                    loc=0.0, scale=self._hs_params["normal_std"])
                new_var_2 = np.random.normal(
                    loc=0.0, scale=self._hs_params["normal_std"])

        new_var  = np.array([new_var_1, new_var_2])
        return new_var

    def _hs_directional(self, dim, var):
        """Directional heuristic search operator for perturbation-based nheuristics.

        Variants (set via self.hs_variant):
            - "pbest"  : pulls towards personal best (pbest)
            - "gbest"  : pulls towards global best (gbest) or fixed point
            - "center" : pulls towards the centre (0, 0)
        """
        match self.hs_variant:
            case "pbest":
                dir1 = self.v1_pbest[dim] - var[0]
                dir2 = self.v2_pbest[dim] - var[1]
            case "center":
                dir1 = 0.0 - var[0]
                dir2 = 0.0 - var[1]
            case _: # default: "gbest" or "fixed"
                dir1 = self.v1_gbest[dim] - var[0]
                dir2 = self.v2_gbest[dim] - var[1]

        new_var  = (np.array(var)
                    + self._hs_params["alpha_scale"]  * np.array([dir1, dir2])
                    + np.random.normal(scale=self._hs_params["jitter_scale"], size=2))
        return new_var

    @staticmethod
    def _sample_F(F_low, F_high):
        """Returns a random scaling factor for the heuristic search operator, particularly for the differential variant."""
        return np.random.uniform(low=F_low, high=F_high)

    def _hs_differential(self, dim, var):
        """Differential heuristic search operator for perturbation-based nheuristics.

        Variants (set via self.hs_variant):
            - "rand"                : uses random neighbours for the perturbation
            - "current"             : uses the current position as a reference
            - "pbest"               : uses the personal best (pbest) position
            - "best"|"gbest"     : uses the global best (gbest) position
            - "current-to-rand"     : pulls towards a random point from the current position
            - "best-to-rand"        : pulls towards a random point from the global best position
            - "pbest-to-rand"       : pulls towards a random point from the personal best position
            - "current-to-best"     : pulls towards the global best from the current position
            - "current-to-pbest"    : pulls towards the personal best from the current position
            - "rand-to-best"        : pulls towards the global best from a random point
            - "rand-to-pbest"       : pulls towards the personal best from a random point
        """
        # Get from neighbours
        if self.num_neighbours > 3:
            ind1, ind2, ind3  = self.neighbour_indices_cache[dim]
            pre_sum_1 = self.vn[ind1, dim]
            the_sum = self.vn[ind2, dim] - self.vn[ind3, dim]
        else:
            pre_sum_1 = np.random.normal(0, self.noise_std)
            the_sum = np.random.normal(0, self.noise_std)
        pre_sum_2 = pre_sum_1 + np.random.normal(0, self.noise_std)

        match self.hs_variant:
            case "rand":
                new_var1 = pre_sum_1
                new_var2 = pre_sum_2
            case "current":
                new_var1 = var[0]
                new_var2 = var[1]
            case "pbest":
                new_var1 = self.v1_pbest[dim]
                new_var2 = self.v2_pbest[dim]
            case "best": # "gbest"
                new_var1 = self.v1_gbest[dim]
                new_var2 = self.v2_gbest[dim]
            case "current-to-rand":
                new_var1 = var[0] + self._get_F() * (pre_sum_1 - var[0])
                new_var2 = var[1] + self._get_F() * (pre_sum_2 - var[1])
            case "best-to-rand":
                new_var1 = self.v1_gbest[dim] + self._get_F() * (pre_sum_1 - self.v1_gbest[dim])
                new_var2 = self.v2_gbest[dim] + self._get_F() * (pre_sum_2 - self.v2_gbest[dim])
            case "pbest-to-rand":
                new_var1 = self.v1_pbest[dim] + self._get_F() * (pre_sum_1 - self.v1_pbest[dim])
                new_var2 = self.v2_pbest[dim] + self._get_F() * (pre_sum_2 - self.v2_pbest[dim])
            case "current-to-pbest":
                new_var1 = var[0] + self._get_F() * (self.v1_pbest[dim]- var[0])
                new_var2 = var[1] + self._get_F() * (self.v2_pbest[dim] - var[1])
            case "rand-to-best":
                new_var1 = pre_sum_1 + self._get_F() * (self.v1_gbest[dim] - pre_sum_1)
                new_var2 = pre_sum_2 + self._get_F() * (self.v2_gbest[dim] - pre_sum_1)
            case "rand-to-pbest":
                new_var1 = pre_sum_1 + self._get_F() * (self.v1_pbest[dim] - pre_sum_1)
                new_var2 = pre_sum_2 + self._get_F() * (self.v2_pbest[dim] - pre_sum_1)
            case _: # "current-to-best"
                new_var1 = var[0] + self._get_F() * (self.v1_gbest[dim] - var[0])
                new_var2 = var[1] + self._get_F() * (self.v2_gbest[dim] - var[1])

        new_var1 += self._get_F() * the_sum
        new_var2 += self._get_F() * the_sum
        new_var = np.array([new_var1, new_var2]) + np.random.normal(
            scale=self._hs_params["jitter_scale"], size=2)
        return new_var

    def _hs_swarm(self, dim, var):
        """Swarm-style heuristic search operator (PSO-inspired).

        Variants (set via self.hs_variant):
            - "cognitive"         : pull towards personal best (pbest)
            - "social"            : pull towards global best (gbest)
            - "cogsocial" (default): combine cognitive + social
            - "neighbourhood"     : pull towards neighbourhood mean (if available, else gbest)
        """
        v1, v2 = var[0], var[1]

        # Cognitive (self) and social (global) directions
        pdir1 = self.v1_pbest[dim] - v1
        pdir2 = self.v2_pbest[dim] - v2
        gdir1 = self.v1_gbest[dim] - v1
        gdir2 = self.v2_gbest[dim] - v2

        # Neighbourhood signal (scalar per dim; it mirrors how differential uses vn)
        if self.num_neighbours > 0:
            nbar = float(np.mean(self.vn[:, dim]))
        else:
            nbar = self.v1_gbest[dim]  # sensible fallback

        # Random coefficients
        r1 = np.random.uniform(0.0, 1.0)
        r2 = np.random.uniform(0.0, 1.0)
        r3 = np.random.uniform(0.0, 1.0)

        # Coefficients
        phi1    = self._hs_params["phi1"]
        phi2    = self._hs_params["phi2"]
        phi3    = self._hs_params["phi3"]
        w       =   self._hs_params["w"]

        match self.hs_variant:
            case "cognitive":
                dv1 = phi1 * r1 * pdir1
                dv2 = phi1 * r1 * pdir2
            case "social":
                dv1 = phi2 * r2 * gdir1
                dv2 = phi2 * r2 * gdir2
            case "neighbourhood":
                # pull both components towards the same neighbourhood scalar
                dv1 = phi3 * r3 * (nbar - v1)
                dv2 = phi3 * r3 * (nbar - v2)
            case _:  # "cogsocial" by default
                dv1 = phi1 * r1 * pdir1 + phi2 * r2 * gdir1
                dv2 = phi1 * r1 * pdir2 + phi2 * r2 * gdir2

        # Mild inertia-like nudge towards staying close to current state
        dv1 += w * (v1 - 0.0)
        dv2 += w * (v2 - 0.0)

        # Assemble update with small scalar noise (same flavour as _hs_directional)
        new_var = np.array([v1 + dv1, v2 + dv2]) + np.random.normal(
            scale=self._hs_params["jitter_scale"], size=2)
        return new_var

    # //////////////////////// WORK IN PROGRESS /////////////////////////
    # def _hs_cma(self, dim, var):
    #     """CMA-ES operator (2D per dim) with variants:
    #        - online  : aCMA-like drift using pbest/gbest/neighbourhood
    #        - 1p1     : (1+1)-CMA style step-size control and rank-1 on success
    #        - recomb  : rank-µ-like mean via fitness-weighted recombination
    #     """
    #     # Initialise CMA state if needed
    #     if not self._cma_initialised[dim]:
    #         self._cma_m[dim, 0] = var[0]
    #         self._cma_m[dim, 1] = var[1]
    #         self._cma_initialised[dim] = True
    #
    #     # Shortcut references to CMA state
    #     m = self._cma_m[dim]
    #     C = self._cma_C[dim]
    #     sigma = self._cma_sigma[dim]
    #     pc = self._cma_pc[dim]
    #     ps = self._cma_ps[dim]
    #
    #     # Eigendecomposition of C to get A where C = A A^T
    #     try:
    #         A = np.linalg.cholesky(C + _EPS * np.eye(2))
    #     except np.linalg.LinAlgError:
    #         C[:] = np.eye(2)
    #         A = np.eye(2)
    #
    #     # Sample new point from N(m, sigma^2 C)
    #     z = np.random.normal(size=2)  # ~ N(0, I)
    #     y = A @ z  # ~ N(0, C)
    #     x_prop = m + sigma * y
    #
    #     # Extract reference points for guidance (elite-based)
    #     x_p = np.array([self.v1_pbest[dim], self.v2_pbest[dim]])
    #     x_g = np.array([self.v1_gbest[dim], self.v2_gbest[dim]])
    #     if self.num_neighbours > 0:
    #         nbar = float(np.mean(self.vn[:, dim]))
    #     else:
    #         nbar = self.v1_gbest[dim]
    #     x_n = np.array([nbar, nbar])
    #
    #     # Weighted mean of reference points (equal weights by default)
    #     w_p0, w_g0, w_n0 = 0.5, 0.5, 0.25
    #     x_mu    = (w_p0 * x_p + w_g0 * x_g + w_n0 * x_n) / (w_p0 + w_g0 + w_n0)
    #     y_w     = (x_mu - m) / (sigma + _EPS)
    #
    #     # CSA (paths and step-size), n = 2
    #     cs      = self._hs_params["c_sigma"]
    #     damps   = self._hs_params["d_sigma"]
    #     ps      = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs)) * z
    #     E_norm  = 1.2533141373155  # E||N(0, I_2)|| = sqrt(2)*Gamma(2)/Gamma(1)
    #
    #     # hsig: Heaviside
    #     # finite-horizon normaliser for gate
    #     t = float(getattr(self, "time_step", 0) + 1)
    #     chi = np.sqrt(1.0 - (1.0 - cs) ** (2.0 * t))
    #     if chi < 1e-12:
    #         chi = 1.0
    #     hsig = float(np.linalg.norm(ps) / chi < (1.4 + 2.0 / 3.0) * E_norm)
    #
    #     cc = self._hs_params["c_c"]
    #     pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc)) * y_w
    #
    #     # match self.hs_variant:
    #     # case "1p1":
    #     # success if personal best moved (greedy selector semantics)
    #     success = float(abs(self.p[dim] - self.prev_p[dim]) > 1e-15)
    #     target = self._hs_params["target_success"]
    #
    #     # 1/5th rule
    #     sigma = sigma * np.exp((cs / damps) * (success - target))
    #     sigma = float(np.clip(sigma, self._hs_params["min_sigma"], self._hs_params["max_sigma"]))
    #
    #     # covariance: strengthen on success, mild forgetting otherwise
    #     c1 = self._hs_params["c1"]
    #     forget = self._hs_params["forget_rate"]
    #     y_unit = y / (np.linalg.norm(y) + _EPS)
    #     if success >= target:
    #         C[:] = (1.0 - c1) * C + c1 * np.outer(y_unit, y_unit)
    #     else:
    #         C[:] = (1.0 - c1 * forget) * C
    #
    #     # mean drift
    #     m[:] = m + self._hs_params["eta_m"] * (x_mu - m)
    #
    #     # persist
    #     self._cma_m[dim] = m
    #     self._cma_C[dim] = C
    #     self._cma_sigma[dim] = sigma
    #     self._cma_pc[dim] = pc
    #     self._cma_ps[dim] = ps
    #
    #     # proposal (match flavour of other operators)
    #     new_var = x_prop + np.random.normal(scale=self._hs_params["jitter_scale"], size=2)
    #     return new_var
    # //////////////////////// WORK IN PROGRESS /////////////////////////

    # DYNAMIC HEURISTIC
    def _apply_hd(self, dim, var):
        """Applies the selected approximation to the dynamic system for a given dimension."""
        # Apply the model using an approximation, if so
        new_var = self.approx_method(self.models[dim], var, dt=self.dt)

        # Post-processing
        return new_var

    def _run_core_process(self):
        """Runs the core process of the two-dimensional spiking core model."""
        for dim in range(self.num_dimensions):
            # Get the spike condition for this neuron
            self.self_fire[dim]     = self._spike_condition(
                v1=self.v1[dim], v2=self.v2[dim], thr=self.threshold[dim], dim=dim)
            fire_condition          = self.self_fire[dim] or self.compulsory_fire[dim]

            upsilon = np.array([self.v1[dim], self.v2[dim]])
            if fire_condition:
                upsilon = self._apply_hs(dim, upsilon)
            else:
                upsilon = self._apply_hd(dim, upsilon)

            self.v1[dim], self.v2[dim] = upsilon

        if self.is_bounded:
            self._check_vbounds()

    def _transform_variables(self):
        """Transforms the variables to the appropriate range and prepares the reference point."""
        if not self.initialised:
            x_refs = np.vstack((self.p, self.g))
        else:
            x_refs = np.vstack((self.p, self.xn, self.g))

        if self.ref_model is None:
            self.x_ref = np.zeros(self.shape).astype(float)
        elif self.ref_model == "p":
            self.x_ref      = self.p.copy()
        elif self.ref_model == "g":
            self.x_ref      = self.g.copy()
        elif self.ref_model == "pg":
            self.x_ref = (self.p + self.g) / 2.0
        elif self.ref_model == "pgn":
            self.x_ref      = np.average(x_refs, axis=0)
        else:
            raise ValueError(f"Unknown reference model: {self.ref_model}")

        # Variable preparation
        self.v1         = self._linear_transform(self.x)
        self.v1_pbest   = self._linear_transform(self.p)
        self.v1_gbest   = self._linear_transform(self.g)

        if self.xn is not None:
            self.vn[:] = self._linear_transform(self.xn)
        self.ref_point  = self._linear_transform(np.average(x_refs, axis=0))

    # Threshold determination
    def _threshold_fixed(self):
        """Returns a fixed threshold value."""
        return self._base_threshold * np.ones_like(self.v1)

    def _threshold_adaptive_time(self):
        """Returns an adaptive threshold based on the time step."""
        scale = 1.0 / (1.0 + self.thr_k * (self.time_step + 1.0))
        return self._base_threshold * scale * np.ones_like(self.v1)

    def _threshold_adaptive_stag(self):
        """Returns an adaptive threshold based on the stagnation count."""
        scale = 1.0 + self.thr_k * self.stag_count
        return self._base_threshold * scale * np.ones_like(self.v1)

    def _threshold_diff_pg(self):
        """Returns a threshold based on the difference between the current position and the global best."""
        return self.thr_alpha * np.abs(self.p - self.g)

    def _threshold_diff_pref(self):
        """Returns a threshold based on the difference between the current position and the reference point."""
        return self.thr_alpha * np.abs(self.x_ref - self.p)

    def _threshold_random(self):
        """Returns a random threshold value based on a normal distribution."""
        noise = np.random.normal(0, self.noise_std, size=self.shape)
        return self._base_threshold + self.thr_k * noise

    def _init_threshold_fn(self):
        """Initialises the threshold function based on the specified threshold mode."""
        if self.thr_mode == "fixed":
            return self._threshold_fixed
        elif self.thr_mode == "adaptive_time":
            return self._threshold_adaptive_time
        elif self.thr_mode == "adaptive_stag":
            return self._threshold_adaptive_stag
        elif self.thr_mode == "diff_pg":
            return self._threshold_diff_pg
        elif self.thr_mode == "diff_pref":
            return self._threshold_diff_pref
        elif self.thr_mode == "random":
            return self._threshold_random
        else:
            raise ValueError(f"Unknown threshold mode: {self.thr_mode}")

    def _update_threshold(self):
        """Updates the threshold based on the current time step and the base threshold."""
        self.threshold = self._threshold_fn()
        self.threshold = np.clip(self.threshold, self.thr_min, self.thr_max)

    # SPIKING CONDITIONS
    def _spike_fixed(self, v1: float, v2: float, thr: float, dim: int):
        """Fixed spiking condition based on a threshold."""
        return np.abs(v1) > thr

    def _spike_l1(self, v1: float, v2: float, thr: float, dim: int):
        """L1 norm spiking condition based on a threshold."""
        return np.abs(v1) + np.abs(v2) > thr

    def _spike_l2(self, v1: float, v2: float, thr: float, dim: int):
        """L2 norm spiking condition based on a threshold."""
        return np.linalg.norm([v1, v2]) > thr

    def _spike_l2_gen(self, v1: float, v2: float, thr: float, dim: int):
        """Generalised L2 norm spiking condition based on a threshold."""
        return (v1 ** 2 + self.spk_alpha * v2 ** 2) > thr ** 2

    def _spike_random(self, v1: float, v2: float, thr: float, dim: int):
        """Random spiking condition based on a threshold."""
        magnitude = np.linalg.norm([v1, v2])
        spike_prob = 1.0 / (1.0 + np.exp(-(magnitude - thr)))
        return np.random.rand() < spike_prob

    def _spike_adaptive(self, v1: float, v2: float, thr: float, dim: int):
        """Adaptive spiking condition based on the spiking alpha and time step."""
        min_threshold = 1e-6
        adaptive_threshold = max(min_threshold, self.spk_alpha / (1 + self.time_step / self.max_steps))
        return np.abs(v1) > adaptive_threshold

    def _spike_stable(self,v1: float, v2: float, thr: float, dim: int):
        """Stable spiking condition based on the stability of the system."""
        eps = 1e-3 / (1 + self.time_step)
        return np.linalg.norm([v1, v2]) < eps

    def _init_spike_condition(self) -> Callable:
        """Initialises the spiking condition based on the specified spiking condition type."""
        if self.spk_cond == "fixed":
            return self._spike_fixed
        elif self.spk_cond == "l1":
            return self._spike_l1
        elif self.spk_cond == "l2":
            return self._spike_l2
        elif self.spk_cond == "l2_gen":
            return self._spike_l2_gen
        elif self.spk_cond == "random":
            return self._spike_random
        elif self.spk_cond == "adaptive":
            return self._spike_adaptive
        elif self.spk_cond == "stable":
            return self._spike_stable
        else:
            raise ValueError(f"Unknown spiking condition: {self.spk_cond}")

    def run_spk(self):
        """Runs the spiking core process model.

        The process is summarised as follows:
            1. If the process is not initialised, it generates random values for `p` and `g`, transforms them, and sets the initial state.
            2. If the process is initialised, it reads the inputs from the inports, transforms the variables, updates the threshold, and runs the core process. This core process involves:
                - Applying the heuristic search operator if the neuron fires or if it is compulsory to fire.
                - Applying the dynamic system model to update the state of the neuron.
                - Checking the bounds of the variables. (Only if `is_bounded` is True)
            3. Finally, it sends the updated state to the outports.
        """
        self.prev_p      = self.p.copy()

        # Read the inputs
        if not self.initialised:  # first iteration
            self.compulsory_fire = np.zeros(self.x.shape).astype(bool)
            self.p = np.random.uniform(-1, 1, self.shape).astype(float)
            self.g = np.random.uniform(-1, 1, self.shape).astype(float)

            self._transform_variables()
            self._update_threshold()
            self.initialised = True

        else:
            s_in        = self.s_in.recv()
            self.g      = self.g_in.recv()
            fg_in       = self.fg_in.recv()
            self.p      = self.p_in.recv()
            fp_in       = self.fp_in.recv()
            self.xn     = self.xn_in.recv()
            fxn_in      = self.fxn_in.recv()

            self.compulsory_fire = s_in.astype(bool)
            self._transform_variables()
            self._update_threshold()
            self._run_core_process()

        self._send_to_ports()


@implements(proc=Selector, protocol=LoihiProtocol)
@requires(CPU)
class PySelectorModel(PyLoihiProcessModel):
    """Selector model for low-level selection in perturbation-based nheuristics

    This model implements a selector process that evaluates the fitness of positions and selects the best one based on a given function. This is intended to be used in conjunction with a perturbation-based nheuristic process inside a NeuroHeuristicUnit.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.Selector`:
        Process that selects the best position based on a fitness function.
    """

    # Inputs
    x_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)

    # Variables
    p:      np.ndarray      = LavaPyType(np.ndarray, _POS_FLOAT_)
    fp:     np.ndarray      = LavaPyType(np.ndarray, _FIT_FLOAT_)

    # Outputs
    p_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out: PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        """Initialises the selector model with the given parameters.

        Arguments
        ---------
            proc_params : dict
                A dictionary containing the parameters for the process model. It should include:
                    - ``agent_id``: int, identifier of the agent
                    - ``num_agents``: int, number of agents in the system
                    - ``function``: callable, function to evaluate the fitness of a position
        """
        super().__init__(proc_params)
        self.agent_id   = proc_params['agent_id']
        self.num_agents = proc_params['num_agents']

        self.funct      = proc_params['function']
        self.sel_mode   = proc_params.get('sel_mode', 'greedy')

        match self.sel_mode:
            case 'random':
                self._selection = self._sel_random
            case 'metropolis':
                self._selection = self._sel_metropolis
            case _: # Default to greedy selection
                self._selection = self._sel_greedy

        self.initialised = False

    def _sel_greedy(self, fx, **kwargs):
        """Greedy selection strategy: always selects the best position."""
        return fx < self.fp[0]

    def _sel_random(self, fx, **kwargs):
        return np.random.rand() < 0.5

    def _sel_metropolis(self, fx, **kwargs):
        """Metropolis selection strategy: accepts a new position with a probability based on the fitness difference."""
        return np.random.rand() < np.exp(- (fx - self.fp[0]))

    def run_spk(self):
        """Runs the selector process model.

        The process is summarised as follows:
            1. Receives the input position `x` from the inport.
            2. Evaluates the fitness of the position using the provided function.
            3. If the fitness is better than the current best fitness or if the process is not initialised, updates the best position and fitness.
            4. Sends the updated-best position and fitness to the outports.
        """
        # Read the input position
        x       = self.x_in.recv()

        # Evaluate the function
        fx      = self.funct(x.flatten())

        # Update the particular position
        if not self.initialised or self._selection(fx, x=x):
            self.initialised    = True
            self.p[:]           = x
            self.fp[:]          = fx

        # Send the updated position
        self.p_out.send(self.p)
        self.fp_out.send(self.fp)


@implements(proc=HighLevelSelection, protocol=LoihiProtocol)
@requires(CPU)
class PyHighLevelSelectionModel(PyLoihiProcessModel):
    """High-level selection model for perturbation-based nheuristics

    This model implements a high-level selection process that aggregates the best positions from multiple NeuroHeuristicUnit and selects the overall best position based on a fitness function.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.HighLevelSelection`:
        Process that performs high-level selection of the best position from multiple NeuroHeuristicUnit processes.
    """

    # Inputs
    p_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    # Variables
    p:      np.ndarray      = LavaPyType(np.ndarray, float)
    fp:     np.ndarray      = LavaPyType(np.ndarray, float)
    g:      np.ndarray      = LavaPyType(np.ndarray, float)
    fg:     np.ndarray      = LavaPyType(np.ndarray, float)

    # Outputs
    g_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fg_out: PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        """Initialises the high-level selection model with the given parameters.

        Arguments
        ---------
            proc_params : dict
                A dictionary containing the parameters for the process model. It must include:
                    - ``num_agents``: int, number of agents in the system
        """
        super().__init__(proc_params)
        self.num_agents     = proc_params['num_agents']
        self.initialised    = False


    def run_spk(self):
        """Runs the high-level selection process model.

        The process is summarised as follows:
            1. Receives the candidate positions and their fitness values from the inports.
            2. Saves these candidates in the internal variables `p` and `fp`.
            3. Searches for the best candidate based on the fitness values.
            4. If the new candidate is better than the current global best or if the process is not initialised, updates the global best position and fitness.
            5. Sends the global best position and fitness to the outports.
        """

        # Read the inputs
        g_candidates        = self.p_in.recv()
        fg_candidates       = self.fp_in.recv()

        # Save these candidates
        self.p[:]           = g_candidates[:]
        self.fp[:]          = fg_candidates[:]

        # Look for the best candidate
        best_candidate      = np.argmin(fg_candidates)

        # Read the best candidate
        new_g               = g_candidates[best_candidate, :]
        new_fg              = fg_candidates[best_candidate]

        # Compare and update the global best
        if not self.initialised or new_fg < self.fg[0]:
            self.initialised    = True
            self.g[:]           = new_g
            self.fg[:]          = new_fg

        # Send the global best
        self.g_out.send(self.g)
        self.fg_out.send(self.fg)


@implements(proc=NeuroHeuristicUnit, protocol=LoihiProtocol)
class SubNeuroHeuristicUnitModel(AbstractSubProcessModel):
    """Sub-process model for NeuroHeuristicUnit

    This model implements the sub-process structure of a NeuroHeuristicUnit, which includes a spiking core, a selector for position evaluation, and handlers for spiking signals and positions. It connects these components to form a complete perturbation-based nheuristic process.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.NeuroHeuristicUnit`:
        Process that implements a NeuroHeuristicUnit for perturbation-based nheuristics.
    :py:class:`neuroptimiser.core.processes.TwoDimSpikingCore`:
        Process that implements a two-dimensional spiking core for perturbation-based nheuristics.
    :py:class:`neuroptimiser.core.processes.Selector`:
        Process that selects the best position based on a fitness function.
    :py:class:`neuroptimiser.core.processes.SpikingHandler`:
        Process that handles spiking signals in the NeuroHeuristicUnit.
    :py:class:`neuroptimiser.core.processes.PositionSender`:
        Process that sends the position of the agent in the NeuroHeuristicUnit.
    :py:class:`neuroptimiser.core.processes.PositionReceiver`:
        Process that receives the positions of neighbouring agents in the NeuroHeuristicUnit.
    """

    def __init__(self, proc: NeuroHeuristicUnit): # noqa
        """Builds sub Process structure of the Process.

        Arguments
        ---------
            proc : NeuroHeuristicUnit
                The process to build the sub-process structure for.
            proc.proc_params : dict
                A dictionary containing the parameters for the process model. It must include:
                    - ``agent_id``: int, identifier of the agent
                    - ``num_dimensions``: int, number of dimensions in the problem space
                    - ``num_agents``: int, number of agents in the system
                    - ``num_neighbours``: int, number of neighbouring agents (default: 0)
                    - ``spiking_core``: str, type of spiking core to use (default: "TwoDimSpikingCore")
                    - ``function``: callable, function to evaluate the fitness of a position
                    - ``core_params``: dict, parameters for the spiking core
                    - ``selector_params``: dict, parameters for the selector
        """

        # PARAMETERS
        # Get the parameters of the Processes
        agent_id        = proc.proc_params.get("agent_id", 0)
        num_dimensions  = proc.proc_params.get("num_dimensions", 2)
        num_agents      = proc.proc_params.get("num_agents", 1)
        num_neighbours  = proc.proc_params.get("num_neighbours", 0)

        spk_core_str    = proc.proc_params.get("spiking_core",                                                      TwoDimSpikingCore)
        if spk_core_str == "TwoDimSpikingCore":
            SpikingCore = TwoDimSpikingCore
        else:
            raise NotImplementedError("This method is not implemented (yet?)")

        internal_shape              = (num_dimensions,)
        external_shape              = (num_agents, num_dimensions)

        internal_shape_neighbours   = (num_neighbours, num_dimensions)
        external_shape_neighbours   = (num_neighbours, num_dimensions, num_agents)

        function        = proc.proc_params.get("function", lambda x: np.linalg.norm(x))

        core_params     = proc.proc_params.get("core_params", {})
        selector_params = proc.proc_params.get("selector_params", {})

        # BUILDING BLOCKS
        # Spiking Core or PerturbationNHeuristic
        self.perturbator = SpikingCore(
            num_dimensions=num_dimensions,
            num_neighbours=num_neighbours,
            **core_params
        )
        proc.core_ref = self.perturbator

        # Selector
        self.selector = Selector(
            agent_id=agent_id,
            num_agents=num_agents,
            function=function,
            num_dimensions=num_dimensions,
            **selector_params
        )
        proc.selector_ref = self.selector

        # Spiking Signal Handler
        self.spiking_handler = SpikingHandler(
            agent_id=agent_id,
            internal_shape=internal_shape,
            external_shape=external_shape
        )

        # Position Sender
        self.position_sender = PositionSender(
            agent_id=agent_id,
            internal_shape=internal_shape,
            external_shape=external_shape
        )

        if num_neighbours > 0:
            # Position Receiver
            self.position_receiver = PositionReceiver(
                agent_id=agent_id,
                internal_shape=internal_shape_neighbours,
                external_shape=external_shape_neighbours
            )

        # INTERNAL CONNECTIONS
        self.perturbator.out_ports.x_out.connect(
            self.selector.in_ports.x_in)

        self.perturbator.out_ports.s_out.connect(
            self.spiking_handler.in_ports.s_in)
        self.spiking_handler.out_ports.a_out.connect(
            self.perturbator.in_ports.s_in)

        self.selector.out_ports.p_out.connect(
            self.perturbator.in_ports.p_in)
        self.selector.out_ports.fp_out.connect(
            self.perturbator.in_ports.fp_in)

        self.selector.out_ports.p_out.connect(
            self.position_sender.in_ports.p_in)
        self.selector.out_ports.fp_out.connect(
            self.position_sender.in_ports.fp_in)

        if num_neighbours > 0:
            self.position_receiver.out_ports.p_out.connect(
                self.perturbator.in_ports.xn_in)
            self.position_receiver.out_ports.fp_out.connect(
                self.perturbator.in_ports.fxn_in)

        # EXTERNAL CONNECTIONS
        # Connect to parent ports to the spiking handler
        self.spiking_handler.out_ports.s_out.connect(
            proc.out_ports.s_out)
        proc.in_ports.a_in.connect(
            self.spiking_handler.in_ports.a_in)

        # parent to perturbator (global information)
        proc.in_ports.g_in.connect(
            self.perturbator.in_ports.g_in)
        proc.in_ports.fg_in.connect(
            self.perturbator.in_ports.fg_in)

        # position sender to parent
        self.position_sender.out_ports.p_out.connect(
            proc.out_ports.p_out)
        self.position_sender.out_ports.fp_out.connect(
            proc.out_ports.fp_out)

        # parent to position receiver
        if num_neighbours > 0:
            proc.in_ports.pn_in.connect(
                self.position_receiver.in_ports.p_in)
            proc.in_ports.fpn_in.connect(
                self.position_receiver.in_ports.fp_in)

        # Finally, connect the variables broadcast from perturbator
        proc.vars.x.alias(self.perturbator.vars.x)
        proc.vars.v1.alias(self.perturbator.vars.v1)
        proc.vars.v2.alias(self.perturbator.vars.v2)

@implements(proc=TensorContractionLayer, protocol=LoihiProtocol)
@requires(CPU)
class PyTensorContractionLayerModel(PyLoihiProcessModel):
    """Tensor contraction layer model for Loihi-based perturbation-based nheuristics

    This model implements a tensor contraction layer that performs a contraction operation on the input spikes and weights, producing an output activation matrix.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.TensorContractionLayer`:
        Process that implements a tensor contraction layer for perturbation-based nheuristics.
    """
    # Inports
    s_in:           PyInPort    = LavaPyType(PyInPort.VEC_DENSE, bool)

    # Variables
    weight_matrix:  np.ndarray  = LavaPyType(np.ndarray, float)
    s_matrix:       np.ndarray  = LavaPyType(np.ndarray, bool)

    # Outports
    a_out:          PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)

    def __init__(self, proc_params):
        """Initialises the tensor contraction layer model with the given parameters.

        Arguments
        ---------
            proc_params : dict
                A dictionary containing additional parameters for the process model.
        """
        super().__init__(proc_params)
        self.weight_tensor = None  # None for the first iteration

    def run_spk(self):
        """Runs the tensor contraction layer process model.

        The process is summarised as follows:
            1. Receives the input spikes from the inport.
            2. If the weight tensor is not initialised, it sets it to the weight matrix with an additional dimension.
            3. Performs the contraction operation using `np.einsum` to compute the output activation matrix.
            4. Sends the output activation matrix to the outport.
        """

        # Check if the weight tensor is initialised
        if self.weight_tensor is None:
            self.weight_tensor = self.weight_matrix[..., np.newaxis]

        # Read the input spikes
        self.s_matrix[:] = self.s_in.recv()
        a_matrix = np.einsum('ikj,kj->ij', self.weight_tensor, self.s_matrix)

        # Send the output activation matrix
        self.a_out.send(a_matrix.astype(bool))


@implements(proc=NeighbourhoodManager, protocol=LoihiProtocol)
@requires(CPU)
class PyNeighbourhoodManagerModel(PyLoihiProcessModel):
    """Neighbourhood manager model for Loihi-based perturbation-based nheuristics

    This model manages the neighbourhood of agents/units in a neuroptimiser architecture, allowing them to access the positions and fitness values of their neighbours.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.NeighbourhoodManager`:
        Process that manages the neighbourhood of agents/units in a neuroptimiser architecture.
    """

    # Inports
    p_in:               PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:              PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    # Variables
    weight_matrix:      np.ndarray  = LavaPyType(np.ndarray, float)
    neighbour_indices:  np.ndarray  = LavaPyType(np.ndarray, int)

    # Outports
    p_out:              PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out:             PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)


    def __init__(self, proc_params):
        """Initialises the neighbourhood manager model with the given parameters.

        Arguments
            proc_params : dict
                A dictionary containing the parameters for the process model. It must include:
                    - ``num_agents``: int, number of agents in the system
                    - ``num_dimensions``: int, number of dimensions in the problem space
                    - ``num_neighbours``: int, number of neighbouring agents

        """
        super().__init__(proc_params)

        # Get the parameters of the Process
        self.neighbourhood_p_tensor = None
        self.neighbourhood_fp_tensor = None

        # Define some internal variables
        self.num_agents     = proc_params.get("num_agents", -1)
        self.num_dimensions = proc_params.get("num_dimensions", -1)
        self.num_neighbours = proc_params.get("num_neighbours", -1)

        self.initialised = False

    def run_spk(self):
        """Runs the neighbourhood manager process model.

        The process is summarised as follows:
            1. If the process is not initialised, it creates tensors to store the neighbourhood positions and fitness values.
            2. Receives the input position and fitness matrices from the inports.
            3. For each agent/unit, retrieves the positions and fitness values of its neighbours based on the pre-defined indices.
            4. Sends the neighbourhood position and fitness tensors to the outports.
        """
        if not self.initialised:
            self.neighbourhood_p_tensor = np.full(
                shape=self.p_out.shape, fill_value=-1.0)
            self.neighbourhood_fp_tensor = np.full(
                shape=self.fp_out.shape, fill_value=-1.0)
            self.initialised = True

        # Read the input position and fitness matrices
        p_matrix    = self.p_in.recv()
        fp_matrix   = self.fp_in.recv()

        # Populate the neighbourhood tensors
        for i in range(self.num_agents):
            neighbour_indices                       = self.neighbour_indices[i]
            self.neighbourhood_p_tensor[:, :, i]    = p_matrix[neighbour_indices]
            self.neighbourhood_fp_tensor[:, i]      = fp_matrix[neighbour_indices]

        # Send the neighbourhood position and fitness tensors to the outports
        self.p_out.send(self.neighbourhood_p_tensor)
        self.fp_out.send(self.neighbourhood_fp_tensor)


@implements(proc=SpikingHandler, protocol=LoihiProtocol)
@requires(CPU)
class PySpikingHandlerModel(PyLoihiProcessModel):
    """Spiking handler model for Loihi-based perturbation-based nheuristics

    This model handles the spiking signals in a nheuristic process, allowing the spiking core read and write spikes to the input and output ports to the external world.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.SpikingHandler`:
        Process that handles spiking signals in a NeuroHeuristicUnit.
    """

    # Inports
    s_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, bool)

    # Outports
    a_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, bool)
    s_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, bool)

    def __init__(self, proc_params):
        """Initialises the spiking handler model with the given parameters.

        Arguments
            proc_params : dict
                Dictionary containing the parameters for the process model. It must include:
                    - ``agent_id``: int, identifier of the agent
                    - ``internal_shape``: tuple, shape of the internal state (e.g., number of dimensions)
                    - ``external_shape``: tuple, shape of the external state (e.g., number of agents and dimensions)
        """
        super().__init__(proc_params)

        # Define the template matrix
        self.s_matrix = np.zeros(
            shape=proc_params["external_shape"]).astype(bool)
        self.a_vector = np.zeros(
            shape=proc_params["internal_shape"]).astype(bool)

        # Get the agent ID from the process parameters
        self.agent_id = proc_params["agent_id"]
        self.initialised = False

    def run_spk(self):
        """Runs the spiking handler process model.

        The process is summarised as follows:
            1. Receives the input spikes and activation matrix from the inports.
            2. Prepares the output spikes based on the input spikes and sends them to the outports.
        """
        # Read the input spikes
        if self.initialised:
            s_vector            = self.s_in.recv()
            a_matrix            = self.a_in.recv()
        else:
            s_vector            = np.zeros(self.s_in.shape).astype(bool)
            a_matrix            = np.zeros(self.a_in.shape).astype(bool)
            self.initialised    = True

        # Prepare s_out using the input spikes
        self.s_matrix[self.agent_id, :] = s_vector[:]

        # Extract the incoming spikes
        self.a_vector[:]                = a_matrix[self.agent_id, :]

        # Send the output spikes
        self.s_out.send(self.s_matrix)
        self.a_out.send(self.a_vector)


@implements(proc=PositionSender, protocol=LoihiProtocol)
@requires(CPU)
class PyPositionSenderModel(PyLoihiProcessModel):
    """Position sender model for Loihi-based perturbation-based nheuristics

    This model sends the position and fitness values of an agent/unit to the external world, allowing other processes to access this information.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.PositionSender`:
        Process that sends the position and fitness values of an agent/unit in a NeuroHeuristicUnit.
    """

    # Inports
    p_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    # Outports
    p_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out: PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        """Initialises the position sender model with the given parameters.

        Arguments
            proc_params : dict
                A dictionary containing the parameters for the process model. It must include:
                    - ``agent_id``: int, identifier of the agent
                    - ``external_shape``: tuple, shape of the external state (e.g., number of agents and dimensions)
        """
        super().__init__(proc_params)

        # Define the template matrix
        self.p_matrix = np.zeros(
            shape=proc_params["external_shape"]).astype(float)
        self.fp_vector = np.zeros(
            shape=(proc_params["external_shape"][0],)).astype(float)

        self.agent_id = proc_params["agent_id"]

    def run_spk(self):
        """Runs the position sender process model.

        The process is summarised as follows:
            1. Receives the input position vector and fitness value from the inports.
            2. Prepares the output position matrix and fitness vector using the input spikes.
            3. Sends the output position matrix and fitness vector to the outports.
        """

        # First read the inputs
        p_vector                        = self.p_in.recv()
        fp_value                        = self.fp_in.recv()

        # Prepare x_out using the input spikes
        self.p_matrix[self.agent_id, :] = p_vector[:]
        self.fp_vector[self.agent_id]   = fp_value[:]

        # Send the output spikes
        self.p_out.send(self.p_matrix)
        self.fp_out.send(self.fp_vector)


@implements(proc=PositionReceiver, protocol=LoihiProtocol)
@requires(CPU)
class PyPositionReceiverModel(PyLoihiProcessModel):
    """Position receiver model for Loihi-based perturbation-based nheuristics

    This model receives the positions and fitness values of neighbouring agents/units in a neuroptimiser architecture, allowing the spiking core to access this information.

    See Also
    --------
    :py:class:`neuroptimiser.core.processes.PositionReceiver`:
        Process that receives the positions and fitness values of neighbouring agents/units in a NeuroHeuristicUnit.
    """

    # Inports
    p_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    # Outports
    p_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out: PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        """Initialises the position receiver model with the given parameters.

        Arguments
            proc_params : dict
                A dictionary containing the parameters for the process model. It must include:
                    - ``agent_id``: int, identifier of the agent
                    - ``external_shape``: tuple, shape of the external state (e.g., number of neighbours, dimensions, and agents)
        """
        super().__init__(proc_params)

        # Define the template matrix
        self.p_matrix = np.zeros(
            shape=proc_params["external_shape"]).astype(float)
        self.fp_vector = np.zeros(
            shape=(proc_params["external_shape"][0],)).astype(float)

        self.agent_id = proc_params["agent_id"]

    def run_spk(self):
        """Runs the position receiver process model.

        The process is summarised as follows:
            1. Receives the input position tensor and fitness vector from the inports.
            2. Prepares the output position matrix and fitness vector using the input spikes.
            3. Sends the output position matrix and fitness vector to the outports.
        """

        # First read the inputs
        p_tensor        = self.p_in.recv()
        fp_matrix       = self.fp_in.recv()

        # Prepare x_out using the input spikes
        self.p_matrix   = p_tensor[:, :, self.agent_id]
        self.fp_vector  = fp_matrix[:, self.agent_id]

        # Send the output spikes
        self.p_out.send(self.p_matrix)
        self.fp_out.send(self.fp_vector)
