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
_POS_FLOAT_ = np.float32
_FIT_FLOAT_ = np.float32
SPK_CORE_OPTIONS = ["TwoDimSpikingCore"]


#%%
class AbstractPerturbationNHeuristicModel(PyLoihiProcessModel):
    """
    Abstract model for a perturbation-based neuro-heuristic process.

    Parameters
    ----------
    proc_params : dict
        Dictionary containing the parameters for the process.

    Attributes
    ----------

    alpha : float
    """

    s_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, bool)

    p_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)
    g_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fg_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)
    xn_in:  PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fxn_in: PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    x:      np.ndarray      = LavaPyType(np.ndarray, float)

    s_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, bool)
    x_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)

    def __init__(self, proc_params):
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



@implements(proc=TwoDimSpikingCore, protocol=LoihiProtocol)
@requires(CPU)
class PyTwoDimSpikingCoreModel(AbstractPerturbationNHeuristicModel):
    v1       : np.ndarray = LavaPyType(np.ndarray, float)
    v2       : np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
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

        if self.hs_operator == "fixed":
            self._apply_hs = self._hs_fixed
        elif self.hs_operator == "random":
            self._apply_hs = self._hs_random
        elif self.hs_operator == "directional":
            self._apply_hs = self._hs_directional
        elif self.hs_operator == "differential":
            self._apply_hs = self._hs_differential
            if self.num_neighbours > 3:
                for dim in range(self.num_dimensions):
                    triplet = np.random.choice(range(self.num_neighbours), 3, replace=False)
                    self.neighbour_indices_cache.append(triplet)

        else:
            raise ValueError(f"Unknown hs_operator: {self.hs_operator}")

        # Initialise the approximation method
        if self.approx == "euler":
            self.approx_method = self._euler_approximation
        elif self.approx == "rk4":
            self.approx_method = self._rk4_approximation
        else:
            raise ValueError(f"Unknown approximation method: {self.approx}")

        self._spike_condition   = self._init_spike_condition()
        self._threshold_fn      = self._init_threshold_fn()
        self._base_threshold    = self.thr_alpha * np.ones_like(self.v1)

        self.v1_best            = np.zeros(self.shape).astype(float)
        self.v2_best            = np.zeros(self.shape).astype(float)

        self.v_bounds  = np.zeros((2,self.shape[0])).astype(float)
        self.x_ref      = np.zeros(self.shape).astype(float)
        self.xn         = np.zeros(
            shape=(self.num_neighbours, self.num_dimensions)
        ).astype(float)
        self.vn         = np.zeros(
            shape=(self.num_neighbours, self.num_dimensions)
        ).astype(float)
        self.ref_point  = np.zeros(self.shape).astype(float)
        self.prev_p     = np.zeros(self.shape).astype(float)
        self.stag_count = 0

    # TRANSFORMATION METHODS
    def _linear_transform(self, x):
        var = self.alpha * (x - self.x_ref)
        return var

    def _linear_transform_inv(self, var):
        return var / self.alpha + self.x_ref

    def _check_vbounds(self):
        # Update the bounds
        self.v_bounds[0,:] = self._linear_transform(-1.0*np.ones(self.shape))
        self.v_bounds[1,:] = self._linear_transform(np.ones(self.shape))

        # Check the bounds
        self.v1 = np.clip(self.v1, self.v_bounds[0,:], self.v_bounds[1,:])
        self.v2 = np.clip(self.v2, self.v_bounds[0,:], self.v_bounds[1,:])

    def _send_to_ports(self):
        # Inverse transformation to send
        self.x  = self._linear_transform_inv(self.v1)

        # Send the new position
        self.s_out.send(self.self_fire)
        self.x_out.send(self.x)

    # APPROXIMATION METHODS
    def _euler_approximation(self, model, var, **kwargs):
        """
        Euler approximation of the system of equations
        """
        return var + model(var) * self.dt

    def _rk4_approximation(self, model, var, **kwargs):
        """
        Runge-Kutta 4th order approximation of the system of equations
        """
        k1 = model(var)
        k2 = model(var + k1/2 * self.dt)
        k3 = model(var + k2/2 * self.dt)
        k4 = model(var + k3 * self.dt)

        return var + (k1 + 2*k2 + 2*k3 + k4) / 6 * self.dt

    # 2D MODELS
    def _izhikevich_system(self, upsilon, dim, **kwargs):
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
        system_matrix = self.models_coeffs[dim] \
            if len(self.stable) > 1 else self.models_coeffs[0]
        deriv_upsilon = system_matrix @ upsilon
        return deriv_upsilon

    def _hs_fixed(self, dim, var):
        new_var_1 = self.v1_best[dim]
        new_var_2 = self.v2_best[dim]

        new_var = np.array([new_var_1, new_var_2]) + np.random.normal(0, self.noise_std, 2)
        return new_var

    def _hs_random(self, dim, var):
        new_var_1 = np.random.normal(0.0, 1)
        new_var_2 = np.random.normal(0.0, 1)

        new_var  = np.array([new_var_1, new_var_2])
        return new_var

    def _hs_directional(self, dim, var):
        dir1 = self.v1_best[dim] - var[0]
        dir2 = self.v2_best[dim] - var[1]
        scale = self.alpha * np.random.randn() * self.noise_std
        new_var  = np.array(var) + scale * np.array([dir1, dir2])
        return new_var

    @staticmethod
    def _get_F():
        return np.random.normal(0.5, 0.1)

    def _hs_differential(self, dim, var):
        if self.hs_variant in ["rand", "current-to-rand", "best-to-rand"]:
            # Get from neighbours
            if self.num_neighbours > 3:
                ind1, ind2, ind3  = self.neighbour_indices_cache[dim]
                pre_sum = self.vn[ind1, dim]
                the_sum = self.vn[ind2, dim] - self.vn[ind3, dim]
            else:
                pre_sum = np.random.normal(0, self.noise_std)
                the_sum = np.random.normal(0, self.noise_std)

            if self.hs_variant == "current-to-rand":
                new_var1 = var[0] + self._get_F() * (pre_sum - var[0]) + self._get_F() * the_sum
            elif self.hs_variant == "best-to-rand":
                new_var1 = var[0] + self._get_F() * (self.v1_best[dim] - var[0]) + self._get_F() * the_sum
            else: # rand
                new_var1 = pre_sum + self._get_F() * the_sum

            new_var2 = var[1] + self._get_F() * np.random.normal(0, self.noise_std)
            new_var = np.array([new_var1, new_var2])
            return new_var

        elif self.hs_variant == "current-to-best":
            # F parameter is calculated using a distribution
            diff1 = self.v1_best[dim] - var[0]
            diff2 = self.v2_best[dim] - var[1]
            new_var = np.array(var) + self._get_F() * np.array([diff1, diff2])
            return new_var

        else:
            raise ValueError(f"Unknown hs_variant: {self.hs_variant}")

    # DYNAMIC HEURISTIC
    def _apply_hd(self, dim, var):
        """Applies the selected approximation to the dynamic system for a given dimension."""
        # Prepare the model
        # model = partial(self.model, dim=dim)

        # Apply the model using an approximation, if so
        new_var = self.approx_method(self.models[dim], var, dt=self.dt)

        # Post-processing
        # new_var += np.exp(-self.time_step / self.max_steps) * np.random.normal(0, self.noise_std, 2)
        return new_var

    def _run_core_process(self):
        for dim in range(self.num_dimensions):
            # Get the spike condition for this neuron
            self.self_fire[dim]     = self._spike_condition(self.v1[dim], self.v2[dim], self.threshold[dim], dim)
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
        self.v1_best    = self._linear_transform(self.p)

        if self.xn is not None:
            self.vn[:] = self._linear_transform(self.xn)
        self.ref_point  = self._linear_transform(np.average(x_refs, axis=0))

    # Threshold determination
    def _threshold_fixed(self, base_threshold):
        return base_threshold

    def _threshold_adaptive_time(self, base_threshold):
        scale = 1.0 / (1.0 + self.thr_k * (self.time_step + 1.0))
        return base_threshold * scale

    def _threshold_adaptive_stag(self, base_threshold):
        scale = 1.0 + self.thr_k * self.stag_count
        return base_threshold * scale

    def _threshold_diff_pg(self, base_threshold=None):
        return self.thr_alpha * np.abs(self.p - self.g)

    def _threshold_diff_pref(self, base_threshold=None):
        return self.thr_alpha * np.abs(self.x_ref - self.p)

    def _threshold_random(self, base_threshold):
        noise = np.random.normal(0, self.noise_std, size=self.shape)
        return base_threshold + self.thr_k * noise

    def _init_threshold_fn(self):
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
        self.threshold = self._threshold_fn(self._base_threshold)
        self.threshold = np.clip(self.threshold, self.thr_min, self.thr_max)

    # SPIKING CONDITIONS
    def _spike_fixed(self, v1, v2, thr, dim):
        return np.abs(v1) > thr

    def _spike_l1(self, v1, v2, thr, dim):
        return np.abs(v1) + np.abs(v2) > thr

    def _spike_l2(self, v1, v2, thr, dim):
        return np.linalg.norm([v1, v2]) > thr

    def _spike_l2_gen(self, v1, v2, thr, dim):
        return (v1 ** 2 + self.spk_alpha * v2 ** 2) > thr ** 2

    def _spike_random(self, v1, v2, thr, dim):
        magnitude = np.linalg.norm([v1, v2])
        spike_prob = 1.0 / (1.0 + np.exp(-(magnitude - thr)))
        return np.random.rand() < spike_prob

    def _spike_adaptive(self, v1, v2, thr, dim):
        min_threshold = 1e-6
        adaptive_threshold = max(min_threshold, self.spk_alpha / (1 + self.time_step / self.max_steps))
        return np.abs(v1) > adaptive_threshold

    def _spike_stable(self, v1, v2, thr, dim):
        eps = 1e-3 / (1 + self.time_step)
        return np.linalg.norm([v1, v2]) < eps

    def _init_spike_condition(self):
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
        self.prev_p      = self.p.copy()

        # Read the inputs
        if not self.initialised:  # first iteration
            self.compulsory_fire = np.zeros(self.x.shape).astype(bool)
            self.p = np.random.uniform(-1, 1, self.shape).astype(float)
            self.g = np.random.uniform(-1, 1, self.shape).astype(float)

            self._transform_variables()
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
    # Inputs
    x_in:   PyInPort        = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)

    # Variables
    p:      np.ndarray      = LavaPyType(np.ndarray, _POS_FLOAT_)
    fp:     np.ndarray      = LavaPyType(np.ndarray, _FIT_FLOAT_)

    # Outputs
    p_out:  PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out: PyOutPort       = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.agent_id   = proc_params['agent_id']
        self.num_agents = proc_params['num_agents']

        self.funct      = proc_params['function']

        self.initialised = False

    def run_spk(self):
        x = self.x_in.recv()

        # Evaluate the function
        fx  = self.funct(x.flatten())

        # Update the particular position
        if fx < self.fp[0] or not self.initialised:
            self.initialised = True
            self.p[:] = x
            self.fp[:] = fx

        # Send the updated position
        self.p_out.send(self.p)
        self.fp_out.send(self.fp)


@implements(proc=HighLevelSelection, protocol=LoihiProtocol)
@requires(CPU)
class PyHighLevelSelectionModel(PyLoihiProcessModel):
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
        super().__init__(proc_params)
        self.num_agents = proc_params['num_agents']
        self.accumulated_ack = 0
        self.initialised = False


    def run_spk(self):
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
        if new_fg < self.fg[0] or not self.initialised:
            self.initialised = True
            self.g[:] = new_g
            self.fg[:] = new_fg

        # Send the global best
        self.g_out.send(self.g)
        self.fg_out.send(self.fg)


@implements(proc=NeuroHeuristicUnit, protocol=LoihiProtocol)
class SubNeuroHeuristicUnitModel(AbstractSubProcessModel):
    def __init__(self, proc: NeuroHeuristicUnit): # noqa
        """Builds sub Process structure of the Process."""

        # Get the parameters of the Processes
        agent_id            = proc.proc_params.get("agent_id", 0)
        num_dimensions      = proc.proc_params.get("num_dimensions", 2)
        num_agents          = proc.proc_params.get("num_agents", 1)
        num_neighbours      = proc.proc_params.get("num_neighbours", 0)
        # num_objectives      = proc.proc_params.get("num_objectives", 1)

        spk_core_str         = proc.proc_params.get("spiking_core",
                                                    TwoDimSpikingCore)
        if spk_core_str == "TwoDimSpikingCore":
            SpikingCore = TwoDimSpikingCore
        else:
            raise NotImplementedError("This method is not implemented (yet?)")

        internal_shape = (num_dimensions,)
        external_shape = (num_agents, num_dimensions)

        internal_shape_neighbours = (num_neighbours, num_dimensions)
        external_shape_neighbours = (num_neighbours, num_dimensions, num_agents)

        function            = proc.proc_params.get("function", lambda x: np.linalg.norm(x))

        core_params          = proc.proc_params.get("core_params", {})
        selector_params      = proc.proc_params.get("selector_params", {})

        # PerturbationNHeuristic
        # TODO: Implement a way to inject a custom perturbation
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

        # ================================
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


        # ================================
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
    s_in:           PyInPort    = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out:          PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    weight_matrix:  np.ndarray  = LavaPyType(np.ndarray, float)
    s_matrix:      np.ndarray  = LavaPyType(np.ndarray, bool)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.weight_tensor = None  # None for the first iteration

    def run_spk(self):
        if self.weight_tensor is None:
            self.weight_tensor = self.weight_matrix[..., np.newaxis]

        self.s_matrix[:] = self.s_in.recv()
        a_matrix = np.einsum('ikj,kj->ij', self.weight_tensor, self.s_matrix)

        self.a_out.send(a_matrix.astype(bool))


@implements(proc=NeighbourhoodManager, protocol=LoihiProtocol)
@requires(CPU)
class PyNeighbourhoodManagerModel(PyLoihiProcessModel):
    # ack_in:         PyInPort    = LavaPyType(PyInPort.VEC_DENSE, int)
    p_in:           PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:          PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    p_out:          PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out:         PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)
    # ack_out:        PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, int)

    weight_matrix:  np.ndarray      = LavaPyType(np.ndarray, float)
    neighbour_indices: np.ndarray   = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.neighbourhood_p_tensor = None
        self.neighbourhood_fp_tensor = None

        # Define some internal variables
        self.num_agents     = proc_params.get("num_agents", -1)
        self.num_dimensions = proc_params.get("num_dimensions", -1)
        self.num_neighbours = proc_params.get("num_neighbours", -1)

        # self.num_objectives = proc_params.get("num_objectives", -1)
        self.accumulated_ack = 0
        self.initialised = False

    def run_spk(self):
        if not self.initialised:
            self.neighbourhood_p_tensor = np.full(
                shape=self.p_out.shape, fill_value=-1.0)
            self.neighbourhood_fp_tensor = np.full(
                shape=self.fp_out.shape, fill_value=-1.0)
            self.initialised = True

        p_matrix    = self.p_in.recv()
        fp_matrix   = self.fp_in.recv()  # FUTURE: Add support to multiple objectives

        for i in range(self.num_agents):
            neighbour_indices = self.neighbour_indices[i]
            self.neighbourhood_p_tensor[:, :, i]    = p_matrix[neighbour_indices]
            self.neighbourhood_fp_tensor[:, i]      = fp_matrix[neighbour_indices]

        self.p_out.send(self.neighbourhood_p_tensor)
        self.fp_out.send(self.neighbourhood_fp_tensor)


@implements(proc=SpikingHandler, protocol=LoihiProtocol)
@requires(CPU)
class PySpikingHandlerModel(PyLoihiProcessModel):
    s_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, bool)
    a_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, bool)

    a_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, bool)
    s_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, bool)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.agent_id = proc_params["agent_id"]
        # Define the template matrix
        self.s_matrix = np.zeros(
            shape=proc_params["external_shape"]).astype(bool)
        self.a_vector = np.zeros(
            shape=proc_params["internal_shape"]).astype(bool)
        self.initialised = False

    def run_spk(self):
        if self.initialised:
        # First read the input spikes
            s_vector = self.s_in.recv()
            a_matrix = self.a_in.recv()
        else:
            s_vector = np.zeros(self.s_in.shape).astype(bool)
            a_matrix = np.zeros(self.a_in.shape).astype(bool)
            self.initialised = True

        # Prepare s_out using the input spikes
        self.s_matrix[self.agent_id, :] = s_vector[:]

        # Extract the incoming spikes
        self.a_vector[:] = a_matrix[self.agent_id, :]

        # Send the output spikes
        self.s_out.send(self.s_matrix)
        self.a_out.send(self.a_vector)


@implements(proc=PositionSender, protocol=LoihiProtocol)
@requires(CPU)
class PyPositionSenderModel(PyLoihiProcessModel):
    p_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    p_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out: PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.agent_id = proc_params["agent_id"]
        # Define the template matrix
        self.p_matrix = np.zeros(
            shape=proc_params["external_shape"]).astype(float)
        self.fp_vector = np.zeros(
            shape=(proc_params["external_shape"][0],)).astype(float)

    def run_spk(self):
        # First read the inputs
        p_vector = self.p_in.recv()
        fp_value = self.fp_in.recv()

        # Prepare x_out using the input spikes
        self.p_matrix[self.agent_id, :]     = p_vector[:]
        self.fp_vector[self.agent_id]       = fp_value[:]

        # Send the output spikes
        self.p_out.send(self.p_matrix)
        self.fp_out.send(self.fp_vector)


@implements(proc=PositionReceiver, protocol=LoihiProtocol)
@requires(CPU)
class PyPositionReceiverModel(PyLoihiProcessModel):
    p_in:   PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _POS_FLOAT_)
    fp_in:  PyInPort    = LavaPyType(PyInPort.VEC_DENSE, _FIT_FLOAT_)

    p_out:  PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _POS_FLOAT_)
    fp_out: PyOutPort   = LavaPyType(PyOutPort.VEC_DENSE, _FIT_FLOAT_)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.agent_id = proc_params["agent_id"]

        # Define the template matrix
        self.p_matrix = np.zeros(
            shape=proc_params["external_shape"]).astype(float)
        self.fp_vector = np.zeros(
            shape=(proc_params["external_shape"][0],)).astype(float)

    def run_spk(self):
        # First read the inputs
        p_tensor        = self.p_in.recv()
        fp_matrix       = self.fp_in.recv()

        # Prepare x_out using the input spikes
        self.p_matrix   = p_tensor[:, :, self.agent_id]
        self.fp_vector  = fp_matrix[:, self.agent_id]

        # Send the output spikes
        self.p_out.send(self.p_matrix)
        self.fp_out.send(self.fp_vector)
