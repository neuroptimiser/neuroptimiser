import numpy as np
import time
from schema import SchemaError
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from neuroptimiser.utils import (
    get_2d_sys, get_izhikevich_sys
)

# %% General elements
class AbstractSpikingCore(AbstractProcess):
    def __init__(self,
                 noise_std: float | tuple | list = 0.1,
                 alpha: float = 1.0,
                 max_steps: int = 100,
                 **kwargs):
        super().__init__(**kwargs)

        num_dimensions  = kwargs.get("num_dimensions", 2)
        init_position   = kwargs.get("init_position", None)
        num_neighbours  = kwargs.get("num_neighbours", 0)
        # seed            = kwargs.get("seed", 69)
        # self.proc_params["seed"] = seed

        self.shape = (num_dimensions,)
        shape_fx = (1,)
        shape_xn = (num_neighbours, num_dimensions)
        shape_fxn = (num_neighbours,)

        self.init_position = init_position
        start_position = init_position if init_position is not None else np.random.uniform(-1, 1, self.shape)

        # Inports
        # self.ack_in             = InPort(shape=(1,)) # ack_particular, ack_global, ack_neighbours
        self.s_in = InPort(shape=self.shape)
        self.p_in = InPort(shape=self.shape)
        self.fp_in = InPort(shape=shape_fx)
        self.g_in = InPort(shape=self.shape)
        self.fg_in = InPort(shape=shape_fx)
        self.xn_in = InPort(shape=shape_xn)
        self.fxn_in = InPort(shape=shape_fxn)

        # Variables
        self.x = Var(shape=self.shape, init=start_position)

        # Outports
        self.s_out = OutPort(shape=self.shape)
        self.x_out = OutPort(shape=self.shape)

        # Read and prepare the common parameters
        noise_std   = kwargs.get("noise_std", 0.1)
        if isinstance(noise_std, (tuple, list)):
            noise_std_val = np.random.uniform(noise_std[0], noise_std[1])
        else:
            noise_std_val = noise_std
        alpha       = kwargs.get("alpha", 1.0)
        max_steps   = kwargs.get("max_steps", 100)

        self.proc_params["noise_std"]   = noise_std_val
        self.proc_params["alpha"]       = alpha
        self.proc_params["max_steps"]   = max_steps

    def reset(self):
        start_position = self.init_position if self.init_position is not None else np.random.uniform(-1, 1, self.shape)
        self.x.set(start_position)


class TwoDimSpikingCore(AbstractSpikingCore):
    def __init__(self,
                 **core_params):
        super().__init__(**core_params)

        _name = core_params.get("name", "linear")

        # Read and prepare the specific parameters
        if _name in ["linear", "izhikevich"]:
            # self.proc_params["core_name"]    = _name
            # self.proc_params["core_dt"]      = core_params.get("dt", 0.01)
            # self.proc_params["core_approx"]  = core_params.get("approx", "euler")
            models_coeffs               = core_params.get("coeffs", None)
            #print(f"[agent:core] {_name}.coeffs: {models_coeffs}")

            if _name == "izhikevich":
                _coeffs = self._process_izh_coeffs(models_coeffs)
            else: #if _name == "linear":
                _coeffs = self._process_models_A(models_coeffs)
            self.proc_params["coeffs_values"] = _coeffs

        # Public Variables
        seed = core_params.get("seed", int(time.time() * 1000) % (2 ** 32))
        self.rng = np.random.default_rng(seed)
        self.v1 = Var(
            shape=self.shape,
            init=self.rng.uniform(-1.0, 1.0, size=self.shape))
        self.v2 = Var(
            shape=self.shape,
            init=self.rng.uniform(-1.0, 1.0, size=self.shape))

    # @staticmethod
    def _process_models_A(self, models_A=None):
        # Process the models
        models_A_ = []
        if models_A is None:
            models_A_ = [np.array([[-0.5, -0.5], [0.5, -0.5]])]
        elif isinstance(models_A, str):
            _model_split = models_A.split("_")
            if len(_model_split) > 1:
                _kind = models_A.split("_")[0]
                _many = models_A.split("_")[1] == "all"
            else:
                _kind = models_A
                _many = False

            for _ in range(self.shape[0]):
                models_A_.append(get_2d_sys(_kind))
                if not _many:
                    break
        else:
            if isinstance(models_A, (list, np.ndarray)) and (
                    len(models_A) == 1 or len(models_A) == self.shape[0]):
                models_A_ = models_A
            else:
                raise ValueError("The number of models must be equal to the number of dimensions or 1")
        return models_A_

    def _process_izh_coeffs(self, coeffs=None):
        # Process the coefficients for Izhikevich model
        # coeffs are returned as a list for [a, b, c, d, I]
        coeffs_ = []
        if coeffs is None: # Default values (RS)
            coeffs_ = {"a": 0.02, "b": 0.2, "c": -65, "d": 8, "I": 0.1}
        elif isinstance(coeffs, str):
            _model_split = coeffs.split("_")
            if len(_model_split) > 1:
                _kind = coeffs.split("_")[0] + "r"
                _many = coeffs.split("_")[1] == "all"
            else:
                _kind = coeffs
                _many = False

            for _ in range(self.shape[0]):
                coeffs_.append(get_izhikevich_sys(_kind))
                if not _many:
                    break
        else:
            if isinstance(coeffs, (list, np.ndarray)) and (
                    len(coeffs) == 1 or len(coeffs) == self.shape[0]):
                coeffs_ = coeffs
            else:
                raise ValueError("The number of coefficients must be equal to the number of dimensions or 1")
        return coeffs_

    def reset(self):
        super().reset()
        self.v1.set(self.rng.uniform(-1.0, 1.0, size=self.shape))
        self.v2.set(self.rng.uniform(-1.0, 1.0, size=self.shape))


class Selector(AbstractProcess):
    def __init__(self,
                 agent_id: int = 0,
                 num_agents: int = 1,
                 num_dimensions: int = 2,
                 function=None,
                 **kwargs):
        super().__init__(**kwargs)
        shape = (num_dimensions,)  # assuming [[x1, x2, ..., xn]]
        self.shape = shape

        self.proc_params["agent_id"] = agent_id
        self.proc_params["num_agents"] = num_agents
        self.proc_params["function"] = function

        # Inports
        # self.ack_in = InPort(shape=(1,))
        self.x_in = InPort(shape=shape)

        # Variables
        self.p = Var(shape=shape, init=0.0)
        self.fp = Var(shape=(1,), init=6.9)

        # Outports
        # self.ack_out= OutPort(shape=(1,))
        self.p_out = OutPort(shape=shape)
        self.fp_out = OutPort(shape=(1,))

    def reset(self):
        self.p.set(np.zeros(self.shape))
        self.fp.set(np.array([6.9]))


class HighLevelSelection(AbstractProcess):
    def __init__(self,
                 num_dimensions: int = 2,
                 num_agents: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        shape = (num_agents, num_dimensions)

        # self.ack_in = InPort(shape=(1,))
        self.p_in   = InPort(shape=shape)
        self.fp_in  = InPort(shape=(num_agents,))

        self.p      = Var(shape=shape, init=0.0)
        self.fp     = Var(shape=(num_agents,), init=6.9)
        self.g      = Var(shape=(num_dimensions,), init=0.0)
        self.fg     = Var(shape=(1,), init=6.9)

        self.g_out  = OutPort(shape=(num_dimensions,))
        self.fg_out = OutPort(shape=(1,))
        # self.ack_out= OutPort(shape=(1,))

        self.proc_params["num_agents"] = num_agents
        self.proc_params["num_dimensions"] = num_dimensions

    def reset(self):
        self.p.set(np.zeros(self.p.shape))
        self.fp.set(np.array([6.9] * self.fp.shape[0]))
        self.g.set(np.zeros(self.g.shape))
        self.fg.set(np.array([6.9]))


class NeuroHeuristicUnit(AbstractProcess):
    """
    General model for a Neuro-Heuristic (NeuroHeuristic) unit.

    Description:
    This unit can be used to define a complete neuro-heuristic model using the atomic neuro-heuristic units, such as GenerationNHeuristic, PerturbationNHeuristic, Selector, and EvaluationNHeuristic. The NeuroHeuristic unit can be used to define the complete optimisation process.

    Diagram with ports and variables:
    ```
    --> GenerationNHeuristic --> PerturbationNHeuristic --> Selector --> EvaluationNHeuristic --> NeuroHeuristic
    ```

    """
    def __init__(self,
                 agent_id: int = 0,
                 num_dimensions: int = 2,
                 num_neighbours: int = 0,
                 num_agents: int = 10,
                 spiking_core: AbstractProcess = None,
                 # num_objectives:    int = 1,
                 function=None,
                 core_params=None,
                 selector_params=None,
                 **kwargs):
        super().__init__(**kwargs)
        internal_shape = (num_dimensions,)
        external_shape = (num_agents, num_dimensions)

        # internal_shape_neighbours = (num_neighbours, num_dimensions)
        external_shape_neighbours = (num_neighbours, num_dimensions, num_agents)

        self.proc_params["agent_id"] = agent_id
        self.proc_params["num_agents"] = num_agents
        self.proc_params["num_dimensions"] = num_dimensions
        self.proc_params["num_neighbours"] = num_neighbours
        # self.proc_params["num_objectives"]  = num_objectives
        self.proc_params["function"] = function
        self.proc_params["core_params"] = core_params
        self.proc_params["selector_params"] = selector_params

        self.proc_params["spiking_core"] = spiking_core

        # These vars come from the spiking_core
        self.x  = Var(shape=internal_shape, init=0.0)
        self.v1 = Var(shape=internal_shape, init=0.0)
        self.v2 = Var(shape=internal_shape, init=0.0)

        # These connect spiking neurons between units
        self.a_in = InPort(shape=external_shape)
        self.s_out = OutPort(shape=external_shape)

        # This receives the global best from the previous iteration
        self.g_in = InPort(shape=internal_shape)
        self.fg_in = InPort(shape=(1,))

        self.p_out = OutPort(shape=external_shape)
        self.fp_out = OutPort(shape=(num_agents,))

        if num_neighbours > 0:
            self.pn_in = InPort(shape=external_shape_neighbours)
            self.fpn_in = InPort(shape=(num_neighbours, num_agents))

        self.core_ref       = core_params.get("core_ref", None)
        self.selector_ref   = selector_params.get("selector_ref", None)

    def reset(self):
        self.x.set(np.zeros(self.x.shape))
        self.v1.set(np.zeros(self.v1.shape))
        self.v2.set(np.zeros(self.v2.shape))

        if self.core_ref and hasattr(self.core_ref, "reset"):
            print("[xxxxxxxxxx reseting core_ref xxxxxxxxxx]")
            self.core_ref.reset()

        if self.selector_ref and hasattr(self.selector_ref, "reset"):
            print("[xxxxxxxxxx reseting selector_ref xxxxxxxxxx]")
            self.selector_ref.reset()


class TensorContractionLayer(AbstractProcess):
    """Tensor Contraction Layer (TCL)
    This layer is used to contract the input tensor along the specified axes. This particular implementation is based on the einsum function from numpy to compute the tensor contraction:
        A_{ij} = \sum_{k} W_{ijk} S_{k}
    where A_{ij} is the output tensor, W_{ijk} is the weight tensor representing the synaptic weights (adjacency), and S_{k} is the input tensor representing the spikes.
    """
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))

        # TODO: Revise and implement checks for the weights tensor
        # self.check_weights(weights)

        self.s_in           = InPort(shape=shape)
        self.weight_matrix  = Var(shape=(shape[0], shape[0]), init=weights)
        self.s_matrix       = Var(shape=shape, init=False)
        self.a_out          = OutPort(shape=shape)

    @staticmethod
    def check_weights(weights):
        if weights:
            if isinstance(weights, np.ndarray):
                if 1 >= len(weights.shape) > 3:
                    raise SchemaError("The weights tensor must be a 2D numpy array.")
                elif len(weights.shape) == 3:
                    raise NotImplementedError(
                        "The weights tensor must be a 2D numpy array. 3D tensors are not supported yet.")
        else:
            raise ValueError("The weights tensor must be provided.")

    def reset(self):
        # No runtime mutable state to reset here (weights and indices are static)
        pass

class NeighbourhoodManager(AbstractProcess):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))

        # TODO: Revise and implement checks for the weights tensor
        # self.check_weights(weights)

        # Compute the number of neighbours for each neuron
        neighbourhoods = np.sum(weights, axis=1).astype(int)
        max_neighbours = np.max(neighbourhoods)

        if max_neighbours == 0:
            raise ValueError("No neighbors found")
        elif np.all(max_neighbours != neighbourhoods):
            raise NotImplementedError("All agents have the same number of neighbors, not yet implemented for this case")

        neighbor_indices = np.argsort(-weights, axis=1)[:, :max_neighbours]

        self.proc_params["num_neighbours"] = max_neighbours
        self.proc_params["num_agents"] = shape[0]
        self.proc_params["num_dimensions"] = shape[1]

        self.weight_matrix = Var(shape=(shape[0], shape[0]), init=weights)
        self.neighbour_indices = Var(shape=(shape[0], max_neighbours), init=neighbor_indices)

        shape_p_out = (max_neighbours, shape[1], shape[0])
        shape_fp_out = (max_neighbours, shape[0])

        # self.ack_in     = InPort(shape=(1,))
        self.p_in = InPort(shape=shape)
        self.fp_in = InPort(shape=(shape[0],))

        self.p_out = OutPort(shape=shape_p_out)
        self.fp_out = OutPort(shape=shape_fp_out)
        # self.ack_out    = OutPort(shape=(1,))

    def reset(self):
        # No runtime mutable state to reset here (weights and indices are static)
        pass

class SpikingHandler(AbstractProcess):
    def __init__(self, agent_id, internal_shape, external_shape, **kwargs):
        super().__init__(**kwargs)
        # Agent ID
        # self.agent_id = Var(shape=(1,), init=agent_id)

        # Ports inside the bounds (going and coming back as vectors)
        self.s_in = InPort(shape=internal_shape)
        self.a_out = OutPort(shape=internal_shape)

        # Ports outside the bounds (going and coming back as matrices)
        self.a_in = InPort(shape=external_shape)
        self.s_out = OutPort(shape=external_shape)

        # Pass the internal shape to the external shape
        self.proc_params["external_shape"] = external_shape
        self.proc_params["internal_shape"] = internal_shape
        self.proc_params["agent_id"] = agent_id

    def reset(self):
        # No runtime state to reset
        pass

class PositionSender(AbstractProcess):
    def __init__(self, agent_id, internal_shape, external_shape, **kwargs):
        super().__init__(**kwargs)
        # Agent ID
        # self.agent_id = Var(shape=(1,), init=agent_id)

        # Ports inside the bound receiving data to be sent
        # self.ack_in = InPort(shape=(1,))
        self.p_in = InPort(shape=internal_shape)
        self.fp_in = InPort(shape=(1,))  # Future: consider multiple objectives

        # Ports outside the bound sending data in the proper shape
        self.p_out = OutPort(shape=external_shape)
        self.fp_out = OutPort(shape=(external_shape[0],))
        # self.ack_out= OutPort(shape=(1,))

        # Pass the internal shape to the external shape
        self.proc_params["external_shape"] = external_shape
        self.proc_params["internal_shape"] = internal_shape
        self.proc_params["agent_id"] = agent_id

    def reset(self):
        # No runtime state to reset
        pass

class PositionReceiver(AbstractProcess):
    def __init__(self, agent_id, internal_shape, external_shape, **kwargs):
        super().__init__(**kwargs)
        # Agent ID
        # self.agent_id = Var(shape=(1,), init=agent_id)

        # Ports outside the bound receiving data in the proper shape
        # self.ack_in = InPort(shape=(1,))
        self.p_in = InPort(shape=external_shape)
        self.fp_in = InPort(shape=(external_shape[0], external_shape[2]))

        # Ports inside the bound sending data in the proper shape
        self.p_out = OutPort(shape=internal_shape)
        self.fp_out = OutPort(shape=(internal_shape[0],))
        # self.ack_out= OutPort(shape=(1,))

        # Pass the internal shape to the external shape
        self.proc_params["external_shape"] = external_shape
        self.proc_params["internal_shape"] = internal_shape
        self.proc_params["agent_id"] = agent_id

    def reset(self):
        # No runtime state to reset
        pass
