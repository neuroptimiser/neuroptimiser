"""Neuroptimiser Core Processes

This module contains the core processes for the Neuroptimiser framework based on Lava.
"""
__author__ = "Jorge M. Cruz-Duarte"
__email__ = "jorge.cruz-duarte@univ-lille.fr"
__version__ = "1.0.0"
__all__ = ["AbstractSpikingCore", "TwoDimSpikingCore", "Selector", "HighLevelSelection",
              "NeuroHeuristicUnit", "TensorContractionLayer", "NeighbourhoodManager",
                "SpikingHandler", "PositionSender", "PositionReceiver"]

import numpy as np
import time
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from neuroptimiser.utils import (
    get_2d_sys, get_izhikevich_sys
)

# %% General elements
class AbstractSpikingCore(AbstractProcess):
    """Abstract process class for a spiking core

    This class is designed to be used as a base class for spiking neuron models. It initialises the core parameters and sets up the necessary ports and variables for the spiking core.

    Attributes
    ----------
        Inports
            s_in : InPort
                Input port for the spiking activity.
            p_in : InPort
                Input port for the position variable.
            fp_in : InPort
                Input port for the fitness variable.
            g_in : InPort
                Input port for the global best position.
            fg_in : InPort
                Input port for the global best fitness.
            xn_in : InPort
                Input port for the neighbours' positions.
            fxn_in : InPort
                Input port for the neighbours' fitness.
        Variables
            x : Var
                Variable for the current position of the spiking core.
        Outports
            s_out : OutPort
                Output port for the spiking activity.
            x_out : OutPort
                Output port for the current position of the spiking core.
    """
    def __init__(self,
                 noise_std: float | tuple | list = 0.1,
                 alpha: float = 1.0,
                 max_steps: int = 100,
                 **kwargs):
        """
        Initialise the spiking core with the given parameters.

        Keyword Arguments
        -----------------
            noise_std: float | tuple | list, optional
                Standard deviation of the noise added to the spiking core. If a tuple or list is provided, a random value will be sampled from the range defined by the tuple/list. Default is 0.1.
            alpha: float, optional
                Scaling factor for the spiking core. Default is 1.0.
            max_steps: int, optional:
                Maximum number of steps for the spiking core to run. Default is 100.
            num_dimensions: int, optional
                Number of dimensions for the spiking core. Default is 2.
            init_position: np.ndarray, optional
                Initial position of the spiking core in the defined dimensions. If not provided, a random position will be generated within the range [-1, 1].
            num_neighbours: int, optional
                Number of neighbours for the spiking core. Default is 0, meaning no neighbours are considered.
        """
        super().__init__(**kwargs)

        num_dimensions  = kwargs.get("num_dimensions", 2)
        init_position   = kwargs.get("init_position", None)
        num_neighbours  = kwargs.get("num_neighbours", 0)

        self.shape      = (num_dimensions,)
        shape_fx        = (1,)
        shape_xn        = (num_neighbours, num_dimensions)
        shape_fxn       = (num_neighbours,)

        self.init_position  = init_position
        start_position      = init_position \
            if init_position is not None \
            else np.random.uniform(-1, 1, self.shape)

        # Inports
        self.s_in       = InPort(shape=self.shape)
        self.p_in       = InPort(shape=self.shape)
        self.fp_in      = InPort(shape=shape_fx)
        self.g_in       = InPort(shape=self.shape)
        self.fg_in      = InPort(shape=shape_fx)
        self.xn_in      = InPort(shape=shape_xn)
        self.fxn_in     = InPort(shape=shape_fxn)

        # Variables
        self.x          = Var(shape=self.shape, init=start_position)

        # Outports
        self.s_out      = OutPort(shape=self.shape)
        self.x_out      = OutPort(shape=self.shape)

        # Read and prepare the common parameters
        alpha       = kwargs.get("alpha", 1.0)
        max_steps   = kwargs.get("max_steps", 100)
        noise_std       = kwargs.get("noise_std", 0.1)
        if isinstance(noise_std, (tuple, list)):
            noise_std_val   = np.random.uniform(noise_std[0], noise_std[1])
        else:
            noise_std_val   = noise_std

        self.proc_params["alpha"]       = alpha
        self.proc_params["max_steps"]   = max_steps
        self.proc_params["noise_std"]   = noise_std_val

    def reset(self):
        """Reset the spiking core to its initial state."""
        start_position = self.init_position if self.init_position is not None else np.random.uniform(-1, 1, self.shape)
        self.x.set(start_position)


class TwoDimSpikingCore(AbstractSpikingCore):
    """Two-Dimensional Spiking Core Process

    This process implements a two-dimensional spiking core with specific neuron models such as linear and Izhikevich. It allows for the definition of coefficients for the models and provides random initial values for the variables v1 and v2.

    Attributes
    ----------
        Inports
            s_in : InPort
                Input port for the spiking activity.
            p_in : InPort
                Input port for the position variable.
            fp_in : InPort
                Input port for the fitness variable.
            g_in : InPort
                Input port for the global best position.
            fg_in : InPort
                Input port for the global best fitness.
            xn_in : InPort
                Input port for the neighbours' positions.
            fxn_in : InPort
                Input port for the neighbours' fitness.
        Variables
            x : Var
                Variable for the current position of the spiking core.
            v1 : Var
                Variable for the first neuromorphic state (e.g., membrane potential).
            v2 : Var
                Variable for the second neuromorphic state (e.g., adaptation/auxiliary variable).
        Outports
            s_out : OutPort
                Output port for the spiking activity.
            x_out : OutPort
                Output port for the new current position of the spiking core.

    See Also
    --------
    AbstractSpikingCore : Base class for spiking cores.
    :py:class:`neuroptimiser.core.models.PyTwoDimSpikingCoreModel` : Model implementation of the TwoDimSpikingCore process.
    """
    def __init__(self, **core_params):
        """
        Initialise the TwoDimSpikingCore with the given parameters.

        Keyword Arguments
        -----------------
            name: str, optional
                Name of the neuron model to be used. Options are ``linear`` or ``izhikevich``. Default is ``linear``.
            coeffs: str | list | np.ndarray, optional
                Coefficients for the neuron model. If a string is provided, it should be in the format ``model_kind`` or ``model_kind_all``. If a list or numpy array is provided, it should contain the coefficients for each dimension. Default is None, which uses default coefficients.
            seed: int, optional
                Seed for the random number generator. Default is a random value based on the current time.
        Notes
        -----
            - The ``linear`` model uses a default coefficient matrix of [[-0.5, -0.5], [0.5, -0.5]].
            - The ``izhikevich`` model uses default coefficients for the regular spiking (RS) neuron type: a=0.02, b=0.2, c=-65, d=8, I=0.1.

        """
        super().__init__(**core_params)

        _name = core_params.get("name", "linear")

        # Read and prepare the specific parameters
        if _name in ["linear", "izhikevich"]:
            models_coeffs               = core_params.get("coeffs", None)

            if _name == "izhikevich":
                _coefficients = self.process_izh_coeffs(models_coeffs)
            else: #if _name == "linear":
                _coefficients = self.process_lin_coeffs(models_coeffs)
            self.proc_params["coeffs_values"] = _coefficients

        # Public Variables
        seed = core_params.get("seed", int(time.time() * 1000) % (2 ** 32))
        self.rng = np.random.default_rng(seed)
        self.v1 = Var(
            shape=self.shape,
            init=self.rng.uniform(-1.0, 1.0, size=self.shape))
        self.v2 = Var(
            shape=self.shape,
            init=self.rng.uniform(-1.0, 1.0, size=self.shape))

    def process_lin_coeffs(self, models_coeffs=None) -> list:
        """Process the coefficients for linear model.

        Parameters
        ----------
            models_coeffs : str | list | np.ndarray, optional
                Coefficients for the linear model. If a string is provided, it should be in the format ``model_kind`` or ``model_kind_all``. If a list or numpy array is provided, it should contain the coefficients for each dimension. Default is None, which uses default coefficients.
        Returns
        -------
        models_coeffs_ : list
            Processed coefficients for the linear model.
        """
        models_coeffs_ = []
        if models_coeffs is None:
            models_coeffs_ = [np.array([[-0.5, -0.5], [0.5, -0.5]])]
        elif isinstance(models_coeffs, str):
            _model_split = models_coeffs.split("_")
            if len(_model_split) > 1:
                _kind = models_coeffs.split("_")[0]
                _many = models_coeffs.split("_")[1] == "all"
            else:
                _kind = models_coeffs
                _many = False

            for _ in range(self.shape[0]):
                models_coeffs_.append(get_2d_sys(_kind))
                if not _many:
                    break
        else:
            if isinstance(models_coeffs, (list, np.ndarray)) and (
                    len(models_coeffs) == 1 or len(models_coeffs) == self.shape[0]):
                models_coeffs_ = models_coeffs
            else:
                raise ValueError("The number of models must be equal to the number of dimensions or 1")
        return models_coeffs_

    def process_izh_coeffs(self, coeffs=None) -> list:
        """Process the coefficients for Izhikevich model.

        Parameters
        ----------
            coeffs : str | list | np.ndarray, optional
                Coefficients for the Izhikevich model. If a string is provided, it should be in the format ``model_kind`` or ``model_kind_all``. If a list or numpy array is provided, it should contain the coefficients for each dimension. Default is None, which uses default coefficients.
        Returns
        -------
        coeffs_ : list
            Processed coefficients for the Izhikevich model.
        """

        coeffs_ = []
        if coeffs is None: # Default values (RS)
            coeffs_ = [{"a": 0.02, "b": 0.2, "c": -65, "d": 8, "I": 0.1}]
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

    def reset(self) -> None:
        """Reset the `TwoDimSpikingCore` to its initial state."""
        super().reset()
        self.v1.set(self.rng.uniform(-1.0, 1.0, size=self.shape))
        self.v2.set(self.rng.uniform(-1.0, 1.0, size=self.shape))


class Selector(AbstractProcess):
    """Selector Process

    This process is designed to select the best position and fitness from the spiking core's current and new candidate based on a given function.

    Attributes
    ----------
        Inports
            x_in : InPort
                Input port for the position variable.
        Variables
            p : Var
                Variable for the position of the agent.
            fp : Var
                Variable for the fitness of the agent.
        Outports
            p_out : OutPort
                Output port for the position variable.
            fp_out : OutPort
                Output port for the fitness variable.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PySelectorModel` : Model implementation of the Selector process.
    """
    def __init__(self,
                 agent_id: int = 0,
                 num_agents: int = 1,
                 num_dimensions: int = 2,
                 function=None,
                 sel_mode="greedy",
                 **kwargs):
        """Initialise the Selector with the given parameters.

        Parameters
        ----------
            agent_id : int, optional
                ID of the agent for which the selector is being created. Default is 0.
            num_agents : int, optional
                Number of agents in the system. Default is 1.
            num_dimensions : int, optional
                Number of dimensions for the position and fitness variables. Default is 2.
            function : callable, optional
                Function to be used for evaluating the fitness of the positions. Default is None, which means no function is applied but an error will be raised if not provided.

        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.
        """
        super().__init__(**kwargs)
        shape       = (num_dimensions,)  # assuming [[x1, x2, ..., xn]]
        self.shape  = shape

        self.proc_params["agent_id"]    = agent_id
        self.proc_params["num_agents"]  = num_agents
        self.proc_params["function"]    = function
        self.proc_params["sel_mode"]    = sel_mode

        # Inports
        self.x_in   = InPort(shape=shape)

        # Variables
        self.p      = Var(shape=shape, init=0.0)
        self.fp     = Var(shape=(1,), init=6.9)

        # Outports
        self.p_out  = OutPort(shape=shape)
        self.fp_out = OutPort(shape=(1,))

    def reset(self) -> None:
        """Reset the Selector to its initial state."""
        self.p.set(np.zeros(self.shape))
        self.fp.set(np.array([6.9]))


class HighLevelSelection(AbstractProcess):
    """High-Level Selection Process

    This process is designed to select the global best position and fitness from current candidates from spiking cores based on a given metric, i.e., fitness.

    Attributes
    ----------
        Inports
            p_in : InPort
                Input port for the position variable.
            fp_in : InPort
                Input port for the fitness variable.
        Variables
            p : Var
                Variable for the position of the agent.
            fp : Var
                Variable for the fitness of the agent.
            g : Var
                Variable for the global best position.
            fg : Var
                Variable for the global best fitness.
        Outports
            g_out : OutPort
                Output port for the global best position.
            fg_out : OutPort
                Output port for the global best fitness.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PyHighLevelSelectorModel` : Model implementation of the HighLevelSelection process.
    """
    def __init__(self,
                 num_dimensions: int = 2,
                 num_agents: int = 1,
                 **kwargs):
        """Initialise the HighLevelSelection with the given parameters.

        Parameters
        ----------
            num_dimensions : int, optional
                Number of dimensions for the position and fitness variables. Default is 2.
            num_agents : int, optional
                Number of agents in the system. Default is 1.

        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.

        """
        super().__init__(**kwargs)
        shape = (num_agents, num_dimensions)

        # Inports
        self.p_in   = InPort(shape=shape)
        self.fp_in  = InPort(shape=(num_agents,))

        # Variables
        self.p      = Var(shape=shape, init=0.0)
        self.fp     = Var(shape=(num_agents,), init=6.9)
        self.g      = Var(shape=(num_dimensions,), init=0.0)
        self.fg     = Var(shape=(1,), init=6.9)

        # Outports
        self.g_out  = OutPort(shape=(num_dimensions,))
        self.fg_out = OutPort(shape=(1,))

        self.proc_params["num_agents"] = num_agents
        self.proc_params["num_dimensions"] = num_dimensions

    def reset(self) -> None:
        """Reset the HighLevelSelection to its initial state."""
        self.p.set(np.zeros(self.p.shape))
        self.fp.set(np.array([6.9] * self.fp.shape[0]))
        self.g.set(np.zeros(self.g.shape))
        self.fg.set(np.array([6.9]))


class NeuroHeuristicUnit(AbstractProcess):
    """General model for a Neuro-Heuristic (Nheuristic) unit

    This unit can be used to define a complete neuro-heuristic model using the "atomic" processes of the Neuroptimiser framework, such as Spiking Core, Selector, Spiking Handler, Position Sender, and Position Receiver.

    Attributes
    ----------
        Inports
            a_in : InPort
                Input port for the spiking activity from other agents.
            g_in : InPort
                Input port for the global best position.
            fg_in : InPort
                Input port for the global best fitness.
            pn_in : InPort, optional
                Input port for the positions of the neighbours.
            fpn_in : InPort, optional
                Input port for the fitness of the neighbours.
        Variables
            x : Var
                Variable for the current position of the unit.
            v1 : Var
                Variable for the first neuromorphic state (e.g., membrane potential).
            v2 : Var
                Variable for the second neuromorphic state (e.g., adaptation/auxiliary variable).
        Outports
            s_out : OutPort
                Output port for the spiking activity to other agents.
            p_out : OutPort
                Output port for the current position of the unit.
            fp_out : OutPort
                Output port for the fitness of the unit.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PyNeuroHeuristicUnitModel` : Model implementation of the NeuroHeuristicUnit process.
    """
    def __init__(self,
                 agent_id: int = 0,
                 num_dimensions: int = 2,
                 num_neighbours: int = 0,
                 num_agents: int = 10,
                 spiking_core: AbstractProcess = None,
                 function=None,
                 core_params=None,
                 selector_params=None,
                 **kwargs):
        """Initialise the NeuroHeuristicUnit with the given parameters.

        Parameters
        ----------
            agent_id : int, optional
                ID of the agent for which the unit is being created. Default is 0.
            num_dimensions : int, optional
                Number of dimensions for the position and fitness variables. Default is 2.
            num_neighbours : int, optional
                Number of neighbours for the unit. Default is 0, meaning no neighbours are considered.
            num_agents : int, optional
                Number of agents in the system. Default is 10.
            spiking_core : AbstractProcess, optional
                Instance of a spiking core process to be used in the unit. If not provided, a default spiking core will be created.
            function : callable, optional
                Function to be used for evaluating the fitness of the positions. Default is None, which means no function is applied but an error will be raised if not provided.
            core_params : dict, optional
                Parameters for the spiking core process. If not provided, default parameters will be used.
            selector_params : dict, optional
                Parameters for the selector process. If not provided, default parameters will be used.

        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.
        """
        super().__init__(**kwargs)
        internal_shape              = (num_dimensions,)
        external_shape              = (num_agents, num_dimensions)
        external_shape_neighbours   = (num_neighbours, num_dimensions, num_agents)

        self.proc_params["agent_id"]        = agent_id
        self.proc_params["num_agents"]      = num_agents
        self.proc_params["num_dimensions"]  = num_dimensions
        self.proc_params["num_neighbours"]  = num_neighbours
        self.proc_params["function"]        = function
        self.proc_params["core_params"]     = core_params
        self.proc_params["selector_params"] = selector_params

        self.proc_params["spiking_core"]    = spiking_core

        # These vars come from the spiking_core
        self.x      = Var(shape=internal_shape, init=0.0)
        self.v1     = Var(shape=internal_shape, init=0.0)
        self.v2     = Var(shape=internal_shape, init=0.0)

        # These connect spiking neurons between units
        self.a_in   = InPort(shape=external_shape)
        self.s_out  = OutPort(shape=external_shape)

        # This receives the global best from the previous iteration
        self.g_in   = InPort(shape=internal_shape)
        self.fg_in  = InPort(shape=(1,))

        self.p_out  = OutPort(shape=external_shape)
        self.fp_out = OutPort(shape=(num_agents,))

        if num_neighbours > 0:
            self.pn_in  = InPort(shape=external_shape_neighbours)
            self.fpn_in = InPort(shape=(num_neighbours, num_agents))

        self.core_ref       = core_params.get("core_ref", None)
        self.selector_ref   = selector_params.get("selector_ref", None)

    def reset(self) -> None:
        """Reset the NeuroHeuristicUnit to its initial state."""
        self.x.set(np.zeros(self.x.shape))
        self.v1.set(np.zeros(self.v1.shape))
        self.v2.set(np.zeros(self.v2.shape))

        if self.core_ref and hasattr(
                self.core_ref, "reset"):
            self.core_ref.reset()

        if self.selector_ref and hasattr(
                self.selector_ref, "reset"):
            self.selector_ref.reset()


class TensorContractionLayer(AbstractProcess):
    """Tensor Contraction Layer Process

    This process implements a tensor contraction layer that takes a weight matrix and computes the output based on the input tensor. It is designed to be used in spiking neural network architectures where tensor contractions are required.

    Attributes
    ----------
        Inports
            s_in : InPort
                Input port for the input tensor.
        Variables
            weight_matrix : Var
                Variable for the weight matrix used in the tensor contraction.
            s_matrix : Var
                Variable for the output tensor after the contraction.
        Outports
            a_out : OutPort
                Output port for the output tensor after the contraction.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PyTensorContractionLayerModel` : Model implementation of the TensorContractionLayer process.
    """
    def __init__(self, weights, **kwargs):
        """Initialise the TensorContractionLayer with the given parameters.

        Parameters
        ----------
            weights : np.ndarray
                Weight matrix for the tensor contraction. It should be a 2D numpy array.

        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.
        """
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))

        # Inports
        self.s_in           = InPort(shape=shape)

        # Variables
        self.weight_matrix  = Var(shape=(shape[0], shape[0]), init=weights)
        self.s_matrix       = Var(shape=shape, init=False)

        # Outports
        self.a_out          = OutPort(shape=shape)

    def reset(self):
        """Reset the TensorContractionLayer to its initial state.

        This method is currently a placeholder and does not perform any operations for compatibility purposes.
        """
        pass

class NeighbourhoodManager(AbstractProcess):
    """Neighbourhood Manager Process

    This process manages the neighbourhood of units based on a weight matrix. It computes the number of neighbours for each unit and stores the indices of the neighbours.

    Attributes
    ----------
        Inports
            p_in : InPort
                Input port for the position tensor.
            fp_in : InPort
                Input port for the fitness tensor.
        Variables
            weight_matrix : Var
                Variable for the weight matrix used to determine neighbourhoods.
            neighbour_indices : Var
                Variable for storing the indices of neighbours for each unit.
        Outports
            p_out : OutPort
                Output port for the position tensor of neighbours.
            fp_out : OutPort
                Output port for the fitness tensor of neighbours.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PyNeighbourhoodManagerModel` : Model implementation of the NeighbourhoodManager process.
    """
    def __init__(self, weights, **kwargs):
        """Initialise the NeighbourhoodManager with the given parameters.

        Parameters
        ----------
            weights : np.ndarray
                Weight matrix for the neighbourhoods. It should be a 2D numpy array where each row represents the weights of a unit to its neighbours.

        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.

        """
        super().__init__(**kwargs)
        shape           = kwargs.get("shape", (1, 1))

        # Compute the number of neighbours for each neuron
        neighbourhoods  = np.sum(weights, axis=1).astype(int)
        max_neighbours  = np.max(neighbourhoods)

        if max_neighbours == 0:
            raise ValueError("No neighbors found")
        elif np.all(max_neighbours != neighbourhoods):
            raise NotImplementedError("All agents have the same number of neighbors, not yet implemented for this case")

        neighbor_indices    = np.argsort(-weights, axis=1)[:, :max_neighbours]
        shape_p_out         = (max_neighbours, shape[1], shape[0])
        shape_fp_out        = (max_neighbours, shape[0])

        self.proc_params["num_neighbours"]  = max_neighbours
        self.proc_params["num_agents"]      = shape[0]
        self.proc_params["num_dimensions"]  = shape[1]

        # Inports
        self.p_in               = InPort(shape=shape)
        self.fp_in              = InPort(shape=(shape[0],))

        # Variables
        self.weight_matrix      = Var(shape=(shape[0], shape[0]), init=weights)
        self.neighbour_indices  = Var(shape=(shape[0], max_neighbours), init=neighbor_indices)

        # Outports
        self.p_out              = OutPort(shape=shape_p_out)
        self.fp_out             = OutPort(shape=shape_fp_out)

    def reset(self) -> None:
        """Reset the NeighbourhoodManager to its initial state.

        This method is currently a placeholder and does not perform any operations for compatibility purposes.
        """
        pass

class SpikingHandler(AbstractProcess):
    """Spiking Handler Process

    This process handles the spiking activity of units back and forth between the internal and external bounds. It manages the input and output ports for spiking activity, allowing units to communicate their states.

    Attributes
    ----------
        Inports
            s_in : InPort
                Input port for the spiking activity from inside the bounds.
            a_in : InPort
                Input port for the spiking activity from outside the bounds.
        Outports
            a_out : OutPort
                Output port for the spiking activity to outside the bounds.
            s_out : OutPort
                Output port for the spiking activity to inside the bounds.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PySpikingHandlerModel` : Model implementation of the SpikingHandler process.
    """
    def __init__(self, agent_id, internal_shape, external_shape, **kwargs):
        """Initialise the SpikingHandler with the given parameters.

        Parameters
        ----------
            agent_id : int
                ID of the agent for which the spiking handler is being created.
            internal_shape : tuple
                Shape of the internal spiking activity (e.g., (num_dimensions,)).
            external_shape : tuple
                Shape of the external spiking activity (e.g., (num_agents, num_dimensions)).

        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.
        """
        super().__init__(**kwargs)

        # Ports inside the bounds (going and coming back as vectors)
        self.s_in   = InPort(shape=internal_shape)
        self.a_out  = OutPort(shape=internal_shape)

        # Ports outside the bounds (going and coming back as matrices)
        self.a_in   = InPort(shape=external_shape)
        self.s_out  = OutPort(shape=external_shape)

        # Pass the internal shape to the external shape
        self.proc_params["agent_id"]        = agent_id
        self.proc_params["external_shape"]  = external_shape
        self.proc_params["internal_shape"]  = internal_shape

    def reset(self) -> None:
        """Reset the SpikingHandler to its initial state.

        This method is currently a placeholder and does not perform any operations for compatibility purposes.
        """
        pass

class PositionSender(AbstractProcess):
    """Position Sender Process

    This process is designed to send the position and fitness of a unit from inside the bounds to outside the bounds. It manages the input and output ports for position and fitness data, allowing units to communicate their states.

    Attributes
    ----------
        Inports
            p_in : InPort
                Input port for the position tensor from inside the bounds.
            fp_in : InPort
                Input port for the fitness tensor from inside the bounds.
        Outports
            p_out : OutPort
                Output port for the position tensor to outside the bounds.
            fp_out : OutPort
                Output port for the fitness tensor to outside the bounds.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PyPositionSenderModel` : Model implementation of the PositionSender process.
    """
    def __init__(self, agent_id, internal_shape, external_shape, **kwargs):
        """Initialise the PositionSender with the given parameters.

        Parameters
        ----------
            agent_id : int
                ID of the agent for which the position sender is being created.
            internal_shape : tuple
                Shape of the internal position and fitness (e.g., (num_dimensions,)).
            external_shape : tuple
                Shape of the external position and fitness (e.g., (num_agents, num_dimensions)).
        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.
        """
        super().__init__(**kwargs)
        self.p_in       = InPort(shape=internal_shape)
        self.fp_in      = InPort(shape=(1,))

        # Ports outside the bound sending data in the proper shape
        self.p_out      = OutPort(shape=external_shape)
        self.fp_out     = OutPort(shape=(external_shape[0],))

        # Pass the internal shape to the external shape
        self.proc_params["agent_id"]        = agent_id
        self.proc_params["external_shape"]  = external_shape
        self.proc_params["internal_shape"]  = internal_shape

    def reset(self) -> None:
        """Reset the PositionSender to its initial state.

        This method is currently a placeholder and does not perform any operations for compatibility purposes.
        """
        pass

class PositionReceiver(AbstractProcess):
    """Position Receiver Process

    This process is designed to receive the position and fitness of a unit from outside the bounds and send it to inside the bounds. It manages the input and output ports for position and fitness data, allowing units to communicate their states.

    Attributes
    ----------
        Inports
            p_in : InPort
                Input port for the position tensor from outside the bounds.
            fp_in : InPort
                Input port for the fitness tensor from outside the bounds.
        Outports
            p_out : OutPort
                Output port for the position tensor to inside the bounds.
            fp_out : OutPort
                Output port for the fitness tensor to inside the bounds.

    See Also
    --------
    :py:class:`neuroptimiser.core.models.PyPositionReceiverModel` : Model implementation of the PositionReceiver process.
    """
    def __init__(self, agent_id, internal_shape, external_shape, **kwargs):
        """Initialise the PositionReceiver with the given parameters.

        Parameters
        ----------
            agent_id : int
                ID of the agent for which the position receiver is being created.
            internal_shape : tuple
                Shape of the internal position and fitness (e.g., (num_dimensions,)).
            external_shape : tuple
                Shape of the external position and fitness (e.g., (num_agents, num_dimensions)).
        Keyword Arguments
        -----------------
            **kwargs : dict, optional
                Additional keyword arguments to be passed to the parent class `AbstractProcess`.
        """
        super().__init__(**kwargs)

        # Ports outside the bound receiving data in the proper shape
        self.p_in   = InPort(shape=external_shape)
        self.fp_in  = InPort(shape=(external_shape[0], external_shape[2]))

        # Ports inside the bound sending data in the proper shape
        self.p_out  = OutPort(shape=internal_shape)
        self.fp_out = OutPort(shape=(internal_shape[0],))

        # Pass the internal shape to the external shape
        self.proc_params["agent_id"]        = agent_id
        self.proc_params["external_shape"]  = external_shape
        self.proc_params["internal_shape"]  = internal_shape

    def reset(self) -> None:
        """Reset the PositionReceiver to its initial state.

        This method is currently a placeholder and does not perform any operations for compatibility purposes.
        """
        pass
