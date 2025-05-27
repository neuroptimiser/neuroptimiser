import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import requires

def reset_all_processes(*processes):
    for proc in processes:
        if hasattr(proc, "reset"):
            proc.reset()

class Topology:
    IMPLEMENTED_TOPOLOGIES = ["ring_clockwise", "ring_anticlockwise", "ring_bidirectional",
                              "fully_connected_directed", "fully_connected_bidirectional",
                              "star_outward", "star_inward", "star_bidirectional"]

    def __init__(self):
        pass

    # def get_mask(self, num_dimensions: int, num_neighbours: int, topology: str):
    #     if topology == "ring":
    #         return np.eye(num_dimensions)
    #     elif topology == "fully_connected":
    #         return np.ones((num_neighbours, num_dimensions))
    #     elif topology == "star":
    #         mask = np.zeros((num_neighbours, num_dimensions))
    #         mask[0] = 1
    #         return mask
    #     else:
    #         raise ValueError(f"Invalid topology: {topology}")

    def apply(self, list_of_objects: list[list | object], attribute: str, topology="ring_clockwise"):
        """
        Connect objects based on the specified topology and dimensionality.

        :param list_of_objects: List or list of lists of objects to connect.
        :param attribute: The attribute name used for connection.
        :param topology: The topology to apply (e.g., "ring", "fully_connected").
        """

        # Check if it is a single list of objects
        if not isinstance(list_of_objects[0], list):
            getattr(self, f"apply_{topology}")(list_of_objects, attribute)
        elif isinstance(list_of_objects[0], list):
            getattr(self, f"apply_{topology}_rows")(list_of_objects, attribute)
        else:
            raise ValueError(f"Invalid list of objects: {list_of_objects}")

    @staticmethod
    def apply_ring_clockwise(list_of_objects: list[object], attribute: str):
        for i in range(len(list_of_objects)):
            getattr(list_of_objects[i], attribute + "_out").connect(
                getattr(list_of_objects[(i + 1) % len(list_of_objects)], attribute + "_in"))
            # print(f"Connected {list_of_objects[i].__class__.__name__} {i} -> {(i + 1) % len(list_of_objects)} with {attribute} on ring clockwise topology")

    @staticmethod
    def apply_ring_anticlockwise(list_of_objects: list[object], attribute: str):
        for i in range(len(list_of_objects)):
            getattr(list_of_objects[i], attribute + "_out").connect(
                getattr(list_of_objects[(i - 1) % len(list_of_objects)], attribute + "_in"))
            # print(f"Connected {list_of_objects[i].__class__.__name__} {i} -> {(i - 1) % len(list_of_objects)} with {attribute} on ring anticlockwise topology")

    @staticmethod
    def apply_ring_bidirectional(list_of_objects: list[object], attribute: str):
        for i in range(len(list_of_objects)):
            # # Get the input port of the current node
            # current_in_port = getattr(list_of_objects[i], attribute + "_in")
            #
            # # Get the output ports of the next and previous nodes
            # next_out_port = getattr(list_of_objects[(i + 1) % len(list_of_objects)], attribute + "_out")
            # prev_out_port = getattr(list_of_objects[(i - 1) % len(list_of_objects)], attribute + "_out")
            #
            # # Concatenate the next and previous output ports
            # concat_port = next_out_port.concat_with([prev_out_port], axis=0)
            #
            # # Connect the concatenated output ports to the current input port
            # concat_port.connect(current_in_port)
            #
            # # Print connection details
            # print(
            #     f"Connected {list_of_objects[(i - 1) % len(list_of_objects)].__class__.__name__} {(i - 1) % len(list_of_objects)} "
            #     f"and {list_of_objects[(i + 1) % len(list_of_objects)].__class__.__name__} {(i + 1) % len(list_of_objects)} "
            #     f"to {list_of_objects[i].__class__.__name__} {i} with {attribute} on ring bidirectional topology")

            getattr(list_of_objects[i], attribute + "_out").connect(
                getattr(list_of_objects[(i + 1) % len(list_of_objects)], attribute + "_in"))
            getattr(list_of_objects[i], attribute + "_out").connect(
                getattr(list_of_objects[(i - 1) % len(list_of_objects)], attribute + "_in"))
            # print(f"Connected {(i - 1) % len(list_of_objects)} <- {list_of_objects[i].__class__.__name__} {i} -> {(i + 1) % len(list_of_objects)} with {attribute} on ring bidirectional topology")

    @staticmethod
    def apply_fully_connected_directed(list_of_objects: list[object], attribute: str):
        for i in range(len(list_of_objects)):
            for j in range(len(list_of_objects)):
                if i != j:
                    getattr(list_of_objects[i], attribute + "_out").connect(
                        getattr(list_of_objects[j], attribute + "_in"))
                    # print(f"Connected {list_of_objects[i].__class__.__name__} {i} -> {list_of_objects[j].__class__.__name__} {j} with {attribute} on fully connected directed topology")

    @staticmethod
    def apply_fully_connected_bidirectional(list_of_objects: list[object], attribute: str):
        for i in range(len(list_of_objects)):
            for j in range(i + 1, len(list_of_objects)): # Only connect once
                if i != j:
                    # Forward connection
                    getattr(list_of_objects[i], attribute + "_out").connect(
                        getattr(list_of_objects[j], attribute + "_in"))

                    # Backward connection
                    getattr(list_of_objects[j], attribute + "_out").connect(
                        getattr(list_of_objects[i], attribute + "_in"))
                    # print(f"Connected {list_of_objects[i].__class__.__name__} {i} <-> {list_of_objects[j].__class__.__name__} {j} with {attribute} on fully connected bidirectional topology")

    @staticmethod
    def apply_star_outward(list_of_objects: list[object], attribute: str):
        central_object = list_of_objects[0]
        for i in range(1, len(list_of_objects)):
            getattr(central_object, attribute + "_out").connect(
                getattr(list_of_objects[i], attribute + "_in"))
            # print(f"Connected {central_object.__class__.__name__} -> {list_of_objects[i].__class__.__name__} with {attribute} on star (outward) topology")

    @staticmethod
    def apply_star_inward(list_of_objects: list[object], attribute: str):
        central_object = list_of_objects[0]
        for i in range(1, len(list_of_objects)):
            getattr(list_of_objects[i], attribute + "_out").connect(
                getattr(central_object, attribute + "_in"))
            # print(f"Connected {central_object.__class__.__name__} <- {list_of_objects[i].__class__.__name__} with {attribute} on star (inward) topology")

    @staticmethod
    def apply_star_bidirectional(list_of_objects: list[object], attribute: str):
        central_object = list_of_objects[0]
        for i in range(1, len(list_of_objects)):
            # Forward connection
            getattr(central_object, attribute + "_out").connect(
                getattr(list_of_objects[i], attribute + "_in"))

            # Backward connection
            getattr(list_of_objects[i], attribute + "_out").connect(
                getattr(central_object, attribute + "_in"))
            # print(f"Connected {central_object.__class__.__name__} <-> {list_of_objects[i].__class__.__name__} with {attribute} on star (bidirectional) topology")

    @staticmethod
    def extend_to_columns(method):
        """Decorator to apply a 1D topology method to columns of a 2D grid."""

        def wrapper(grid, *args, **kwargs):
            for col in range(len(grid[0])):
                column = [grid[row][col] for row in range(len(grid))]
                method(column, *args, **kwargs)

        return wrapper

    @staticmethod
    @extend_to_columns
    def apply_ring_clockwise_rows(objects, attribute):
        Topology.apply_ring_clockwise(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_ring_bidirectional_rows(objects, attribute):
        Topology.apply_ring_bidirectional(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_ring_anticlockwise_rows(objects, attribute):
        Topology.apply_ring_anticlockwise(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_fully_connected_directed_rows(objects, attribute):
        Topology.apply_fully_connected_directed(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_fully_connected_bidirectional_rows(objects, attribute):
        Topology.apply_fully_connected_bidirectional(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_star_outward_rows(objects, attribute):
        Topology.apply_star_outward(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_star_inward_rows(objects, attribute):
        Topology.apply_star_inward(objects, attribute)

    @staticmethod
    @extend_to_columns
    def apply_star_bidirectional_rows(objects, attribute):
        Topology.apply_star_bidirectional(objects, attribute)


class AbstractConcatenator(AbstractProcess):
    """Abstract class for a concatenator"""

    x_in: dict[int, InPort]
    x_out: OutPort
    n_inputs: int
    shape: tuple[int]

    def __init__(self, shape):
        super().__init__(shape=shape)

        for i in range(self.n_inputs):
            setattr(self, f"_x_in_{i}", InPort(shape=shape))

        self.x_in = {i: getattr(self, f"_x_in_{i}") for i in range(self.n_inputs)}
        self.x_out = OutPort(shape=(self.n_inputs, shape[0]))


@requires(CPU)
class PyAbstractConcatenatorModel(PyLoihiProcessModel):
    x_in: dict[int, PyInPort]
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    n_inputs: int

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params.get("shape", (1,))

    def run_spk(self):
        _inp_data = np.zeros(self.x_out.shape)
        _inp_data = np.stack(
            [getattr(self, f"_x_in_{i}").recv() for i in range(self.n_inputs)], axis=0
        )
        # print("Concatenated Input:")
        # print(_inp_data)
        self.x_out.send(_inp_data)


class Concatenator:
    def __new__(cls, n_inputs, shape):
        model_attrs = {"n_inputs": n_inputs}

        new_model = type(
            f"PyConcatenatorModel_{n_inputs}",
            (PyAbstractConcatenatorModel,),
            model_attrs,
        )

        new_proc = type(
            f"Concatenator_{n_inputs}",
            (AbstractConcatenator,),
            {"n_inputs": n_inputs, "_model": new_model},
        )

        return new_proc(shape=shape)


class PyConcatenatorModel:
    def __new__(cls, proc):
        model_attrs = {"n_inputs": proc.n_inputs}
        for i in range(proc.n_inputs):
            model_attrs[f"_x_in_{i}"] = LavaPyType(PyInPort.VEC_DENSE, float)

        new_model = type(
            f"PyConcatenatorModel_{proc.n_inputs}",
            (PyAbstractConcatenatorModel,),
            model_attrs,
        )

        # set up implementation details
        new_model.implements_process = type(proc)
        new_model.implements_protocol = LoihiProtocol

        return new_model

# Define the transformation functions for the search space
# * From original to scaled search space: $\mathfrak{X}\in\mathbb{R}^d\to[-1,1]^d$
# * From scaled to origial search space: $[-1,1]^d\to\mathfrak{X}\in\mathbb{R}^d$
def tro2s(x: np.ndarray|float, lb: np.ndarray|float, ub: np.ndarray|float) -> np.ndarray|float:
    return 2 * (x - lb) / (ub - lb) - 1

def trs2o(x: np.ndarray|float, lb: np.ndarray|float, ub: np.ndarray|float) -> np.ndarray|float:
    return (x + 1) / 2 * (ub - lb) + lb

ADJ_MAT_OPTIONS = [
    "one-way-ring", "1dr", "ring",
    "two-way-ring", "2dr", "bidirectional-ring",
    "fully-connected", "all", "full",
    "random", "rand",
]

def get_arch_matrix(length,
                    topology: str = "ring",
                    num_neighbours: int = None):
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

DYN_MODELS_KIND = ["saddle", "attractor", "repeller", "source", "sink"]

def get_2d_sys(kind="sink", trA_max=1.5, detA_max=3.0, eps=1e-6):

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
        else: # Centre
            delta = 0.0

        detA = trAsq4 - delta

        a = 2.0 * np.random.uniform(eps, trA_max) - 1.0
        b = 2.0 * np.random.uniform(eps, trA_max) - 1.0

        d = trA - a
        c = (a * d - detA) / b

    return np.array([[a, b], [c, d]])

IZHIKEVICH_MODELS_KIND = ["RS", "IB", "CH", "FS", "TC", "TCn", "RZ", "LTS", "random"]

def get_izhikevich_sys(kind="RS", scale=0.1):
    if kind == "random":
        kind = np.random.choice(IZHIKEVICH_MODELS_KIND) + "r"
        return get_izhikevich_sys(kind)
    else:
        # Default parameters for Izhikevich model
        a       = 0.02
        b       = 0.2
        c       = -65
        d       = 8.0
        I       = 0.1
        vmin    = -80.    # [V]
        vmax    = 30.
        umin    = -20.    # [V]
        umax    = 0.
        Lt      = 1.0

        if kind == "IB":
            c = -55; d = 4.0
        elif kind == "CH":
            c = -50; d = 2.0
        elif kind == "FS":
            a = 0.1; d = 2.0
        elif kind == "TC":
            a = 0.02; b = 0.25; d = 0.05; I = 0.0
        elif kind == "TCn":
            a = 0.02; b = 0.25; d = 0.05; I = -10.0
        elif kind == "RZ":
            a = 0.1; b = 0.26; d = 2.0
        elif kind == "LTS":
            a = 0.2; b = 0.25; d = 2.0
        else:
            pass # RS

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