import numpy as np
import pytest
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import TensorContractionLayer
from neuroptimiser.utils import get_arch_matrix

rng = np.random.default_rng()
agent_id        = 3
num_agents      = 10
num_neighbours  = 2
num_dimensions  = 7
num_steps       = 10

@pytest.fixture
def run_cfg():
    """Fixture to provide the run configuration."""
    return Loihi2SimCfg()

@pytest.fixture
def tensor_contraction_layer():
    arch_matrix = get_arch_matrix(num_agents, "2dr")
    """Fixture to initialize the PerturbationNHeuristic process."""
    return TensorContractionLayer(weights=arch_matrix, shape=(num_agents, num_dimensions))

def test_initialisation(tensor_contraction_layer):
    """Test the initialization of PerturbationNHeuristic."""
    assert tensor_contraction_layer.s_in.shape == (num_agents, num_dimensions), "Output should be a 2D tensor."
    assert tensor_contraction_layer.a_out.shape == (num_agents, num_dimensions), "Input should be a 3D tensor."

@pytest.mark.parametrize("num_steps, num_agents, topology, num_dimensions", [
    (3, 3, "ring", 2),
    (10, 10, "ring", 7),
    (20, 15, "2dr", 10),
    (30, 20, "all", 12),
    (40, 25, "ring", 100),
    # (100, 30, 20, 200),
])
def test_with_different_input_data(num_steps, num_agents, topology, num_dimensions):
    """Test the process with different input data."""
    # Define the input
    inp_data = rng.integers(0, 2, size=(num_agents, num_dimensions, num_steps))

    # Create the processes
    source = Source(data=inp_data)

    arch_matrix = get_arch_matrix(num_agents, topology)
    dense = TensorContractionLayer(weights=arch_matrix, shape=(num_agents, num_dimensions))

    sink = Sink(shape=dense.a_out.shape, buffer=num_steps)

    # Wire the processes
    source.s_out.connect(dense.s_in)
    dense.a_out.connect(sink.a_in)

    # Run simulation
    dense.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()
    )

    out_data = sink.data.get().astype(int)
    dense.stop()

    for step in range(num_steps):
        T = arch_matrix[..., np.newaxis]
        expected_output = np.einsum('ikj,kj->ij', T, inp_data[:, :, step]).astype(bool)

        print(T, T.shape)
        print(inp_data[:, :, step])
        print(expected_output)

        assert np.allclose(expected_output, out_data[:, :, step],
                           atol=1e-6, rtol=1e-6), "Output should match the input data."
