import numpy as np
import pytest
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import NeighbourhoodManager
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
def neighbourhood_manager():
    arch_matrix = get_arch_matrix(num_agents, "2dr")
    """Fixture to initialize the PerturbationNHeuristic process."""
    return NeighbourhoodManager(weights=arch_matrix, shape=(num_agents, num_dimensions))

def test_initialisation(neighbourhood_manager):
    """Test the initialization of PerturbationNHeuristic."""
    assert neighbourhood_manager.p_in.shape == (num_agents, num_dimensions), "Output should be a 2D tensor."
    assert neighbourhood_manager.fp_in.shape == (num_agents,), "Output should be a 1D tensor."
    assert neighbourhood_manager.p_out.shape == (num_neighbours, num_dimensions, num_agents), "Input should be a 3D tensor."
    assert neighbourhood_manager.fp_out.shape == (num_neighbours, num_agents), "Input should be a 2D tensor."

@pytest.mark.parametrize("num_steps, num_agents, topology, num_dimensions", [
    (10, 10, "ring", 7),
    (20, 15, "2dr", 10),
    (30, 20, "all", 12),
    (40, 25, "ring", 100),
    # (100, 30, 20, 200),
])
def test_with_different_input_data(num_steps, num_agents, topology, num_dimensions):
    """Test the process with different input data."""
    # Define the input
    inp_data_p = np.random.rand(num_agents, num_dimensions, num_steps)
    inp_data_fp = np.random.rand(num_agents, num_steps)

    # Create the processes
    source_p = Source(data=inp_data_p)
    source_fp = Source(data=inp_data_fp)

    arch_matrix = get_arch_matrix(num_agents, topology)
    manager = NeighbourhoodManager(weights=arch_matrix,
                                   shape=(num_agents, num_dimensions))

    neighbour_indices = manager.neighbour_indices.get()

    num_neighbours = manager.fp_out.shape[0]

    sink_p = Sink(shape=manager.p_out.shape, buffer=num_steps)
    sink_fp = Sink(shape=manager.fp_out.shape, buffer=num_steps)

    # Wire the processes
    source_p.s_out.connect(manager.p_in)
    source_fp.s_out.connect(manager.fp_in)

    manager.p_out.connect(sink_p.a_in)
    manager.fp_out.connect(sink_fp.a_in)

    # Run simulation
    manager.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()  # select_tag="floating_pt",)
    )

    out_data_p = sink_p.data.get().astype(float)
    out_data_fp = sink_fp.data.get().astype(float)
    manager.stop()

    for step in range(1, num_steps):
        expected_output_p = np.zeros((num_neighbours, num_dimensions, num_agents))
        expected_output_fp = np.zeros((num_neighbours, num_agents))
        for i in range(num_agents):
            A = inp_data_p[:, :, step]
            fA = inp_data_fp[:, step]
            expected_output_p[:, :, i] = A[neighbour_indices[i]]
            expected_output_fp[:, i] = fA[neighbour_indices[i]]

        assert np.allclose(expected_output_p, out_data_p[:, :, :, step],
                           atol=1e-6, rtol=1e-6), "Output should match the input data."
        assert np.allclose(expected_output_fp, out_data_fp[:, :, step],
                           atol=1e-6, rtol=1e-6), "Output should match the input data."

