import numpy as np
import pytest
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import PositionReceiver

rng = np.random.default_rng()
agent_id        = 3
num_agents      = 10
num_neighbours  = 4
num_dimensions  = 7
num_steps       = 10

@pytest.fixture
def run_cfg():
    """Fixture to provide the run configuration."""
    return Loihi2SimCfg()

@pytest.fixture
def io_position_receiver():
    """Fixture to initialize the PerturbationNHeuristic process."""
    return PositionReceiver(
        agent_id=agent_id,
        internal_shape=(num_neighbours, num_dimensions),
        external_shape=(num_neighbours, num_dimensions, num_agents)
    )

def test_initialisation(io_position_receiver):
    """Test the initialization of PerturbationNHeuristic."""
    assert io_position_receiver.p_out.shape == (num_neighbours, num_dimensions), "Output should be a 3D tensor."
    assert io_position_receiver.fp_out.shape == (num_neighbours,), "Output should be a 2D tensor."
    assert io_position_receiver.p_in.shape == (num_neighbours, num_dimensions, num_agents), "Input should be a 2D tensor."
    assert io_position_receiver.fp_in.shape == (num_neighbours, num_agents), "Input should be a 1D tensor."

@pytest.mark.parametrize("num_steps, num_agents, num_neighbours, num_dimensions", [
    (10, 10, 4, 7),
    (20, 15, 5, 10),
    (30, 20, 6, 12),
    (40, 25, 7, 100),
    # (100, 30, 20, 200),
])
def test_with_different_input_data(num_steps, num_agents, num_neighbours, num_dimensions):
    """Test the process with different input data."""
    # Define the input
    p_inp_data = rng.uniform(-1, 1,
                             size=(num_neighbours, num_dimensions, num_agents, num_steps))
    fp_inp_data = rng.uniform(-1, 1, size=(num_neighbours, num_agents, num_steps))

    # Create the processes
    internal_source_1 = Source(data=p_inp_data)
    internal_source_2 = Source(data=fp_inp_data)

    receiver = PositionReceiver(
        agent_id=agent_id,
        internal_shape=(num_neighbours, num_dimensions),
        external_shape=(num_neighbours, num_dimensions, num_agents)
    )

    external_sink_1 = Sink(shape=receiver.p_out.shape, buffer=num_steps)
    external_sink_2 = Sink(shape=receiver.fp_out.shape, buffer=num_steps)

    # Wire the processes
    internal_source_1.s_out.connect(receiver.p_in)
    internal_source_2.s_out.connect(receiver.fp_in)

    receiver.p_out.connect(external_sink_1.a_in)
    receiver.fp_out.connect(external_sink_2.a_in)

    # Run simulation
    receiver.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()
    )

    p_out_data = external_sink_1.data.get().astype(float)
    fp_out_data = external_sink_2.data.get().astype(float)
    receiver.stop()

    for step in range(num_steps):
        assert np.allclose(p_inp_data[:, :, agent_id, step], p_out_data[:, :, step],
                           atol=1e-6, rtol=1e-6), "Output should match the input data."
        assert np.allclose(fp_inp_data[:, agent_id, step], fp_out_data[:, step],
                           atol=1e-6, rtol=1e-6), "Output should match the input data."

