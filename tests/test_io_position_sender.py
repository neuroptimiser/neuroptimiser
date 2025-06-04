import numpy as np
import pytest
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import PositionSender

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
def io_position_sender():
    """Fixture to initialize the PerturbationNHeuristic process."""
    return PositionSender(
        agent_id=agent_id,
        internal_shape=(num_dimensions,),
        external_shape=(num_agents, num_dimensions)
    )

def test_initialisation(io_position_sender):
    """Test the initialization of PerturbationNHeuristic."""
    assert io_position_sender.p_in.shape == (num_dimensions,), "Input should be a 2D tensor."
    assert io_position_sender.fp_in.shape == (1,), "Input should be a 1D tensor."
    assert io_position_sender.p_out.shape == (num_agents, num_dimensions), "Output should be a 3D tensor."
    assert io_position_sender.fp_out.shape == (num_agents,), "Output should be a 2D tensor."

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
    p_inp_data = rng.uniform(-1, 1, size=(num_dimensions, num_steps))
    fp_inp_data = rng.uniform(-1, 1, size=(1, num_steps))

    # Create the processes
    internal_source_1 = Source(data=p_inp_data)
    internal_source_2 = Source(data=fp_inp_data)

    sender = PositionSender(agent_id=agent_id,
                            internal_shape=(num_dimensions,),
                            external_shape=(num_agents, num_dimensions))

    external_sink_1 = Sink(shape=sender.p_out.shape, buffer=num_steps)
    external_sink_2 = Sink(shape=sender.fp_out.shape, buffer=num_steps)

    # Wire the processes
    internal_source_1.s_out.connect(sender.p_in)
    internal_source_2.s_out.connect(sender.fp_in)

    sender.p_out.connect(external_sink_1.a_in)
    sender.fp_out.connect(external_sink_2.a_in)

    # Run simulation
    sender.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()
    )

    p_out_data = external_sink_1.data.get().astype(float)
    fp_out_data = external_sink_2.data.get().astype(float)
    sender.stop()

    for step in range(num_steps):
        assert np.allclose(p_inp_data[:, step], p_out_data[agent_id, :, step],
                      atol=1e-6, rtol=1e-6), "Output should match the input data."
        assert np.allclose(fp_inp_data[:, step], fp_out_data[agent_id, step],
                      atol=1e-6, rtol=1e-6), "Output should match the input data."

