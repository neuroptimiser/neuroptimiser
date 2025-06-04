import numpy as np
import pytest
from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import SpikingHandler
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
def io_spiking_signal_handler():
    """Fixture to initialize the PerturbationNHeuristic process."""
    return SpikingHandler(agent_id=agent_id,
                          internal_shape=(num_dimensions,),
                          external_shape=(num_agents, num_dimensions))

def test_initialisation(io_spiking_signal_handler):
    """Test the initialization of PerturbationNHeuristic."""
    assert io_spiking_signal_handler.s_in.shape == (num_dimensions,), "Output should be a 1D tensor."
    assert io_spiking_signal_handler.a_in.shape == (num_agents, num_dimensions), "Input should be a 2D tensor."
    assert io_spiking_signal_handler.s_out.shape == (num_agents, num_dimensions), "Output should be a 2D tensor."
    assert io_spiking_signal_handler.a_out.shape == (num_dimensions,), "Input should be a 1D tensor."

@pytest.mark.parametrize("num_steps, num_agents, num_dimensions", [
    (5, 10, 7),
    (20, 15, 10),
    (30, 20, 12),
    (40, 25, 100),
])
def test_with_different_input_data(num_steps, num_agents, num_dimensions):
    """Test the process with different input data."""

    # Create a random agent ID
    agent_id = rng.integers(0, num_agents)

    # Define the input
    s_inp_data = rng.integers(0, 2, size=(num_dimensions, num_steps))
    a_inp_data = rng.integers(0, 2, size=(num_agents, num_dimensions, num_steps))

    # Create the processes
    internal_source = Source(data=s_inp_data)
    external_source = Source(data=a_inp_data)

    handler = SpikingHandler(agent_id=agent_id,
                             internal_shape=(num_dimensions,),
                             external_shape=(num_agents, num_dimensions))

    internal_sink = Sink(shape=handler.a_out.shape, buffer=num_steps)
    external_sink = Sink(shape=handler.s_out.shape, buffer=num_steps)

    # Wire the processes
    internal_source.s_out.connect(handler.s_in)
    external_source.s_out.connect(handler.a_in)

    handler.a_out.connect(internal_sink.a_in)
    handler.s_out.connect(external_sink.a_in)

    # Run simulation
    handler.run(
        condition=RunSteps(num_steps=1),
        run_cfg=Loihi2SimCfg()
    )

    # Run simulation
    handler.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()
    )

    a_out_data = internal_sink.data.get().astype(int)
    # a_out_data = np.transpose(internal_sink.data.get().astype(int), axes=(1, 0))
    s_out_data = external_sink.data.get().astype(int)
    handler.stop()

    # print("in", s_inp_data.shape, s_out_data.shape)
    # print("out", a_inp_data.shape, a_out_data.shape)

    for step in range(num_steps):
        assert np.all(s_inp_data[:, step] == s_out_data[agent_id, :, step]), \
            "Output should match the input data."
        assert np.all(a_inp_data[agent_id, :, step] == a_out_data[:, step]), \
            "Output should match the input data."
