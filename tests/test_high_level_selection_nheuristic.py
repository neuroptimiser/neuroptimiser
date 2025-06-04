import numpy as np
import pytest

from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import HighLevelSelection

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
def selection_nheuristic():
    """Fixture to initialize the PerturbationNHeuristic process."""
    return HighLevelSelection(
        num_dimensions=num_dimensions, num_agents=num_agents
    )

def test_initialisation(selection_nheuristic):
    """Test the initialization of PerturbationNHeuristic."""
    assert selection_nheuristic.p_in.shape == (num_agents, num_dimensions), "Input should be a 1D tensor."
    assert selection_nheuristic.fp_in.shape == (num_agents,), "Input should be a 1D tensor."
    assert selection_nheuristic.g_out.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert selection_nheuristic.fg_out.shape == (1,), "Input should be a 0D tensor."

@pytest.mark.parametrize("num_steps, num_agents, num_dimensions", [
    (10, 5, 7),
    (20, 15, 10),
    (30, 30, 12),
    (1000, 50, 100),
    # (100, 30, 20, 200),
])
def test_with_different_input_data(num_steps, num_agents, num_dimensions):
    """Test the process with different input data."""
    # Define the input
    x_inp_data = rng.uniform(-1, 1, size=(num_agents, num_dimensions, num_steps))
    y_inp_data = rng.uniform(-1, 1, size=(num_agents, num_steps))

    # Create the processes
    source_p = Source(data=x_inp_data)
    source_fp = Source(data=y_inp_data)

    selector = HighLevelSelection(
        num_dimensions=num_dimensions, num_agents=num_agents
    )

    sink_g = Sink(shape=selector.g_out.shape, buffer=num_steps)
    sink_fg = Sink(shape=selector.fg_out.shape, buffer=num_steps)

    # Wire the processes
    source_p.s_out.connect(selector.p_in)
    source_fp.s_out.connect(selector.fp_in)

    selector.g_out.connect(sink_g.a_in)
    selector.fg_out.connect(sink_fg.a_in)

    print("Running the simulation")

    # Run simulation
    selector.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()
    )

    g_out_data = sink_g.data.get().astype(float)
    fg_out_data = sink_fg.data.get().astype(float)
    selector.stop()

    expected_g  = None
    expected_fg = None
    for step in range(num_steps):
        p_val   = x_inp_data[:, :, step]
        fp_val  = y_inp_data[:, step]

        best_candidate = np.argmin(fp_val)

        new_g   = p_val[best_candidate,:]
        new_fg  = fp_val[best_candidate]

        if step == 0:
            expected_g = new_g
            expected_fg = new_fg
        else:
            if new_fg < expected_fg:
                expected_g = new_g
                expected_fg = new_fg

        assert np.allclose(g_out_data[:, step], expected_g,
                           atol=1e-6, rtol=1e-6), "Output should match the input data."
        assert np.allclose(fg_out_data[:, step], expected_fg,
                           atol=1e-6, rtol=1e-6), "Output should match the input data."

