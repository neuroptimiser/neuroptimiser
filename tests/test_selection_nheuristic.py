import numpy as np
import pytest

from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import Selector

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
    return Selector(
        num_dimensions=num_dimensions,
        function=lambda x: np.linalg.norm(x),
    )

def test_initialisation(selection_nheuristic):
    """Test the initialization of PerturbationNHeuristic."""
    assert selection_nheuristic.x_in.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert selection_nheuristic.p_out.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert selection_nheuristic.fp_out.shape == (1,), "Input should be a 0D tensor."

@pytest.mark.parametrize("num_steps, num_dimensions", [
    (10, 7),
    (20, 10),
    (30, 12),
    (1000, 100),
    # (100, 30, 20, 200),
])
def test_with_different_input_data(num_steps, num_dimensions):
    """Test the process with different input data."""
    # Define the input
    x_inp_data  = rng.uniform(-1, 1, size=(num_dimensions, num_steps))

    function    = lambda x: np.linalg.norm(x)
    y_out_data  = [function(x_inp_data[:, step]) for step in range(num_steps)]

    # Create the processes
    source_x    = Source(data=x_inp_data)

    selector    = Selector(num_dimensions=num_dimensions, function=function)

    sink_x      = Sink(shape=selector.p_out.shape, buffer=num_steps)
    sink_fx     = Sink(shape=selector.fp_out.shape, buffer=num_steps)

    # Wire the processes
    source_x.s_out.connect(selector.x_in)

    selector.p_out.connect(sink_x.a_in)
    selector.fp_out.connect(sink_fx.a_in)

    print("Running the simulation")

    # Run simulation
    selector.run(
        condition=RunSteps(num_steps=num_steps),
        run_cfg=Loihi2SimCfg()
    )

    p_out_data = sink_x.data.get().astype(float)
    fp_out_data = sink_fx.data.get().astype(float)
    selector.stop()

    best_y = np.minimum.accumulate(y_out_data)

    best_x = []
    current_best_idx = 0
    for step in range(num_steps):
        if y_out_data[step] < y_out_data[current_best_idx]:
            current_best_idx = step
        best_x.append(x_inp_data[:, current_best_idx])
    best_x = np.stack(best_x, axis=1)

    assert np.allclose(p_out_data, best_x, atol=1e-6), "Best-so-far positions mismatch."
    assert np.allclose(fp_out_data[0], best_y, atol=1e-6), "Best-so-far fitness mismatch."

