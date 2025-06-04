import numpy as np
import pytest
import matplotlib.pyplot as plt

from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from neuroptimiser.core.models import TwoDimSpikingCore
from lava.proc.monitor.process import Monitor

rng = np.random.default_rng()
agent_id        = 3
num_agents      = 10
num_neighbours  = 4
num_dimensions  = 7
num_steps       = 10

initial_pos     = rng.uniform(-1, 1, size=(num_dimensions,))

@pytest.fixture
def run_cfg():
    """Fixture to provide the run configuration."""
    return Loihi2SimCfg()

@pytest.fixture
def perturbation_nheuristic():
    """Fixture to initialize the PerturbationNHeuristic process."""
    return TwoDimSpikingCore(
        num_dimensions=num_dimensions,
        init_position=initial_pos,
        num_neighbours=num_neighbours,
        delta=1.5,
        theta=0.6,
        noise_std=(0.0, 0.1),
    )

def test_initialisation(perturbation_nheuristic):
    """Test the initialization of PerturbationNHeuristic."""
    assert perturbation_nheuristic.s_in.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert perturbation_nheuristic.s_out.shape == (num_dimensions,), "Input should be a 1D tensor."

    assert perturbation_nheuristic.p_in.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert perturbation_nheuristic.fp_in.shape == (1,), "Input should be a 1D tensor."

    assert perturbation_nheuristic.g_in.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert perturbation_nheuristic.fg_in.shape == (1,), "Input should be a 1D tensor."

    assert perturbation_nheuristic.xn_in.shape == (num_neighbours, num_dimensions), "Input should be a 2D tensor."
    assert perturbation_nheuristic.fxn_in.shape == (num_neighbours,), "Input should be a 1D tensor."

    assert perturbation_nheuristic.x.shape == (num_dimensions,), "Input should be a 1D tensor."
    assert np.all(perturbation_nheuristic.x.get() == initial_pos), "Initial position should match the input data."
