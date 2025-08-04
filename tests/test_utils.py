from neuroptimiser.utils import get_2d_sys
import numpy as np
import pytest

@pytest.mark.parametrize("trials", [10, 50, 100])
def test_get_2d_sys(trials):
    for _ in range(trials):
        # Sink (spiral sink): complex eigenvalues, Re(λ) < 0
        A = get_2d_sys(kind="sink")
        tr = np.trace(A)
        det = np.linalg.det(A)
        disc = tr**2 - 4 * det
        assert tr < 0, "Trace should be negative for 'sink' kind"
        assert det > 0, "Determinant should be positive for 'sink' kind"
        assert disc < 0, "Discriminant should be negative for 'sink' (spiral)"

        # Source (spiral source): complex eigenvalues, Re(λ) > 0
        A = get_2d_sys(kind="source")
        tr = np.trace(A)
        det = np.linalg.det(A)
        disc = tr**2 - 4 * det
        assert tr > 0, "Trace should be positive for 'source' kind"
        assert det > 0, "Determinant should be positive for 'source' kind"
        assert disc < 0, "Discriminant should be negative for 'source' (spiral)"

        # Repeller (unstable node): real positive eigenvalues
        A = get_2d_sys(kind="repeller")
        tr = np.trace(A)
        det = np.linalg.det(A)
        disc = tr**2 - 4 * det
        assert tr > 0, "Trace should be positive for 'repeller'"
        assert det > 0, "Determinant should be positive for 'repeller'"
        assert disc > 0, "Discriminant should be positive for 'repeller' (node)"

        # Attractor (stable node): real negative eigenvalues
        A = get_2d_sys(kind="attractor")
        tr = np.trace(A)
        det = np.linalg.det(A)
        disc = tr**2 - 4 * det
        assert tr < 0, "Trace should be negative for 'attractor'"
        assert det > 0 or np.isclose(det, 0, atol=1e-8), "Determinant should be positive for 'attractor'"
        assert disc > 0, "Discriminant should be positive for 'attractor' (node)"

        # Centre: purely imaginary eigenvalues
        A = get_2d_sys(kind="centre")
        tr = np.trace(A)
        det = np.linalg.det(A)
        disc = tr**2 - 4 * det
        assert np.isclose(disc, 0, atol=1e-4), "Discriminant should be zero for 'centre'"
        assert det > 0, "Determinant should be positive for 'centre'"

        # Saddle: real eigenvalues of opposite sign
        A = get_2d_sys(kind="saddle")
        det = np.linalg.det(A)
        disc = np.trace(A)**2 - 4 * det
        assert det < 0, "Determinant should be negative for 'saddle'"
        assert disc > 0, "Discriminant should be positive for 'saddle'"