import numpy as np

IZHIKEVICH_MODELS_KIND = ["RS", "IB", "CH", "FS", "TC", "TCn", "RZ", "LTS", "random"]

DYN_MODELS_KIND = ["saddle", "attractor", "repeller", "source", "sink"]

ADJ_MAT_OPTIONS = [
    "one-way-ring", "1dr", "ring",
    "two-way-ring", "2dr", "bidirectional-ring",
    "fully-connected", "all", "full",
    "random", "rand",
]

def reset_all_processes(*processes):
    for proc in processes:
        if hasattr(proc, "reset"):
            proc.reset()

def tro2s(x: np.ndarray|float, lb: np.ndarray|float, ub: np.ndarray|float) -> np.ndarray|float:
    return 2 * (x - lb) / (ub - lb) - 1

def trs2o(x: np.ndarray|float, lb: np.ndarray|float, ub: np.ndarray|float) -> np.ndarray|float:
    return (x + 1) / 2 * (ub - lb) + lb

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