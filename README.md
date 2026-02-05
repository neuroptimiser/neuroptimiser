![Status](https://img.shields.io/badge/status-active-success)
[![PyPI Version](https://img.shields.io/pypi/v/neuroptimiser)](https://pypi.org/project/neuroptimiser/)
![GitHub Tag](https://img.shields.io/github/v/tag/neuroptimiser/neuroptimiser)
[![Python Versions](https://img.shields.io/pypi/pyversions/neuroptimiser.svg)](https://pypi.org/project/neuroptimiser/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/neuroptimiser)](https://pypi.org/project/neuroptimiser/)
![License](https://img.shields.io/github/license/neuroptimiser/neuroptimiser)
[![arXiv](https://img.shields.io/badge/arXiv-2507.08320-b31b1b.svg)](https://arxiv.org/abs/2507.08320)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15592900.svg)](https://doi.org/10.5281/zenodo.15592900)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858610.svg)](https://doi.org/10.5281/zenodo.15858610)

# NeurOptimiser

**NeurOptimiser** is a neuromorphic optimisation framework in which metaheuristic search emerges from asynchronous spiking dynamics. It defines optimisation as a decentralised process executed by interconnected Neuromorphic Heuristic Units (NHUs), each embedding a spiking neuron model and a spike-triggered heuristic rule.

This framework enables fully event-driven, low-power optimisation by integrating spiking computation with local heuristic adaptation. It supports multiple neuron models, perturbation operators, and network topologies.

---

## ✨ Key Features

- Modular and extensible architecture using **Intel’s Lava**.
- Supports **linear** and **Izhikevich** neuron dynamics.
- Implements **random**, **fixed**, **directional**, and **Differential Evolution** operators as spike-triggered perturbations.
- Includes asynchronous **neighbourhood management**, **tensor contraction layers**, and **greedy selectors**.
- Compatible with **BBOB (COCO)** suite.
- Designed for **scalability**, **reusability**, and **future deployment** on **Loihi-class neuromorphic hardware**.

---

## 📖 Documentation
For detailed documentation, examples, and API reference, please visit the [Neuroptimiser Documentation](https://neuroptimiser.github.io/).

## 📦 Installation

```bash
pip install -e .
```
Ensure you have Python ≥ 3.10 and the Lava-NC environment configured.

You can also clone the repository and install it. Check the Makefile for additional options.


## 🚀 Example Usage
```python
from neuroptimiser import NeuroOptimiser
import numpy as np

problem_function    = lambda x: np.linalg.norm(x)
problem_bounds      = np.array([[-5.0, 5.0], [-5.0, 5.0]])

optimiser = NeurOptimiser()

optimiser.solve(
    obj_func=problem_function,
    search_space=problem_bounds,
    debug_mode=True,
    num_iterations=1000,
)
```

For more examples, please, visit [Neuroptimiser Usage](https://neuroptimiser.github.io/usage.html)

## 📊 Benchmarking
Neuroptimiser has been validated over the [BBOB suite](https://github.com/numbbo/coco), showing:
* Competitive convergence versus Random Search
* Consistent results across function types and dimensions
* Linear runtime scaling with number of units and problem size

## 🔬 Citation
```bibtex
@misc{neuroptimiser2025,
  author={Cruz-Duarte, Jorge M. and Talbi, El-Ghazali},
  title        = {Neuroptimiser: A neuromorphic optimisation framework},
  year         = {2025},
  url          = {https://github.com/neuroptimiser/neuroptimiser},
  note         = {Version 1.0.X, accessed on 20XX-XX-XX}
}
```

## 🔗 Resources
* 📘 [Documentation](https://neuroptimiser.github.io)
* 📜 [Paper](https://doi.org/10.48550/arXiv.2507.08320)
* 🧠 [Intel Lava-NC](https://github.com/lava-nc/lava)
* 🧪 [COCO Platform](https://github.com/numbbo/coco)

## 🛠️ License
BSD-3-Clause License — [see LICENSE](LICENSE)

## 🧑‍💻 Authors
* [Jorge M. Cruz-Duarte](https://github.com/jcrvz) — University of Lille
* El-Ghazali Talbi — University of Lille
