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

## 🚀 Example Usage
```python
from neuroptimiser import NeuroOptimiser

neuropt = NeuroOptimiser(
    problem=my_bbob_function,
    dimension=10,
    num_units=30,
    neuron_model="izhikevich",
    spike_operator="de_rand",
    spike_condition="l2"
)

neuropt.run(max_steps=1000)
```

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
  note         = {Version 1.0.0, accessed on 20XX-XX-XX}
}
```

## 🔗 Resources
* 📘 [Documentation](https://neuroptimiser.github.io)
* 🧠 [Intel Lava-NC](https://github.com/lava-nc/lava)
* 🧪 [COCO Platform](https://github.com/numbbo/coco)

## 🛠️ License
MIT License — [see LICENSE](LICENSE)

## 🧑‍💻 Authors
* [Jorge M. Cruz-Duarte](https://github.com/jcrvz) — University of Lille
* El-Ghazali Talbi — University of Lille