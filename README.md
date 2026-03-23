# Populating Cellular Metamaterials via Neuroevolution

<p align="center">
  <em>Companion code repository for the paper<br>
  "Populating cellular metamaterials on the extrema of attainable elasticity through neuroevolution"</em>
</p>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/abs/pii/S0045782525002221"><strong>Paper</strong></a>
  ·
  <a href="https://doi.org/10.1016/j.cma.2025.117950"><strong>DOI</strong></a>
  ·
  <a href="https://doi.org/10.57760/sciencedb.22416"><strong>Database</strong></a>
  ·
  <a href="LICENSE"><strong>MIT License</strong></a>
</p>

<p align="center">
  Maohua Yan, Ruicheng Wang, Ke Liu<br>
  <em>Computer Methods in Applied Mechanics and Engineering</em>, 438 (2025), 117950
</p>

<p align="center">
  <img width="868" height="444" alt="Representative evolved metamaterial topologies" src="https://github.com/user-attachments/assets/6c2c3015-98ed-4f55-ad59-85673d55aaad" />
</p>

## Overview

This repository contains the research code used to generate 2D cellular metamaterial unit cells with **CPPNs (Compositional Pattern-Producing Networks)** and a **modified NEAT-based multi-objective evolutionary algorithm**. The workflow couples geometry generation, mesh construction, constraint handling, and periodic homogenization to search for designs near the empirical extrema of attainable elasticity.

The primary research path in this repository is centered on [`main.py`](main.py), [`config.ini`](config.ini), [`tools/`](tools), and the local modified [`neat/`](neat) implementation.

## Method at a Glance

```mermaid
flowchart LR
    A[Initialize population] --> B[Decode genome to CPPN]
    B --> C[Sample structured point cloud]
    C --> D[Threshold and extract geometry]
    D --> E[Generate mesh]
    E --> F[Evaluate effective elastic properties]
    F --> G[Assign objectives and constraints]
    G --> H[Evolve next generation]
```

At the repository level, the main workflow is:

1. Generate a structured point cloud for the selected symmetry class.
2. Evaluate a CPPN on that point cloud to produce a scalar field.
3. Threshold and interpolate the field into material and void regions.
4. Build a mesh and check geometric validity or connectivity constraints.
5. Solve the periodic homogenization problem to obtain effective properties.
6. Feed the evaluated objectives back into the evolutionary loop.

## Repository Structure

```text
.
├── main.py               # Main entry point for metamaterial evolution
├── config.ini            # NEAT and experiment configuration
├── multi_task.py         # Batch task definitions across symmetries and trade-offs
├── gen_pcd.py            # Point-cloud generation utilities
├── tools/                # Geometry, meshing, homogenization, constraints, utilities
├── neat/                 # Local modified neat-python implementation
├── test/                 # Exploratory and utility test scripts
├── test.py               # Standalone test/demo entry
└── design_framework.zip  # Archived framework snapshot
```

## Main Components

### Core workflow

- `main.py`: experiment orchestration and genome evaluation loop
- `gen_pcd.py`: structured point-cloud generation for different symmetry assumptions
- `tools/shape.py`: contour extraction, triangulation, and topology construction
- `tools/read_mesh.py`: conversion from generated geometry to FEniCS-compatible meshes
- `tools/period.py`: periodic homogenization and effective-property evaluation
- `tools/handle_constraints.py`: connectivity and validity checks

### Evolution engine

- `neat/`: local fork of `neat-python`
- `neat/population.py`: modified population flow for this research setup
- `neat/spea2.py`: archive and Pareto-related utilities
- `neat/ns.py`: novelty-related support code

## Requirements

The main metamaterial pipeline was developed around the following stack:

- Python 3.8
- NumPy
- SciPy
- Matplotlib
- FEniCS / Dolfin
- `cvxopt`

Because this is a research codebase, environment setup is somewhat workflow-specific. For the main experiments, the critical requirement is a working **FEniCS/Dolfin** environment.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/mh-yan/evo_cppn_material.git
cd evo_cppn_material
```

Install the dependencies required by your target workflow, then run the main experiment with:

```bash
python main.py
```

This uses the settings in `config.ini` and the default trade-off defined in `main.py`.

For customization:

- edit `config.ini` to change population size, number of generations, density, and symmetry type
- inspect `multi_task.py` for batch experiment definitions
- focus on `main.py`, `tools/`, and `neat/` if you are reproducing the paper pipeline

## Data and Reproducibility

The paper-related metamaterial database is publicly available at:

- [Science Data Bank: Metamaterial Database](https://doi.org/10.57760/sciencedb.22416)

This repository is best understood as a companion research codebase rather than a polished software package. The main reference implementation is the metamaterial pipeline centered on `main.py`, `config.ini`, `tools/`, and `neat/`.

## Citation

If you use this repository in academic work, please cite:

> Maohua Yan, Ruicheng Wang, Ke Liu. Populating cellular metamaterials on the extrema of attainable elasticity through neuroevolution. *Computer Methods in Applied Mechanics and Engineering*, 438 (2025), 117950. https://doi.org/10.1016/j.cma.2025.117950

```bibtex
@article{yan2025populating,
  title   = {Populating cellular metamaterials on the extrema of attainable elasticity through neuroevolution},
  author  = {Yan, Maohua and Wang, Ruicheng and Liu, Ke},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {438},
  pages   = {117950},
  year    = {2025},
  doi     = {10.1016/j.cma.2025.117950},
  url     = {https://doi.org/10.1016/j.cma.2025.117950}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
