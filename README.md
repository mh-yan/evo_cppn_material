# Populating cellular metamaterials on the extrema of attainable elasticity through neuroevolution

## Project Overview
This project is the code of the paper "Populating cellular metamaterials on the extrema of attainable elasticity through neuroevolution" 
## Paper Information
**Title**: CMAME 25 - CPPN  
**Authors**: Maohua Yan , Ruicheng Wang , Ke Liu*
**Abstract**: The trade-offs between different mechanical properties of materials pose fundamental challenges
in engineering material design, such as balancing stiffness versus toughness, weight versus
energy-absorbing capacity, and among the various elastic coefficients. Although gradient-based
topology optimization approaches have been effective in finding specific designs and properties,
they are not efficient tools for surveying the vast design space of metamaterials, and thus
unable to reveal the attainable bound of interdependent material properties. Other common
methods, such as parametric design or data-driven approaches, are limited by either the lack of
diversity in geometry or the difficulty to extrapolate from known data, respectively. In this work,
we formulate the simultaneous exploration of multiple competing material properties as a multi-
objective optimization (MOO) problem and employ a neuroevolution algorithm to efficiently
solve it. The Compositional Pattern-Producing Networks (CPPNs) is used as the generative model
for unit cell designs, which provide very compact yet lossless encoding of geometry. A modified
Neuroevolution of Augmenting Topologies (NEAT) algorithm is employed to evolve the CPPNs
such that they create metamaterial designs on the Pareto front of the MOO problem, revealing
empirical bounds of different combinations of elastic properties. Looking ahead, our method
serves as a universal framework for the computational discovery of diverse metamaterials across a
range of fields, including robotics, biomedicine, thermal engineering, and photonics.  

## Dependencies
To run this project, you need the following environment and dependencies:
- Python 3.8
- neat
- fenics
- dolfin

## Installation Guide
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/CMAME_25_CPPN.git
   cd CMAME_25_CPPN
   ```
2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Run the main script:
  ```bash
  python main.py
  ```
- Adjust configuration parameters to fit different experimental needs.

## Contribution
If you wish to contribute or improve this project, please submit a Pull Request or contact the authors for discussion.

## License
This project is released under the MIT license.

