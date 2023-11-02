Data-driven discovery of dynamics from time-resolved coherent X-ray scattering
==============================================================================
|docs|

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://dynamicxs.readthedocs.io/en/latest

This repository contains supporting code for the work, "Data-driven discovery of dynamics from time-resolved coherent X-ray scattering" by Nina Andrejevic, *et al*.

The ``ode.py`` module contains classes for defining and solving systems of ordinary differential equations (ODEs) using ``PyTorch`` tensors and the ``torchdiffeq`` library of ODE solvers for numerical integration. Implementations of the systems reported in this work are provided as examples.

The ``cxs.py`` module contains classes for computing coherent speckle patterns from objects defined either on a grid or as point clouds and is also implemented using `PyTorch` tensors.

We have also included Jupyter notebooks to reproduce the results of the three computational case studies reported in the work, which are also recommended as a starting point for adapting the code to new dynamical systems. Please refer to ``requirements.txt`` for python package dependencies. Visualizations of the simulated and predicted dynamics in real and reciprocal space for the three computational examples are shown below. Please see the manuscript for additional details.

.. figure:: images/kuramoto_results.gif
    :width: 400
**Figure 1: Locally-coupled moments** Learning the coupling kernel governing a two-dimensional lattice of locally-interacting moments evolving in time according to the Kuramoto model.

.. figure:: images/swarm_results.gif
    :width: 400
**Figure 2: Self-organizing particles** Learning clustering dynamics in a collection of interacting particles with an unknown interaction potential.

.. figure:: images/lotka_results.gif
    :width: 400
**Figure 3: Fluctuating source** Learning the dynamics of a periodic, fluctuating source from observations of a random test pattern.