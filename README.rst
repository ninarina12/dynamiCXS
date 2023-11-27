Data-driven discovery of dynamics from time-resolved coherent scattering
==============================================================================
|docs|

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://dynamicxs.readthedocs.io/en/latest

This repository contains supporting code for the work, `"Data-driven discovery of dynamics from time-resolved coherent scattering" <https://arxiv.org/abs/2311.14196>`_ by Nina Andrejevic, *et al*.

The ``ode.py`` module contains classes for defining and solving systems of ordinary differential equations (ODEs) using ``PyTorch`` tensors and the ``torchdiffeq`` library of ODE solvers for numerical integration. Implementations of the systems reported in this work are provided as examples. The ``cxs.py`` module contains classes for computing coherent speckle patterns from objects defined either on a grid or as point clouds and is also implemented using `PyTorch` tensors.

Under the ``dynamicxs/systems`` directory, we have included Jupyter notebooks to reproduce the results of the three computational case studies and experimental proof-of-concept reported in the work. Corresponding datasets for the experimental example are available `here <https://zenodo.org/doi/10.5281/zenodo.10204976>`_. Please refer to ``requirements.txt`` for python package dependencies. Visualizations of the true and predicted dynamics in real and reciprocal space for these examples are shown below. Please see the manuscript for additional details.

.. figure:: images/kuramoto_results.gif
    :width: 400
**Locally-coupled moments  |**  Learning the coupling kernel governing a two-dimensional lattice of locally-interacting moments evolving in time according to the Kuramoto model.

.. figure:: images/swarm_results.gif
    :width: 400
**Self-organizing particles  |**  Learning clustering dynamics in a collection of interacting particles with an unknown interaction potential.

.. figure:: images/lotka_results.gif
    :width: 400
**Fluctuating source  |**  Learning the dynamics of a periodic, fluctuating source from observations of a random test pattern.

.. figure:: images/traj_results.gif
    :width: 900
**Ptychographic scan  |**  Learning the probe trajectory during a ptychographic scan. Blue markers denote inference within the time window seen during training; pink markers denote inference beyond the time window seen during training.