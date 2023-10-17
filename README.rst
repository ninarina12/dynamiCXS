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

We have also included Jupyter notebooks to reproduce the results of the three computational case studies reported in the work, which are also recommended as a starting point for adapting the code to new dynamical systems.

.. figure:: images/kuramoto_results.gif
    :width: 400
    
.. figure:: images/swarm_results.gif
    :width: 400
    
.. figure:: images/lotka_results.gif
    :width: 400
