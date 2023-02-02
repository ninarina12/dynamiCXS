import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn

from skimage.data import binary_blobs
from torchdiffeq import odeint, odeint_adjoint
from utils import cm, props, format_axis


class ODE(nn.Module):
    r'''Base class to define, solve, and visualize a system of ODEs.
            
    Parameters
    ----------
    method : str
        Name of ODE solver to use. Default is ``dopri5``.
    
    adjoint : bool
        Whether or not to use the adjoint method for backpropagating through ODE solutions. Default is False.
    
    requires_grad: bool
        Whether or not gradients should be computed for the tensors defining the ODE system.

    Attributes
    ----------
    odeint : torchdiffeq.odeint or torchdiffeq.odeint_adjoint
        Numerical integrator for a system of ODEs given an initial value.
        The adjoint method will not be used if ``requires_grad`` is `False`.
        
    ''' 
    
    def __init__(self, method='dopri5', adjoint=False, requires_grad=True):      
        super(ODE, self).__init__()
        
        self.method = method
        self.adjoint = adjoint if requires_grad else False
        self.requires_grad = requires_grad
        self.odeint = odeint_adjoint if self.adjoint else odeint
        
        
    def _color(self, y, ntype=None, vmin=1e2, vmax=1e5):
        if ntype == 'mod':
            norm = plt.Normalize(vmin=0., vmax=2*np.pi)
            cmap = cm.romaO
        elif ntype == 'log':
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = cm.davos
        elif ntype == 'unit':
            norm = plt.Normalize(vmin=0, vmax=1)
            cmap = cm.davos
        else:
            norm = plt.Normalize(vmin=y.min(), vmax=y.max())
            cmap = cm.davos
        return cmap, norm

    
    def _solve(self, t, y0, device='cpu'):
        return self.odeint(self.to(device), y0.to(device), t.to(device), method=self.method)
        
        
    def _solve_no_grad(self, t, y0=None, device='cpu'):
        if self.requires_grad:
            ti = time.time()
            with torch.no_grad():
                y = odeint(self.to(device), y0.to(device), t.to(device), method=self.method)
            tf = time.time()
            print('Elapsed time: {:.2f} s'.format(tf - ti))
            return y
        
        else:
            self.t = t
            ti = time.time()
            with torch.no_grad():
                self.y = odeint(self.to(device), self.y0.to(device), t.to(device), method=self.method)
            tf = time.time()
            print('Elapsed time: {:.2f} s'.format(tf - ti))
        
        
    def solve(self, t, y0=None, device='cpu'):
        r'''Numerically integrate the ODE system and return the solution at times ``t``.
        
        Parameters
        ----------
        t : torch.tensor
            1-dimensional tensor of evaluation times.

        y0 : torch.tensor of shape (M, ..., D)
            Initial state of the system for ``M`` initial conditions. ``D`` denotes the flattened system size.
            
        device : str
            The name of the device on which the computation will be performed (e.g. ``cpu`` or ``cuda``).

        Returns
        -------
        y : torch.tensor of shape (T, M, ..., D)
            Solution evaluated at ``T`` time points for ``M`` initial conditions. ``D`` denotes the flattened system size.
            
        '''
        
        if self.requires_grad:
            if self.training:
                return self._solve(t, y0, device)
            else:
                return self._solve_no_grad(t, y0, device)
        else:
            self._solve_no_grad(t)
            
            
    def trim(self, t0=0):
        r'''Trim the solution at early time points to exclude initial transients.
        
        Parameters
        ----------
        t0 : int
            The first index into ``t`` at which to return the solution in order to exclude initial transients.
            The initial state will be set to the solution at this point and initial time set to 0 at this point.
            
        '''
        
        if t0 > 0:
            self.t = self.t[:-t0]
            self.y0.data = self.y[t0]
            self.y = self.y[t0:]
            
        
    def plot_frame(self, ax, y, ntype=None, vmin=1e2, vmax=1e5, alpha=0.9, extent=None):
        r'''Plot a single frame of an ODE solution.
        
        Parameters
        ----------
        ax : matplotlib.axes
            Axis object on which to display the solution. 
        
        y : torch.tensor of shape (N,N)
            Solution of the system to plot. The input should be reshaped to the dimensions of the simulation box.
        
        ntype : str
            Type of normalization to apply when displaying the image. The options are:
            - 'mod' : Linear normalization modulo 2:math:\pi
            - 'log' : Logarithmic normalization scale
            - 'unit' : Linear normalization between 0 and 1
            - 'None' : Linear normalization between the minimum and maximum values of ``y``
            The default is ``None``.     
        
        vmin : float, optional
            Minimum value of logarithmic scale when ``ntype`` is ``log``.    
        
        vmax : float, optional
            Maximum value of logarithmic scale when ``ntype`` is ``log``.
        
        alpha : float between 0 and 1, optional
            Opacity of the image.
        
        extent : list or tuple of floats (left, right, bottom, top), optional
            Coordinates of the bounding box of the image.
        '''
        
        if ntype == 'mod':
            y = np.mod(y, 2*np.pi)
        cmap, norm = self._color(y, ntype, vmin, vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        colors = cmap(norm(y))
        colors[...,-1] = alpha
        
        try: len(extent)
        except: ax.imshow(colors, extent=(-self.L/2., self.L/2., -self.L/2., self.L/2.), origin='lower') 
        else: ax.imshow(colors, extent=extent, origin='lower')
            
        ax.set_xlim(-self.L/2., self.L/2.)
        ax.set_ylim(-self.L/2., self.L/2.)
        ax.axis('off')
            
        
    def plot_series(self, y, ntype=None, vmin=1e2, vmax=1e5, clabel=None):
        r'''Plot a time series of frames of an ODE solution.
        
        Parameters
        ----------
        y : torch.tensor of shape (T,N,N)
            Solution of the system to plot. ``T`` denotes the time points at which the system was evaluated.
            The input should be reshaped so that the last two dimensions match the dimensions of the simulation box.
        
        ntype : str
            Type of normalization to apply when displaying the image. The options are:
            - 'mod' : Linear normalization modulo 2\pi
            - 'log' : Logarithmic normalization scale
            - 'unit' : Linear normalization between 0 and 1
            - 'None' : Linear normalization between the minimum and maximum values of ``y``
            The default is ``None``.     
        
        vmin : float, optional
            Minimum value of logarithmic scale when ``ntype`` is ``log``.    
        
        vmax : float, optional
            Maximum value of logarithmic scale when ``ntype`` is ``log``.
        
        clabel : str, optional
            Text used to label the colorbar.
        '''
        
        if ntype == 'mod':
            y = np.mod(y, 2*np.pi)
        cmap, norm = self._color(y, ntype, vmin, vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        n = 6
        fig, ax = plt.subplots(1, n + 1, figsize=(3*n,3), gridspec_kw={'width_ratios': n*[1] + [0.07]})
        step = y.shape[0]//n
        for i in range(n):
            colors = cmap(norm(y[i*step]))
            colors[..., -1] = 0.9
            ax[i].imshow(colors, extent=(-1,1,-1,1), origin='lower')
            ax[i].set_xlim(-1,1)
            ax[i].set_ylim(-1,1)
            ax[i].axis('off')
                
        plt.colorbar(sm, cax=ax[-1])
        format_axis(ax[-1], props, xlabel='', ylabel=clabel, ybins=4)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1)
        return fig
    
    
    
class Kuramoto(ODE):
    r'''Class to define, initialize, solve, and visualize the Kuramoto model of coupled oscillators.

    Parameters
    ----------
    args : dict
        Dictionary of parameters defining the ODE system:
        - N (int) - Dimension of the simulation box (``N x N``)
        - L (float) - Length of the real-space simulation box (``L x L``)
        - v (float) - Intrinsic frequency of the oscillators
        - K (float) - Coupling strength

    method : str
        Name of ODE solver to use. Default is ``dopri5``.
        
    default_type: type
        Set the default ``torch.Tensor`` type.

    Attributes
    ----------
    conv : torch.nn.Conv2d
        Convolution operator coupling neighboring oscillators.

    '''
    
    def __init__(self, args, method='dopri5', default_type=torch.float64):   
        super(Kuramoto, self).__init__(method, adjoint=False, requires_grad=False)
        
        default_args = {'N': 100,
                        'L': 2.,
                        'v': 0.5,
                        'K': 0.2
                       }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)
        
        kernel = torch.tensor([[[[0,0,1,0,0],
                                 [0,1,1,1,0],
                                 [1,1,0,1,1],
                                 [0,1,1,1,0],
                                 [0,0,1,0,0]]]], dtype=default_type)
        
        self.conv = nn.Conv2d(1, 1, kernel.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)
    
    
    def init_state(self, M=1, seed=12):
        r'''Randomly generate the initial state(s) of the ODE system.
            
        Parameters
        ----------
        M : int
            Number of initial conditions to generate. Default is 1.
        
        seed : int, optional
            Default seed used to set the state of a random number generator. Default is 12.
                
        Attributes
        ----------
        y0 : torch.tensor of shape (M, ..., D)
            Initial state of the system for ``M`` initial conditions. ``D`` denotes the flattened system size.
                
        '''
        
        torch.manual_seed(seed)
        self.y0 = nn.Parameter(2*np.pi*torch.rand(M, 1, self.N, self.N).flatten(start_dim=-2), requires_grad=False)
    
    
    def forward(self, t, y):
        r'''Evaluate the ODE system at a specified time ``t`` and state ``y``.
            
        Parameters
        ----------
        t : torch.tensor
            1-dimensional tensor of the evaluation time point.
        
        y : torch.tensor of shape (M, ..., D)
            State of the system for ``M`` initial conditions. ``D`` denotes the flattened system size.
                
        Returns
        -------
        dy/dt : torch.tensor of shape (M, ..., D)
            Derivative of the system.
                
        '''
        
        y = y.view((-1, 1, self.N, self.N))
        cosy = torch.cos(y)
        siny = torch.sin(y)
        conv_cosy = self.conv(cosy)
        conv_siny = self.conv(siny)
        return self.v + self.K*(cosy*conv_siny - siny*conv_cosy).flatten(start_dim=-2)
    
    
    
class GrayScott(ODE):
    r'''Class to define, initialize, solve, and visualize the Gray-Scott model of a reaction diffusion system.
    The system consists of components U and V with concentrations `u` and `v`, respectively, which react according to:
    .. math::
        :nowrap:

        \begin{gather*}
        U + 2V & \rightarrow 3V \\
        V & \rightarrow P
        \end{gather*}

    Parameters
    ----------
    args : dict
        Dictionary of parameters defining the ODE system:
        - N (int) - Dimension of the simulation box (``N x N``)
        - L (float) - Length of the real-space simulation box (``L x L``)
        - Du (float) - Diffusion constant of component U
        - Dv (float) - Diffusion constant of component V
        - f (float) - Inflow rate of U (i.e. feed rate)
        - k (float) - Depletion rate of V (i.e. kill rate)
        - f0 (float) - Initial inflow rate of U
        - k0 (float) - Initial depletion rate of V
        Note that ``f0`` and ``k0`` are used only when the desired initial state is an equilibrium solution of a Gray-Scott system. In that case, the system will first be solved from a random initial state using ``f0`` and ``k0``, then solved from the final (equilibrium) state using ``f`` and ``k``.

    method : str
        Name of ODE solver to use. Default is ``dopri5``.
        
    default_type: type
        Set the default ``torch.Tensor`` type.

    Attributes
    ----------
    conv : torch.nn.Conv2d
        Convolution operator used to compute the discrete Laplacian.

    '''
    
    def __init__(self, args, method='dopri5', default_type=torch.float64):
        super(GrayScott, self).__init__(method, adjoint=False, requires_grad=False)
        
        default_args = {'N': 100,
                        'L': 2.,
                        'Du': 1e-5,
                        'Dv': 5e-6,
                        'f': 0.040,
                        'k': 0.060,
                        'f0': None,
                        'k0': None
                       }
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)
        
        h = self.L/self.N
        kernel = (1./h)**2*torch.tensor([[[[0,1,0],
                                           [1,-4,1],
                                           [0,1,0]]]], dtype=default_type)
        
        self.conv = nn.Conv2d(1, 1, kernel.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)
        
    
    def init_state(self, M=1, seed=12):
        r'''Randomly generate the initial state(s) of the ODE system.
            
        Parameters
        ----------
        M : int
            Number of initial conditions to generate. Default is 1.
        
        seed : int, optional
            Default seed used to set the state of a random number generator. Default is 12.
                
        Attributes
        ----------
        y0 : torch.tensor of shape (M, ..., D)
            Initial state of the system for ``M`` initial conditions. ``D`` denotes the flattened system size.
                
        '''
        
        torch.manual_seed(seed)
        u = torch.ones((M, 1, self.N, self.N))
        v = torch.zeros((M, 1, self.N, self.N))
        for i in range(M):
            mask = binary_blobs(self.N, blob_size_fraction=0.2, volume_fraction=0.5,
                                seed=seed).reshape(1, self.N, self.N)
            u[i,0][mask] = 0.5
            v[i,0][mask] = 0.25 
        self.y0 = nn.Parameter(torch.cat([u, v], dim=1).flatten(start_dim=-2), requires_grad=False)
    
    
    def init_solve(self, t):
        device = self.y0.device
        f, k = self.f, self.k
        self.f, self.k = self.f0, self.k0
        ti = time.time()
        with torch.no_grad():
            self.y0.data = self.odeint(self, self.y0, t.to(device), method=self.method)[-1]
        tf = time.time()
        print('Elapsed time: {:.2f} s'.format(tf - ti))
        self.f, self.k = f, k
        
        
    def forward(self, t, y):
        r'''Evaluate the ODE system at a specified time ``t`` and state ``y``.
            
        Parameters
        ----------
        t : torch.tensor
            1-dimensional tensor of the evaluation time point.
        
        y : torch.tensor of shape (M, ..., D)
            State of the system for ``M`` initial conditions. ``D`` denotes the flattened system size.
                
        Returns
        -------
        dy/dt : torch.tensor of shape (M, ..., D)
            Derivative of the system.
                
        '''
        y = y.view((-1, 2, self.N, self.N))
        u, v = y.split([1,1], dim=1)
        du = self.Du*self.conv(u) - u*v*v + self.f*(1 - u)
        dv = self.Dv*self.conv(v) + u*v*v - (self.f + self.k)*v
        return torch.cat([du, dv], dim=1).flatten(start_dim=-2)
    
    
    
class LotkaVolterra(ODE):
    def __init__(self, N, R, alpha, beta, gamma, delta, method='dopri5', adjoint=False, default_type=torch.float64):
        super(LotkaVolterra, self).__init__(method, adjoint, requires_grad=False)
        
        self.N = N
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.L = 2.
        
    
    def init_state(self, M=1, seed=12):
        torch.manual_seed(seed)
        self.y0 = self.L*torch.rand((M, self.N, 2)) - self.L/2.
        
        
    def forward(self, t, y):
        r'''Evaluate the ODE system at a specified time ``t`` and state ``y``.
            
        Parameters
        ----------
        t : torch.tensor
            1-dimensional tensor of the evaluation time point.
        
        y : torch.tensor of shape (M, ..., N, 2)
            State of the system for ``M`` initial conditions.
                
        Returns
        -------
        dy/dt : torch.tensor of shape (M, ..., N, 2)
            Derivative of the system.
                
        '''
        x0 = y[...,[0]].mean(dim=-2, keepdims=True) + 1.
        y0 = y[...,[1]].mean(dim=-2, keepdims=True) + 1.
        dx = self.alpha*x0 - self.beta*x0*y0
        dy = self.delta*x0*y0 - self.gamma*y0
        dx = torch.tile(dx, (1,self.N,1))
        dy = torch.tile(dy, (1,self.N,1))
        return torch.cat([dx, dy], dim=-1)
    
    
    def plot_frame(self, ax, y):
        circles = [plt.Circle((xi,yi), radius=self.R) for xi,yi in y]
        c = mpl.collections.PatchCollection(circles, lw=0, color='#527C9C')
        ax.add_collection(c)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-self.L, self.L])
        ax.set_ylim([-self.L, self.L])
    
    
    def plot_series(self, y):
        n = 6
        fig, ax = plt.subplots(1, n, figsize=(3*n,3), sharey=True)
        step = y.shape[0]//n
        for i in range(n):
            circles = [plt.Circle((xi,yi), radius=self.R) for xi,yi in y[0]]
            c = mpl.collections.PatchCollection(circles, lw=0, color='#D0D0D0', )
            ax[i].add_collection(c)

            circles = [plt.Circle((xi,yi), radius=self.R) for xi,yi in y[i*step]]
            c = mpl.collections.PatchCollection(circles, lw=0, color='#527C9C')
            ax[i].add_collection(c)

            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_xlim([-self.L, self.L])
            ax[i].set_ylim([-self.L, self.L])
                
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1)
        return fig