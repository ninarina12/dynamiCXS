import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import time

import torch
import torch.nn as nn

#from skimage.data import binary_blobs
#from torchdiffeq import odeint, odeint_adjoint
#from utils import cm, props, format_axis


class ODE(nn.Module):
    '''Base class to define, solve, and visualize a system of ODEs.
            
    Parameters
    ----------
    method : str
        Name of ODE solver to use. Default is `dopri5`.

    Attributes
    ----------
    seed : int
        Seed used to set the state of a random number generator.
    L : float
        Length of the real-space simulation box.
    odeint : torchdiffeq.odeint
        Numerical integrator for a system of ODEs given an initial value.
        
    ''' 
    
    def __init__(self, method='dopri5'):      
        super(ODE, self).__init__()
        
        self.seed = 12
        self.L = 2.
        self.method = method    
        self.odeint = odeint
        
        
    def solve(self, t, t0=0):
        '''Load a network from file.
        
        Parameters
        ----------
        t : torch.tensor
            1-dimensional tensor of evaluation times.

        t0 : int or float
            Either an index (int) into t or a time point (float) at which to start (usually to eliminate initial transients).

        Returns
        -------
        
        '''
        
        device = self.y0.device
        ti = time.time()
        with torch.no_grad():
            self.y = self.odeint(self, self.y0, t.to(device), method=self.method)
        tf = time.time()
        print('Elapsed time: {:.2f} s'.format(tf - ti))
        
        if not isinstance(t0, int):
            t0 = np.argmin(np.abs(t - t0))
        self.y = self.y[t0:]
        self.y0.data = self.y[0]
        
        if t0 > 0:
            self.t = t[:-t0]
        else:
            self.t = t
        
        
    def get_colors(self, y, ntype=None, vmin=1e2, vmax=1e5):
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

    
    def plot_frame(self, ax, y, ntype=None, vmin=1e2, vmax=1e5, alpha=0.9, extent=None):
        if ntype == 'mod':
            y = np.mod(y, 2*np.pi)
        cmap, norm = self.get_colors(y, ntype, vmin, vmax)
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
        if ntype == 'mod':
            y = np.mod(y, 2*np.pi)
        cmap, norm = self.get_colors(y, ntype, vmin, vmax)
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
    def __init__(self, N, v, K, method='dopri5', default_type=torch.float64):
        super(Kuramoto, self).__init__(method)
        
        self.N = N
        self.v = v
        self.K = K
        
        kernel = torch.tensor([[[[0,0,1,0,0],
                                 [0,1,1,1,0],
                                 [1,1,0,1,1],
                                 [0,1,1,1,0],
                                 [0,0,1,0,0]]]], dtype=default_type)
        
        self.conv = nn.Conv2d(1, 1, kernel.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)
    
    
    def init_state(self, M=1):
        torch.manual_seed(self.seed)
        self.y0 = nn.Parameter(2*np.pi*torch.rand(M, 1, self.N, self.N).flatten(start_dim=-2), requires_grad=False)
    
    
    def forward(self, t, y):
        y = y.view((-1, 1, self.N, self.N))
        cosy = torch.cos(y)
        siny = torch.sin(y)
        conv_cosy = self.conv(cosy)
        conv_siny = self.conv(siny)
        return self.v + self.K*(cosy*conv_siny - siny*conv_cosy).flatten(start_dim=-2)
    
    
    
class GrayScott(ODE):
    def __init__(self, N, Du, Dv, f, k, f0=None, k0=None, method='dopri5', default_type=torch.float64):
        super(GrayScott, self).__init__(method)
        
        self.N = N
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.f0 = f0
        self.k0 = k0 
        self.L = 2.
        
        h = self.L/N
        kernel = (1./h)**2*torch.tensor([[[[0,1,0],
                                           [1,-4,1],
                                           [0,1,0]]]], dtype=default_type)
        
        self.conv = nn.Conv2d(1, 1, kernel.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)
        
    
    def init_state(self, M=1):
        torch.manual_seed(self.seed)
        u = torch.ones((M, 1, self.N, self.N))
        v = torch.zeros((M, 1, self.N, self.N))
        for i in range(M):
            mask = binary_blobs(self.N, blob_size_fraction=0.2, volume_fraction=0.5,
                                seed=self.seed).reshape(1, self.N, self.N)
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
        y = y.view((-1, 2, self.N, self.N))
        u, v = y.split([1,1], dim=1)
        du = self.Du*self.conv(u) - u*v*v + self.f*(1 - u)
        dv = self.Dv*self.conv(v) + u*v*v - (self.f + self.k)*v
        return torch.cat([du, dv], dim=1).flatten(start_dim=-2)
    
    
    
class LotkaVolterra(ODE):
    def __init__(self, N, R, alpha, beta, gamma, delta, method='dopri5', default_type=torch.float64):
        super(LotkaVolterra, self).__init__(method)
        
        self.N = N
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.L = 2.
        
    
    def init_state(self, M):
        torch.manual_seed(self.seed)
        self.y0 = self.L*torch.rand((M, self.N, 2)) - self.L/2.
        
        
    def forward(self, t, y):
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