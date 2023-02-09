import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn

from torch.fft import fftn, ifftn, fftshift, ifftshift
from utils import cm, props, format_axis


class CXS(nn.Module):
    r"""Base class to compute the coherent X-ray scattering pattern of an object.

    Parameters
    ----------
    n : int
        Dimension of reciprocal-space pattern ``n x n`` for ``dq = 1`` (default).
        When ``dq < 1``, the new dimension ``n`` will be recomputed and saved as an attribute.

    L : float
        Length of the real-space simulation box ``(L x L)``.
        
    dq : float
        Sampling in reciprocal space, `i.e.` the fraction of :math:`2 \pi/L` at which to sample.
        Default is ``dq = 1``, corresponding to a spacing of :math:`2 \pi/L`.
        
    f_probe : float
        Standard deviation of the Gaussian probe intensity as a fraction of ``L``.
        
    f_mask : 
        Radius of center beam stop as a fraction of ``n``.

    Attributes
    ----------
    cmap : ``matplotlib.colors.Colormap``
        Default colormap for plotting.
        
    gmap : ``matplotlib.colors.Colormap``
        ``cmap`` with an opacity gradient.

    q : ``torch.tensor``
        1-D tensor of reciprocal space coordinates determined by the dimension ``n`` and sampling ``dq``.
        
    n : int
        Recomputed dimension of reciprocal-space pattern ``n x n`` for the input sampling ``dq``.
        
    Q : ``torch.tensor``
        2-D tensor of shape ``(n x n, 2)`` containing the reciprocal space coordinate grid.
        
    mask : ``torch.tensor``
        Boolean tensor storing the masked detector pixels.
        
    probe : ``torch.nn.Conv2d``
        Convolution operator for convolving the reciprocal space pattern with the probe.
        
    """
    def __init__(self, n, L=1., dq=1., f_probe=None, f_mask=None):
        super(CXS, self).__init__()
        # TO DO:
        # - Passing in a real space probe as an argument, default to gaussian if not passed
        # - Handle complex probe (manually split into convolutions with real and imag parts of probe fft)
        
        self.L = L
        self.dq = dq
        self.f_probe = f_probe
        self.cmap = plt.cm.bone
        self.gmap = self.cmap.copy()
        self.gmap(0);
        self.gmap._lut[:self.gmap.N,3] = np.linspace(0, 0.7, self.gmap.N)[::-1]
        
        # Reciprocal-space coordinates
        self.q = 2*np.pi/L*torch.arange(-n/2., n/2., dq)
        self.n = len(self.q)
        Qx, Qy = torch.meshgrid(self.q, self.q, indexing='xy')
        self.Q = nn.Parameter(torch.stack((Qx.flatten(), Qy.flatten()), dim=1), requires_grad=False)
        
        if f_mask:
            self.mask = nn.Parameter((Qx**2 + Qy**2 > (f_mask*2*np.pi/L*n)**2).flatten(), requires_grad=False)
        else:
            self.mask = nn.Parameter(torch.tensor(1.), requires_grad=False)
        
        # Gaussian profile
        if f_probe:
            tol = 1e-16
            r_probe = f_probe*L
            probe = torch.exp(-(self.Q**2).sum(dim=1)/(2./r_probe**2)).view((1,1,self.n,self.n))
            k = 2*int(np.ceil(np.sqrt(-2*np.log(tol)/(r_probe**2))/(self.q.max() - self.q.min())*self.n)//2)
            k += 1 - self.n%2
            probe = self.center_crop(probe, k)
            self.probe = nn.Conv2d(1, 1, n, bias=False, padding='same', padding_mode='zeros')
            self.probe.weight = nn.Parameter(probe/probe.max(), requires_grad=False)
            
        else:
            self.probe = nn.Conv2d(1, 1, 1, bias=False, padding='same', padding_mode='zeros')
            self.probe.weight = nn.Parameter(torch.tensor(1.).view(1,1,1,1), requires_grad=False)
    
    
    def shapes(self):
        r"""Print detector and probe shapes.
        """
        print('Detector: {:g} x {:g} \t'.format(self.n,self.n), end='')
        print('Probe: {:g} x {:g}'.format(*self.probe.weight.shape[-2:]))
            
            
    def center_crop(self, y, k):
        r"""Crop ``y`` down to its central ``k x k`` pixels.
            
        Parameters
        ----------
        y : ``torch.tensor``
            Tensor to crop.
        
        k : int
            Target dimension, ``k x k``.
                
        Returns
        -------
        y : ``torch.tensor`` of shape ``(...,k,k)``
            Cropped tensor.
                
        """
        l = y.shape[-1]
        s = 1 - k%2
        return y[..., l//2 - k//2 + s:l//2 + k//2 + k%2 + s, l//2 - k//2 + s:l//2 + k//2 + k%2 + s]
    
    
    def plot_probe(self, ax):
        r"""Plot the real and, optionally, reciprocal space views of the probe.
            
        Parameters
        ----------
        ax : ``matplotlib.axes``
            Axis or list of axis objects on which to display the image(s).
                
        Returns
        -------
        sm : list
            List of scalar mappables storing the color scales of the two plots.
                
        """
        f_probe = self.f_probe if self.f_probe else 1.     
        _x = torch.arange(-self.L/2., self.L/2., self.L/100.)
        _X, _Y = np.meshgrid(_x, _x, indexing='xy')
        _r = np.stack((_X, _Y), axis=0)
        
        probe = np.exp((-(_r)**2).sum(axis=0)/(2.*(f_probe*self.L)**2))
        probe /= probe.max()
        
        circles = [plt.Circle((0,0), radius=k*f_probe*self.L) for k in range(1,2)]
        c = mpl.collections.PatchCollection(circles, fc='none', ec='white', ls='dashed')
            
        sm = []
        try: len(ax)
        except:
            sm.append(ax.imshow(probe, cmap=self.gmap, vmin=0., vmax=1.,
                                extent=(-self.L/2.,self.L/2.,-self.L/2.,self.L/2.), zorder=1))
            ax.add_collection(c)
            ax.set_xlim([-self.L/2., self.L/2.])
            ax.set_ylim([-self.L/2., self.L/2.])
            
        else:
            sm.append(ax[0].imshow(probe, cmap=self.gmap, vmin=0., vmax=1.,
                                   extent=(-self.L/2.,self.L/2.,-self.L/2.,self.L/2.), zorder=1))
            ax[0].add_collection(c)
            ax[0].set_xlim([-self.L/2., self.L/2.])
            ax[0].set_ylim([-self.L/2., self.L/2.])

            sm.append(ax[1].imshow(self.probe.weight[0,0].cpu(), origin='lower', cmap=self.cmap, vmin=0, vmax=1))
            ax[1].axis('off')

            k = self.probe.weight.shape[-1]
            ax[1].text(0.9, 0.85, str(k) + r'$\times$' + str(k), color='white', ha='right', va='center',
                       transform=ax[1].transAxes, fontproperties=props)
        return sm
    
    
    def plot_example(self, ode, y, ntype=None, vmin=None, vmax=None):
        r"""Plot images visualizing the scattering conditions of a given example.
            
        Parameters
        ----------
        ode : ``ode``
            ODE object of the system.
        
        y : ``torch.tensor``
            Solution of the system to plot.
            
        ntype : str
            Type of normalization to apply when displaying the image. The options are:
            
            - ``none`` -- No normalization; point cloud data
            - ``mod`` -- Linear normalization modulo :math:`2 \pi`
            - ``log`` -- Logarithmic normalization scale
            - ``unit`` -- Linear normalization between 0 and 1
            - ``None`` -- Linear normalization between the min. and max. values of ``y``
            
            The default is ``None``.     
        
        vmin : float, optional
            Minimum normalization value when ``ntype`` is ``log``.    
        
        vmax : float, optional
            Maximum normalization value when ``ntype`` is ``log``.
                
        Returns
        -------
        fig : ``matplotlib.figure``
            Figure object.
                
        """
        fig, ax = plt.subplots(1,4, figsize=(14,3.5))
        fig.subplots_adjust(wspace=0.2)

        cax = []
        for k in range(len(ax)):
            cax.append(fig.add_axes([ax[k].get_position().x0, ax[k].get_position().y1 + 0.07,
                                     ax[k].get_position().width, 0.04]))
        
        sm = [] 
        if (ntype == 'none') or hasattr(self, 'R'):
            ode.plot_frame(ax[0], y, ntype='none')
            
            # Real-space density field
            f = (self.f*(torch.exp(-1j*torch.matmul(y, self.Q.transpose(1,0))).sum(dim=0))).view(self.n,self.n)
            
        else:
            cax.insert(1, fig.add_axes([ax[0].get_position().x0, ax[0].get_position().y1 + 0.13,
                                        ax[0].get_position().width, 0.04]))
            sm.append(ode.plot_frame(ax[0], y[0].reshape(self.N, self.N), ntype=ntype))
                
            # Real-space density field
            f_real = torch.matmul(self.f(y[0]), torch.cos(self.arg))
            f_imag = torch.matmul(self.f(y[0]), torch.sin(self.arg))
            f = (f_real - 1j*f_imag).view(self.n,self.n)
  
        p = np.real(ifftshift(ifftn(fftshift(f))))
        
        sm.extend(self.plot_probe(ax[:2]))
        sm.append(ode.plot_frame(ax[2], self(y).reshape(self.n, self.n), vmin=vmin, vmax=vmax, ntype='log'))
        
        l = self.L/self.dq
        sm.append(ode.plot_frame(ax[3], p, alpha=0.9, extent=(-l/2.,l/2.,-l/2.,l/2.)))
        ax[3].set_xlim(ax[0].get_xlim())
        ax[3].set_ylim(ax[0].get_ylim())

        cbar = []
        cprops = props.copy()
        cprops.set_size(props.get_size()-2)
        for k in range(len(sm)):
            cbar.append(plt.colorbar(sm[k], cax=cax[k], orientation='horizontal'))
            if (k > 0) & (len(sm) > 4):
                cbar[k].ax.xaxis.set_ticks_position('top')
            format_axis(cbar[k].ax, cprops, xbins=3)

        ax[0].text(0.1, 0.85, r'$\theta(\mathbf{r}) \odot p(\mathbf{r})$', color='white', ha='left', va='center',
                   transform=ax[0].transAxes, fontproperties=props)
        ax[1].text(0.1, 0.85, r'$P(\mathbf{q})$', color='white', ha='left', va='center',
                   transform=ax[1].transAxes, fontproperties=props)
        ax[2].text(0.1, 0.85, r'$I(\mathbf{q})$', color='white', ha='left', va='center',
                   transform=ax[2].transAxes, fontproperties=props)
        ax[3].text(0.1, 0.85, r'$\Re(\rho(\mathbf{r}))$', color='white', ha='left', va='center',
                   transform=ax[3].transAxes, fontproperties=props)
        return fig
    
    
    
class CXSGrid(CXS):
    def __init__(self, N, n, L=1., dq=1., f_probe=None, f_mask=None, f=None):
        super(CXSGrid, self).__init__(n, L, dq, f_probe, f_mask)
        
        self.N = N
        
        # Real-space grid
        x = torch.arange(-L/2., L/2., L/N)
        X, Y = torch.meshgrid(x, x, indexing='xy')
        self.r = torch.stack((X.flatten(), Y.flatten()), dim=1)
        self.arg = nn.Parameter(torch.matmul(self.r, self.Q.transpose(1,0)), requires_grad=False)
        
        if (f not in ['phase']) or (f==None):
            self.f = getattr(self, 'f_none')
        else:
            self.f = getattr(self, 'f_' + f)
            
            
    def f_phase(self, y, pol=1.):
        return 1. + pol*torch.cos(y)
        
        
    def f_none(self, y, pol=1.):
        return y
            
        
    def forward(self, y, pol=1):
        y_real = torch.matmul(self.f(y, pol), torch.cos(self.arg))
        y_imag = torch.matmul(self.f(y, pol), torch.sin(self.arg))

        # Sum over channel dimension
        y_real = y_real.sum(dim=-2)
        y_imag = y_imag.sum(dim=-2)

        # Convolve with probe
        size = y_real.shape
        y_real = self.probe(y_real.view(-1,1,self.n,self.n)).view(*size)
        y_imag = self.probe(y_imag.view(-1,1,self.n,self.n)).view(*size)

        return (y_real**2 + y_imag**2)*self.mask
    
    
    
class CXSPoint(CXS):
    def __init__(self, R, n, L=1., dq=1., f_probe=None, f_mask=None):
        super(CXSPoint, self).__init__(n, L, dq, f_probe, f_mask)
        
        self.R = R
        self.f = nn.Parameter(self.f_sphere(torch.sqrt((self.Q**2).sum(dim=1)), R), requires_grad=False)
    
    def f_sphere(self, q, R):
        V = 4./3.*np.pi*R**3
        f = V*torch.ones_like(q)
        f[q > 0.] = V*3*(torch.sin(q[q > 0.]*R) - q[q > 0.]*R*torch.cos(q[q > 0.]*R))/(q[q > 0.]*R)**3
        f /= f.max()
        return f
        
        
    def forward(self, y):
        arg = torch.matmul(y, self.Q.transpose(1,0))
        th = ((y**2).sum(dim=-1, keepdims=True) < self.L**2)
        
        # Threshold values outside the domain
        y_real = self.f*((th*torch.cos(arg)).sum(dim=-2))
        y_imag = self.f*((th*torch.sin(arg)).sum(dim=-2))
        
        # Convolve with probe
        size = y_real.shape
        y_real = self.probe(y_real.view(-1,1,self.n,self.n)).view(*size)
        y_imag = self.probe(y_imag.view(-1,1,self.n,self.n)).view(*size)
        
        return (y_real**2 + y_imag**2)*self.mask