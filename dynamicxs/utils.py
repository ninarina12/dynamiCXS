import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cmcrameri.cm as cm

props = fm.FontProperties(family=['Myriad Pro', 'Trebuchet MS', 'sans-serif'], size='xx-large')
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.linewidth'] = 1

def format_axis(ax, props, xlabel='', ylabel='', xbins=None, ybins=None):
    ax.set_xlabel(xlabel, fontproperties=props)
    ax.set_ylabel(ylabel, fontproperties=props)
    ax.yaxis.offsetText.set_fontproperties(props)
    
    for label in ax.get_xticklabels():
        label.set_fontproperties(props)
    for label in ax.get_yticklabels():
        label.set_fontproperties(props)
        
    if xbins:
        try: ax.locator_params(axis='x', nbins=xbins)
        except: ax.locator_params(axis='x', numticks=xbins+1)
    if ybins:
        try: ax.locator_params(axis='y', nbins=ybins)
        except: ax.locator_params(axis='y', numticks=ybins+1)