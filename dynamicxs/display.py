import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cmcrameri.cm as cm

from matplotlib import animation

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'

#props = fm.FontProperties(family=['Trebuchet MS', 'sans-serif'], size='xx-large')
props = fm.FontProperties(family=['Lato', 'sans-serif'], size='xx-large')
    
plt.rcParams['animation.writer'] = 'pillow'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.linewidth'] = 1

def format_axis(ax, props, xlabel='', ylabel='', xbins=None, ybins=None):
    ax.set_xlabel(xlabel, fontproperties=props)
    ax.set_ylabel(ylabel, fontproperties=props)
    ax.yaxis.offsetText.set_fontproperties(props)
    
    for label in ax.get_xticklabels(which='both'):
        label.set_fontproperties(props)
    for label in ax.get_yticklabels(which='both'):
        label.set_fontproperties(props)
        
    if xbins:
        try: ax.locator_params(axis='x', nbins=xbins)
        except: ax.locator_params(axis='x', numticks=xbins+1)
    if ybins:
        try: ax.locator_params(axis='y', nbins=ybins)
        except: ax.locator_params(axis='y', numticks=ybins+1)
            
            
def format_str(x):
    if isinstance(x, float):
        if x > 9:
            return '{:.1g}'.format(x).replace('+', '')
        else:
            return str(x).replace('.', 'p')
    elif isinstance(x, int):
        return str(x)
    else:
        return ''
    
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Adapted from: https://gist.github.com/salotz/4f585aac1adb6b14305c
    '''
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(
        n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap