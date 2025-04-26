# plot_settings.py

import matplotlib.pyplot as plt
import seaborn as sns
from plotly.colors import qualitative

plt.rcParams['figure.figsize'] = (15, 4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['grid.linewidth'] = 0.1

sns.set(rc={
    'axes.facecolor':'#111111', 
    'axes.edgecolor': '#444',
    'figure.facecolor':'#111111',
    'grid.color': '#444',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
})

sns.set_palette(qualitative.Plotly)
