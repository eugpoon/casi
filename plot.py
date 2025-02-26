# # Imports
# import os
# import calendar
# import pandas as pd
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

# # Suppress warnings
# warnings.filterwarnings('ignore')

# # Configure matplotlib
# plt.rcParams['figure.figsize'] = (15, 4)
# plt.rcParams['figure.dpi'] = 600
# plt.rcParams['font.size'] = 10
# plt.rcParams['figure.titlesize'] = 15
# plt.rcParams['axes.linewidth'] = 0.1
# plt.rcParams['patch.linewidth'] = 0
# plt.rcParams['grid.linewidth'] = 0.1  

# month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
# colors = px.colors.qualitative.Plotly
# global title
# title = f'{center}: {title_var} Per Year {month_title} Across CMIP6 Models'

# def plot_dist(df, title_var, agg='mean', alpha=1):
    
#     for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
#         sns.kdeplot(data=df.loc[ssp].agg(agg, axis=1), label=f'{ssp}_{agg}', 
#                     color=colors[i % len(colors)], linewidth=0.5, alpha=alpha)
#     plt.title(title)
#     plt.xlabel(title_var)
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.show()
    

# def plot_ts(df, title_var, agg='mean', alpha=1, plotly=False):
#     if plotly:
#         fig = go.Figure()
#         for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
#             data = df.loc[ssp].agg(agg, axis=1)
#             fig.add_trace(go.Scatter(x=data.index, y=data, name=ssp, opacity=alpha,
#                                      line=dict(color=colors[i], width=0.8), mode='lines'))
#         fig.update_layout(title=title, xaxis_title='Year', yaxis_title=title_var,
#                           width=1000, height=300, margin=dict(l=20, r=20, t=30, b=20),
#                           paper_bgcolor='white', plot_bgcolor='white',
#                           xaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1),
#                           yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1))
#         fig.show()
#     else:
#         for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
#             data = df.loc[ssp].agg(agg, axis=1)
#             sns.lineplot(data=data, label=ssp, color=colors[i], linewidth=0.5, alpha=alpha)
#         plt.title(title)
#         plt.xlabel('Year')
#         plt.ylabel(title_var)
#         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         plt.tight_layout()
#         plt.show()


# def plot_severity(comp_thres_dfs, title_var, center, month_title, regex=None, agg='mean', alpha=1, plotly=False):
#     ssps = list(pd.unique([idx[0] for df in comp_thres_dfs.values() for idx in df.index]))
#     colors = px.colors.qualitative.Plotly
#     title = f'{center}: {title_var} Per Year {month_title} Across CMIP6 Models'

#     # Filter dataframes based on regex if provided
#     if regex:
#         comp_thres_dfs = {k: v[v.index.get_level_values(1).str.contains(regex)] for k, v in comp_thres_dfs.items()}

#     if plotly:
#         fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes=True, shared_yaxes=True, 
#                             subplot_titles=ssps, vertical_spacing=0.05)

#         for i, ssp in enumerate(ssps, start=1):
#             for j, (threshold, df) in enumerate(comp_thres_dfs.items()):
#                 ssp_data = df[df.index.get_level_values(0) == ssp]
#                 fig.add_trace(
#                     go.Scatter(x=ssp_data.index.get_level_values(1), y=ssp_data.agg(agg, axis=1), name=threshold,
#                                line=dict(color=colors[j % len(colors)], width=0.8),
#                                mode='lines', opacity=alpha,
#                                legendgroup=threshold, showlegend=(i == 1)),
#                     row=i, col=1
#                 )

#         fig.update_layout(title=title, width=1000, height=200*len(ssps), margin=dict(l=50, r=20, t=120, b=50),
#             paper_bgcolor='white', plot_bgcolor='white', legend_title='Thresholds',
#             legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
#         )

#         for i in range(1, len(ssps)+1):
#             fig.update_xaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1)
#             fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1,
#                              title_text=title_var if i == len(ssps)//2 + 1 else '')
#             fig.update_xaxes(title_text='Year' if i == len(ssps) else '', row=i, col=1)

#         fig.show()

#     else:
#         fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 5*len(ssps)), sharex=True, sharey=True)
#         fig.suptitle(title, fontsize=16, y=0.95)

#         lines, labels = [], []

#         for i, ssp in enumerate(ssps):
#             ax = axes[i] if len(ssps) > 1 else axes
#             for j, (threshold, df) in enumerate(comp_thres_dfs.items()):
#                 ssp_data = df[df.index.get_level_values(0) == ssp]
#                 line = sns.lineplot(x=ssp_data.index.get_level_values(1), y=ssp_data.agg(agg, axis=1), 
#                                     label=threshold, alpha=alpha, color=colors[j % len(colors)], 
#                                     linewidth=0.8, ax=ax)
#                 if i == 0:
#                     lines.append(line.lines[-1])
#                     labels.append(threshold)

#             ax.set_title(ssp)
#             ax.grid(True, color='lightgrey', linewidth=0.1)
#             ax.set_ylabel(title_var if i == len(ssps)//2 else '')
#             ax.set_xlabel('Year' if i == len(ssps)-1 else '')
#             ax.legend().remove()

#         fig.legend(lines, labels, title='Thresholds', bbox_to_anchor=(0.5, 0.98), 
#                    loc='lower center', ncol=len(comp_thres_dfs), frameon=False)
#         plt.tight_layout()
#         plt.subplots_adjust(top=0.90)
#         plt.show()

# def main(center, months):
#     month_title = '(' + ''.join(month_dict[month] for month in months) + ')'

# def plot_all(df_dict, title_var, agg='mean', alpha=1):
#     plot_dist(df_dict, title_var, agg, alpha)
#     plot_ts(df_dict, title_var, agg, alpha)

# Imports
import os
import calendar
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['figure.figsize'] = (15, 4)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['grid.linewidth'] = 0.1  

month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
colors = px.colors.qualitative.Plotly

def plot_dist(df, title, title_var, agg='mean', alpha=1):
    for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
        sns.kdeplot(data=df.loc[ssp].agg(agg, axis=1), label=f'{ssp}_{agg}', 
                    color=colors[i % len(colors)], linewidth=0.5, alpha=alpha)
    plt.title(title)
    plt.xlabel(title_var)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_ts(df, title, title_var, agg='mean', alpha=1, plotly=False):
    if plotly:
        fig = go.Figure()
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp].agg(agg, axis=1)
            fig.add_trace(go.Scatter(x=data.index, y=data, name=ssp, opacity=alpha,
                                     line=dict(color=colors[i], width=0.8), mode='lines'))
        fig.update_layout(title=title, xaxis_title='Year', yaxis_title=title_var,
                          width=1000, height=300, margin=dict(l=20, r=20, t=30, b=20),
                          paper_bgcolor='white', plot_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1),
                          yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1))
        fig.show()
    else:
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp].agg(agg, axis=1)
            sns.lineplot(data=data, label=ssp, color=colors[i], linewidth=0.5, alpha=alpha)
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel(title_var)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

def plot_severity(comp_thres_dfs, title, title_var, ssps, filter, agg='mean', alpha=1, plotly=False):
    if plotly:
        fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes=True, shared_yaxes=True, 
                            subplot_titles=ssps, vertical_spacing=0.05)

        for i, ssp in enumerate(ssps, start=1):
            for j, (threshold, df) in enumerate(comp_thres_dfs.items()):
                ssp_data = df[df.index.get_level_values(0) == ssp].filter(regex=filter)
                fig.add_trace(
                    go.Scatter(x=ssp_data.index.get_level_values(1), y=ssp_data.agg(agg, axis=1), name=threshold,
                               line=dict(color=colors[j % len(colors)], width=0.8),
                               mode='lines', opacity=alpha,
                               legendgroup=threshold, showlegend=(i == 1)),
                    row=i, col=1
                )

        fig.update_layout(title=title, width=1000, height=200*len(ssps), margin=dict(l=50, r=20, t=120, b=50),
            paper_bgcolor='white', plot_bgcolor='white', legend_title='Thresholds',
            legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
        )

        for i in range(1, len(ssps)+1):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1,
                             title_text=title_var if i == len(ssps)//2 + 1 else '')
            fig.update_xaxes(title_text='Year' if i == len(ssps) else '', row=i, col=1)

        fig.show()

    else:
        fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 5*len(ssps)), sharex=True, sharey=True)
        fig.suptitle(title, fontsize=16, y=0.95)

        lines, labels = [], []

        for i, ssp in enumerate(ssps):
            ax = axes[i] if len(ssps) > 1 else axes
            for j, (threshold, df) in enumerate(comp_thres_dfs.items()):
                ssp_data = df[df.index.get_level_values(0) == ssp].filter(regex=filter)
                line = sns.lineplot(x=ssp_data.index.get_level_values(1), y=ssp_data.agg(agg, axis=1), 
                                    label=threshold, alpha=alpha, color=colors[j % len(colors)], 
                                    linewidth=0.8, ax=ax)
                if i == 0:
                    lines.append(line.lines[-1])
                    labels.append(threshold)

            ax.set_title(ssp)
            ax.grid(True, color='lightgrey', linewidth=0.1)
            ax.set_ylabel(title_var if i == len(ssps)//2 else '')
            ax.set_xlabel('Year' if i == len(ssps)-1 else '')
            ax.legend().remove()

        fig.legend(lines, labels, title='Thresholds', bbox_to_anchor=(0.5, 0.98), 
                   loc='lower center', ncol=len(comp_thres_dfs), frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

def plot_all(comp_thres_dfs, title_var, center, month_title, agg='mean', alpha=1):
    ssps = list(pd.unique([idx[0] for df in comp_thres_dfs.values() for idx in df.index]))
    title = f'{center}: {title_var} Per Year {month_title} Across CMIP6 Models'
    
    # Use the first key/value for plot_dist and plot_ts
    first_key = next(iter(comp_thres_dfs))
    first_df = comp_thres_dfs[first_key]
    
    plot_dist(first_df, title, title_var, agg, alpha)
    plot_ts(first_df, title, title_var, agg, alpha)
    
    # Use the remaining values for plot_severity
    severity_dfs = {k: v for k, v in list(comp_thres_dfs.items())[1:]}
    plot_severity(severity_dfs, title, title_var, ssps, filter, agg=agg, alpha=alpha)
    
# This function can be used to set up the global variables
def setup_globals(center, months, title_var):
    global title, month_title
    month_title = '(' + ''.join(month_dict[month] for month in months) + ')'
    title = f'{center}: {title_var} Per Year {month_title} Across CMIP6 Models'



