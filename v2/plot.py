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

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (15, 4)
plt.rcParams['figure.dpi'] = 300 # 600
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['grid.linewidth'] = 0.1


colors = px.colors.qualitative.Plotly
month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}

def check_df(df):
    df = df.filter(regex=var)
    if df.empty:
        raise ValueError('Empty dataframe')
    return df
    

def plot_dist(df, agg='mean', alpha=1):
    df = check_df(df)
    for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
        sns.kdeplot(data=df.loc[ssp].agg(agg, axis=1), label=f'{ssp}_{agg}', 
                    color=colors[i % len(colors)], linewidth=0.5, alpha=alpha)
    plt.title(title)
    plt.xlabel(title_var)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_ts(df, agg='mean', alpha=1):
    df = check_df(df)
    if plotly:
        fig = go.Figure()
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp]
            mean_data = data.agg(agg, axis=1)
            p10 = data.quantile(0.1, axis=1)
            p90 = data.quantile(0.9, axis=1)
             # Add error range (with no legend entry)
            fig.add_trace(go.Scatter(x=mean_data.index.tolist() + mean_data.index[::-1].tolist(),
                                     y=p90.tolist() + p10[::-1].tolist(),
                                     fill='toself', fillcolor=colors[i],
                                     line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip',
                                     showlegend=False, legendgroup=ssp, opacity=0.2))
            fig.add_trace(go.Scatter(x=mean_data.index, y=mean_data, name=ssp, mode='lines',
                                     line=dict(color=colors[i], width=0.8), opacity=alpha,
                                     legendgroup=ssp))
            
            # fig.add_trace(go.Scatter(x=agg_data.index, y=agg_data, name=ssp, opacity=alpha,
            #                          line=dict(color=colors[i], width=0.8), mode='lines'))

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
        plt.legend(loc='center left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp]
            agg_data = data.agg(agg, axis=1)
            p10 = data.quantile(0.1, axis=1)
            p90 = data.quantile(0.9, axis=1)
            
            sns.lineplot(data=agg_data, label=ssp, color=colors[i], linewidth=0.8, alpha=alpha)
            plt.fill_between(agg_data.index, p10, p90, color=colors[i], alpha=0.1)
            plt.title(title)
            plt.xlabel('Year')
            plt.ylabel(title_var)
            plt.legend(loc='center left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()

def plot_severity(results, ssps, var, agg='mean', alpha=1):
    if plotly:
        fig = make_subplots(rows=3, cols=1, shared_xaxes='all', shared_yaxes='all', 
                            subplot_titles=ssps, vertical_spacing=0.05)

        for i, ssp in enumerate(ssps, start=1):
            for j, (threshold, df) in enumerate(results.items()):
                data = check_df(df[df.index.get_level_values(0) == ssp])
                mean_data = data.agg(agg, axis=1)
                p10 = data.quantile(0.1, axis=1)
                p90 = data.quantile(0.9, axis=1)

                # Add error range
                fig.add_trace(go.Scatter(
                    x=(mean_data.index.get_level_values(1).tolist() + 
                       mean_data.index.get_level_values(1)[::-1].tolist()),
                    y=p90.tolist() + p10[::-1].tolist(), showlegend=False, legendgroup=threshold,
                    fill='toself', fillcolor=colors[j % len(colors)], opacity=0.2, 
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'), row=i, col=1)
                
                fig.add_trace(go.Scatter(
                    x=mean_data.index.get_level_values(1), y=mean_data, name=threshold,
                    line=dict(color=colors[j % len(colors)], width=0.8), mode='lines',
                    opacity=alpha, legendgroup=threshold, showlegend=(i == 1)), row=i, col=1)
                
                # fig.add_trace(go.Scatter(
                #     x=data.index.get_level_values(1), y=data.agg(agg, axis=1), name=threshold,
                #     line=dict(color=colors[j % len(colors)], width=0.8), mode='lines',
                #     opacity=alpha, legendgroup=threshold, showlegend=(i == 1)), row=i, col=1)

        fig.update_layout(title='Severity Comparison: '+title, width=1000, height=200*len(ssps),
                          margin=dict(l=50, r=20, t=120, b=50),
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
        fig.suptitle('Severity Comparison: '+title, fontsize=16, y=0.95)

        lines, labels = [], []

        for i, ssp in enumerate(ssps):
            ax = axes[i] if len(ssps) > 1 else axes
            for j, (threshold, df) in enumerate(results.items()):
                data = check_df(df[df.index.get_level_values(0) == ssp])
                mean_data = data.agg(agg, axis=1)
                p10 = data.quantile(0.1, axis=1)
                p90 = data.quantile(0.9, axis=1)
                
                line = sns.lineplot(x=data.index.get_level_values(1), y=mean_data, 
                                    label=threshold, alpha=alpha, color=colors[j % len(colors)], 
                                    linewidth=0.8, ax=ax)
                ax.fill_between(mean_data.index.get_level_values(1), p10, p90, 
                        color=colors[j % len(colors)], alpha=0.1)
                
                if i == 0:
                    lines.append(line.lines[-1])
                    labels.append(threshold)

            ax.set_title(ssp)
            ax.grid(True, color='lightgrey', linewidth=0.2)
            ax.set_ylabel(title_var if i == len(ssps)//2 else '')
            ax.set_xlabel('Year' if i == len(ssps)-1 else '')
            ax.legend().remove()

        fig.legend(lines, labels, title='Thresholds', bbox_to_anchor=(0.5, 0.98), 
                   loc='lower center', ncol=len(results), frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

def plot_all(results, agg='mean', alpha=1):
    ssps = list(pd.unique([idx[0] for df in results.values() for idx in df.index]))
    
    first_key = next(iter(results))
    first_df = results[first_key]
    
    plot_dist(first_df, agg, alpha)
    plot_ts(first_df, agg, alpha)
    
    # Use the remaining values for plot_severity
    severity_dfs = {k: v for k, v in list(results.items())[1:]}
    plot_severity(severity_dfs, ssps, var, agg=agg, alpha=alpha)
    
# This function can be used to set up the global variables
def initialize(center_, months_, title_var_, var_, plotly_=False):
    global title, title_var, var, plotly
    month_title = '(' + ''.join(month_dict[month] for month in months_) + ')'
    title = f'{center_} - {title_var_} Per Year {month_title} Across CMIP6 Models'
    title_var = title_var_
    var = var_
    plotly = plotly_



