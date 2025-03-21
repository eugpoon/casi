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

# Set plot defaults
plt.rcParams['figure.figsize'] = (15, 4)
plt.rcParams['figure.dpi'] = 100 # 300, 600
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.2
plt.rcParams['legend.loc'] = 'center left'


class Plot:
    ##################################################
    ##                   General                    ##
    ##################################################

    def __init__(self, center, months, var_title, var):
        self.center = center
        self.months = months
        self.var_title = var_title
        self.var = var
        self.colors = px.colors.qualitative.Plotly
        self.month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
        self.month_title = '(' + ''.join(self.month_dict[month] for month in months) + ') ' if months else ''
        self.title = f'{center} - {var_title} Per Year {self.month_title}Across CMIP6 Models'

    ##################################################
    ##                   KDE Plot                   ##
    ##################################################

    def dist_sns(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot distribution using seaborn.'''
        for i, (ssp, dd) in enumerate(df.groupby('ssp')):
            sns.kdeplot(x=dd[f'{agg}{self.var}'], color=self.colors[i % len(self.colors)], 
                        label=ssp, alpha=alpha, linewidth=0.5)
        plt.title(f'{self.title} {threshold}')
        plt.xlabel(self.var_title)
        plt.legend(bbox_to_anchor=(1, 0.5))
        plt.show()

    ##################################################
    ##                  Time Series                 ##
    ##################################################

    def ts_plotly(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot time series using plotly.'''
        fig = go.Figure()
        for i, (ssp, dd) in enumerate(df.groupby('ssp')):
            fig.add_trace(go.Scatter(x=dd.date.tolist() + dd.date[::-1].tolist(),
                    y=dd[f'p90{self.var}'].tolist() + dd[f'p10{self.var}'][::-1].tolist(), fill='toself',
                    fillcolor=self.colors[i], showlegend=False, legendgroup=ssp, opacity=0.1,
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=dd.date, y=dd[f'{agg}{self.var}'], name=ssp, mode='lines', opacity=alpha,
                                     line=dict(color=self.colors[i], width=0.8), legendgroup=ssp))
        fig.update_layout(title=f'{self.title} {threshold}', xaxis_title='Year', yaxis_title=self.var_title,
                          width=1000, height=300, margin=dict(l=20, r=20, t=30, b=20),
                          paper_bgcolor='white', plot_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1),
                          yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1))
        fig.show()

    def ts_sns(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot time series using seaborn.'''
        ssps = df.ssp.unique()
        sns.lineplot(data=df, x='date', y=f'{agg}{self.var}', hue='ssp',
                     palette=self.colors, linewidth=0.5, alpha=alpha)
        plt.title(f'{self.title} {threshold}')
        plt.xlabel('Year')
        plt.ylabel(self.var_title)
        plt.xlim(min(df.date), max(df.date))
        plt.legend(bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

        handles=[]
        fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 4*len(ssps)), sharex=True, sharey=True)
        fig.suptitle(f'{self.title} {threshold}', fontsize=16)
        for i, (ssp, dd) in enumerate(df.groupby('ssp')):
            ax = axes[i] if len(ssps) > 1 else axes
            line = sns.lineplot(data=dd, x='date', y=f'{agg}{self.var}', ax=ax,
                        color=self.colors[i % len(self.colors)], label=ssp, alpha=alpha, linewidth=0.5)
            ax.fill_between(dd.date, dd[f'p10{self.var}'], dd[f'p90{self.var}'],
                            color=self.colors[i], alpha=0.1)
            ax.set(ylabel=(self.var_title if i == len(ssps)//2 else ''), 
                   xlabel=('Year' if i == len(ssps)-1 else ''), xlim=(min(df.date), max(df.date)))
            ax.grid(True, color='lightgrey', linewidth=0.2)
            ax.get_legend().set_visible(False)
            handles.append(line.lines[0])
        fig.legend(handles, ssps, bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
            
    ##################################################
    ##    Threshold Comparison (1 ssp per plot)     ##
    ##################################################

    def severity_plotly(self, df, agg='mean', alpha=1):
        '''Plot severity comparison using plotly.'''
        ssps, thres = df.ssp.unique(), df.threshold.unique()
        fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes='all', shared_yaxes='all', 
                            subplot_titles=ssps, vertical_spacing=0.03)

        for i, ssp in enumerate(ssps, start=1):
            for j, t in enumerate(thres):
                dd = df[(df.threshold==t) & (df.ssp==ssp)]
                fig.add_trace(go.Scatter(x=dd.date.tolist() + dd.date[::-1].tolist(),
                    y=dd[f'p90{self.var}'].tolist() + dd[f'p10{self.var}'][::-1].tolist(), showlegend=False, 
                    legendgroup=t, fill='toself', fillcolor=self.colors[j % len(self.colors)], opacity=0.1, 
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'), row=i, col=1)
                fig.add_trace(go.Scatter(x=dd.date, y=dd[f'{agg}{self.var}'], name=t,
                    line=dict(color=self.colors[j % len(self.colors)], width=0.8), mode='lines',
                    opacity=alpha, legendgroup=t, showlegend=(i==1)), row=i, col=1)

        fig.update_layout(title=f'Severity Comparison: {self.title}', width=1000, height=250*len(ssps),
                          margin=dict(l=20, r=20, t=50, b=50),
                          paper_bgcolor='white', plot_bgcolor='white', legend_title='Thresholds',
                          legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5))

        for i in range(1, len(ssps)+1):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1, 
                             title_text='Year' if i==len(ssps) else '')
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1,
                             title_text=self.var_title if i==len(ssps)//2+1 else '')
        fig.show()

    def severity_sns(self, df, agg='mean', alpha=1):
        '''Plot severity comparison using seaborn.'''
        ssps, thres = df.ssp.unique(), df.threshold.unique()
        fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 4*len(ssps)), sharex=True, sharey=True)
        fig.suptitle(f'Severity Comparison: {self.title}', fontsize=16)
        for i, (ssp, dd) in enumerate(df.groupby('ssp')):
            ax = axes[i] if len(ssps) > 1 else axes
            sns.lineplot(data=dd, x='date', y=f'{agg}{self.var}', hue='threshold', ax=ax,
                        color=self.colors[i % len(self.colors)], alpha=alpha, linewidth=0.5)
            # for j, t in enumerate(thres):
            #     mask = dd.threshold == t
            #     p10, p90 = dd[f'p10{self.var}'], dd[f'p90{self.var}']
            #     ax.fill_between(x=dd[mask].date, y1=p10[mask], y2=p90[mask], color=self.colors[j], alpha=0.1)
            ax.set(title=ssp, ylabel=(self.var_title if i == len(ssps)//2 else ''), 
                   xlabel=('Year' if i == len(ssps)-1 else ''), xlim=(min(df.date), max(df.date)))
            ax.grid(True, color='lightgrey', linewidth=0.2)
            ax.get_legend().set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1, 0.5))
        plt.tight_layout()        
        plt.show()
    
    ##################################################
    ##     Variable Comparison (1 ssp per plot)     ##
    ##################################################

    def variable_comp(self, df, vars:list, metrics:dict, agg='mean', alpha=1):
        '''Plot variable comparison.'''
        ssps = df.ssp.unique()
        for j, (t, tt) in enumerate(df.groupby('threshold')):
            fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 4*len(ssps)), sharex=True, sharey=True)
            fig.suptitle(f'{self.center} - ... Per Year {self.month_title} Across CMIP6 Models; {t}', 
                         fontsize=16)
            for i, (ssp, dd) in enumerate(tt.groupby('ssp')):
                ax = axes[i] if len(ssps) > 1 else axes
                for v, var in enumerate(vars):
                    line = sns.lineplot(data=dd, x='date', y=f'{agg}{var}', ax=ax,
                            label=metrics[var], color=self.colors[v], linewidth=0.8, alpha=alpha)
                ax.set(title=ssp, ylabel='', xlabel=('Year' if i == len(ssps)-1 else ''),
                       xlim=(min(df.date), max(df.date)))
                ax.grid(True, color='lightgrey', linewidth=0.2)
                ax.get_legend().set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()
            
    ##################################################
    ##    Time Scale Comparison (1 ssp per plot)    ##
    ##################################################
    
    def scale_comp(self, df:pd.DataFrame(), metrics:dict, threshold:str='', agg:str='mean', alpha:float=1):
        ssps = df.ssp.unique()
        for var, var_title in metrics.items():
            fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 4*len(ssps)), sharex=True, sharey=True)
            fig.suptitle(f'{self.center} - {var_title} Per Year {self.month_title}'+
                         f'Across CMIP6 Models; {threshold}', fontsize=16)
            # for i, ssp in enumerate(ssps):
            for i, (ssp, dd) in enumerate(df.groupby('ssp')):
                ax = axes[i] if len(ssps) > 1 else axes
                sns.lineplot(data=dd, x='date', y=f'{agg}{var}', hue='scale', ax=ax,
                            palette=self.colors, linewidth=0.8, alpha=alpha)
                ax.set(title=ssp, ylabel=(var_title if i == len(ssps)//2 else ''), 
                       xlabel=('Year' if i == len(ssps)-1 else ''), xlim=(min(df.date), max(df.date)))
                ax.get_legend().set_visible(False)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()
                
    ##################################################
    ##               Plot 1 Variable                ##
    ##################################################
    
    def all(self, df:pd.DataFrame(), plotly_:bool, agg:str='mean', alpha:float=1):
        '''Plot all types of plots.'''
        ts_plot = self.ts_plotly if plotly_ else self.ts_sns
        severity_plot = self.severity_plotly if plotly_ else self.severity_sns
        for t in df.threshold.unique():
            dd = df[df.threshold == t].drop(columns='threshold')
            self.dist_sns(dd, agg, alpha, t)
            ts_plot(dd, agg, alpha, t)
        severity_plot(df, agg, alpha)

















        