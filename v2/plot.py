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
plt.rcParams['figure.dpi'] = 150 # 300, 600
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.2

class Plot:
    ##################################################
    ##                   General                    ##
    ##################################################

    def __init__(self, center_, months_, title_var_, var_):
        self.center = center_
        self.months = months_
        self.title_var = title_var_
        self.var = var_
        self.colors = px.colors.qualitative.Plotly
        self.month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}

        self.month_title = '(' + ''.join(self.month_dict[month] for month in months_) + ')'
        self.title = f'{center_} - {title_var_} Per Year {self.month_title} Across CMIP6 Models'

    def check_df(self, df):
        '''Filter dataframe based on variable.'''
        df = df.filter(regex=self.var)
        if df.empty:
            raise ValueError('Empty dataframe')
        return df
    
    ##################################################
    ##                   KDE Plot                   ##
    ##################################################

    def dist_sns(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot distribution using seaborn.'''
        df = self.check_df(df)
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            sns.kdeplot(data=df.loc[ssp].agg(agg, axis=1), label=f'{ssp}_{agg}', 
                        color=self.colors[i % len(self.colors)], linewidth=0.5, alpha=alpha)
        plt.title(f'{self.title} {threshold}')
        plt.xlabel(self.title_var)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    ##################################################
    ##                  Time Series                 ##
    ##################################################

    def ts_plotly(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot time series using plotly.'''
        df = self.check_df(df)
        fig = go.Figure()
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp]
            agg_data, p10, p90 = data.agg(agg, axis=1), data.quantile(0.1, axis=1), data.quantile(0.9, axis=1)
            
            # Add error range (with no legend entry)
            fig.add_trace(go.Scatter(x=agg_data.index.tolist() + agg_data.index[::-1].tolist(),
                                     y=p90.tolist() + p10[::-1].tolist(), fill='toself',
                                     fillcolor=self.colors[i], showlegend=False, legendgroup=ssp, opacity=0.2,
                                     line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'))
            
            fig.add_trace(go.Scatter(x=agg_data.index, y=agg_data, name=ssp, mode='lines', opacity=alpha,
                                     line=dict(color=self.colors[i], width=0.8), legendgroup=ssp))

        fig.update_layout(title=f'{self.title} {threshold}', xaxis_title='Year', yaxis_title=self.title_var,
                          width=1000, height=300, margin=dict(l=20, r=20, t=30, b=20),
                          paper_bgcolor='white', plot_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1),
                          yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.1))
        fig.show()

    def ts_sns(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot time series using seaborn.'''
        df = self.check_df(df)

        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp].agg(agg, axis=1)
            sns.lineplot(data=data, label=ssp, color=self.colors[i], linewidth=0.5, alpha=alpha)
        plt.title(f'{self.title} {threshold}')
        plt.xlabel('Year')
        plt.ylabel(self.title_var)
        plt.legend(loc='center left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
        for i, ssp in enumerate(df.index.get_level_values('ssp').unique()):
            data = df.loc[ssp]
            agg_data, p10, p90 = data.agg(agg, axis=1), data.quantile(0.1, axis=1), data.quantile(0.9, axis=1)
            
            sns.lineplot(data=agg_data, label=ssp, color=self.colors[i], linewidth=0.8, alpha=alpha)
            plt.fill_between(agg_data.index, p10, p90, color=self.colors[i], alpha=0.1)
            plt.title(f'{self.title} {threshold}')
            plt.xlabel('Year')
            plt.ylabel(self.title_var)
            plt.legend(loc='center left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()
            
    ##################################################
    ##    Threshold Comparison (1 ssp per plot)     ##
    ##################################################

    def severity_plotly(self, results, ssps, agg='mean', alpha=1):
        '''Plot severity comparison using plotly.'''
        fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes='all', shared_yaxes='all', 
                            subplot_titles=ssps, vertical_spacing=0.05)

        for i, ssp in enumerate(ssps, start=1):
            for j, (threshold, df) in enumerate(results.items()):
                data = self.check_df(df[df.index.get_level_values(0) == ssp])
                agg_data,p10,p90 = data.agg(agg, axis=1), data.quantile(0.1, axis=1), data.quantile(0.9, axis=1)

                # Add error range
                fig.add_trace(go.Scatter(
                    x=(agg_data.index.get_level_values(1).tolist() + 
                       agg_data.index.get_level_values(1)[::-1].tolist()),
                    y=p90.tolist() + p10[::-1].tolist(), showlegend=False, legendgroup=threshold,
                    fill='toself', fillcolor=self.colors[j % len(self.colors)], opacity=0.2, 
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'), row=i, col=1)
                
                fig.add_trace(go.Scatter(
                    x=agg_data.index.get_level_values(1), y=agg_data, name=threshold,
                    line=dict(color=self.colors[j % len(self.colors)], width=0.8), mode='lines',
                    opacity=alpha, legendgroup=threshold, showlegend=(i == 1)), row=i, col=1)

        fig.update_layout(title='Severity Comparison: '+self.title, width=1000, height=200*len(ssps),
                          margin=dict(l=50, r=20, t=120, b=50),
                          paper_bgcolor='white', plot_bgcolor='white', legend_title='Thresholds',
                          legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
        )

        for i in range(1, len(ssps)+1):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1, 
                             title_text='Year' if i == len(ssps) else '')
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1,
                             title_text=self.title_var if i == len(ssps)//2 + 1 else '')
        fig.show()

    def severity_sns(self, results, ssps, agg='mean', alpha=1):
        '''Plot severity comparison using seaborn.'''
        fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 5*len(ssps)), sharex=True, sharey=True)
        fig.suptitle('Severity Comparison: '+self.title, fontsize=16, y=0.95)

        lines, labels = [], []

        for i, ssp in enumerate(ssps):
            ax = axes[i] if len(ssps) > 1 else axes
            for j, (threshold, df) in enumerate(results.items()):
                data = self.check_df(df[df.index.get_level_values(0) == ssp])
                agg_data,p10,p90 = data.agg(agg, axis=1), data.quantile(0.1, axis=1), data.quantile(0.9, axis=1)
                
                line = sns.lineplot(x=data.index.get_level_values(1), y=agg_data, 
                                    label=threshold, alpha=alpha, color=self.colors[j % len(self.colors)], 
                                    linewidth=0.8, ax=ax)
                ax.fill_between(agg_data.index.get_level_values(1), p10, p90, 
                                color=self.colors[j % len(self.colors)], alpha=0.1)
                
                if i == 0:
                    lines.append(line.lines[-1])
                    labels.append(threshold)

            ax.set(title=ssp, ylabel=(self.title_var if i == len(ssps)//2 else ''), 
                   xlabel=('Year' if i == len(ssps)-1 else ''))
            ax.grid(True, color='lightgrey', linewidth=0.2)
            ax.legend().remove()

        fig.legend(lines, labels, title='Thresholds', bbox_to_anchor=(0.5, 0.98), 
                   loc='lower center', ncol=len(results), frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

    
    ##################################################
    ##     Variable Comparison (1 ssp per plot)     ##
    ##################################################

    def variable_comp(self, results, var, metrics, agg='mean', alpha=1):
        '''Plot variable comparison.'''
        ssps = list(pd.unique([idx[0] for df in results.values() for idx in df.index]))
        title = f'{self.center} - <variable> Per Year {self.month_title} Across CMIP6 Models'
        
        for threshold, df in results.items(): 
            for ssp in df.index.get_level_values('ssp').unique():
                for i, col in enumerate(var):
                    data = df.filter(regex=col).loc[ssp].agg(agg, axis=1)
                    sns.lineplot(data=data, label=metrics[col], color=self.colors[i], 
                                 linewidth=0.5, alpha=alpha)
                plt.title(f'{title}; {ssp}; {threshold}')
                plt.xlabel('Year')
                plt.ylabel('')    
                plt.legend(loc='center left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                plt.show()
    
    def all(self, results, plotly_, agg='mean', alpha=1):
        '''Plot all types of plots.'''
        ssps = list(pd.unique([idx[0] for df in results.values() for idx in df.index]))
        if plotly_:
            for threshold, df in results.items(): 
                self.dist_sns(df, agg, alpha, threshold)
                self.ts_plotly(df, agg, alpha, threshold)
            self.severity_plotly(results, ssps, agg=agg, alpha=alpha)
        else:
            for threshold, df in results.items(): 
                self.dist_sns(df, agg, alpha, threshold)
                self.ts_sns(df, agg, alpha, threshold)
            self.severity_sns(results, ssps, agg=agg, alpha=alpha)
            







        






        






        






        






        