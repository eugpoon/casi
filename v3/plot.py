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
plt.rcParams['axes.linewidth'] = 0.1
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.2
plt.rcParams['legend.loc'] = 'upper left'


class Plot:
    ##################################################
    ##                   General                    ##
    ##################################################

    def __init__(self, center, months, title_var, var):
        self.center = center
        self.months = months
        self.title_var = title_var
        self.var = var
        self.colors = px.colors.qualitative.Plotly
        self.month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
        self.month_title = '(' + ''.join(self.month_dict[month] for month in months) + ') ' if months else ''
        self.title = f'{center} - {title_var} Per Year {self.month_title}Across CMIP6 Models'

    def check_df(self, df, var=None):
        '''Filter dataframe based on variable.'''
        temp = df.filter(regex='ssp|date|threshold|scale')
        df = df.filter(regex=var if var else self.var)
        if df.empty:
            raise ValueError('Empty dataframe')
        return pd.concat([temp, df], axis=1), df.columns

    def get_agg(self, df, cols, l=0.1, u=0.9, agg='mean'):
        df = df.set_index('date')
        return df, df[cols].agg(agg, axis=1), df[cols].quantile(0.1, axis=1), df[cols].quantile(0.9, axis=1)
    
    ##################################################
    ##                   KDE Plot                   ##
    ##################################################

    def dist_sns(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot distribution using seaborn.'''
        df, cols = self.check_df(df)
        for i, ssp in enumerate(df.ssp.unique()):
            sns.kdeplot(data=df[df.ssp==ssp][cols].agg(agg, axis=1),
                label=f'{ssp}_{agg}', color=self.colors[i % len(self.colors)], linewidth=0.5, alpha=alpha)
        plt.title(f'{self.title} {threshold}')
        plt.xlabel(self.title_var)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()

    ##################################################
    ##                  Time Series                 ##
    ##################################################

    def ts_plotly(self, df, agg='mean', alpha=1, threshold=''):
        '''Plot time series using plotly.'''
        df, cols = self.check_df(df)
        fig = go.Figure()
        for i, ssp in enumerate(df.ssp.unique()):
            data, agg_data, p10, p90 = self.get_agg(df[df.ssp==ssp], cols)
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
        df, cols = self.check_df(df)
        for i, ssp in enumerate(df.ssp.unique()):
            data = df[df.ssp==ssp].set_index('date')[cols].agg(agg, axis=1)
            sns.lineplot(data=data, label=ssp, color=self.colors[i], linewidth=0.5, alpha=alpha)
        plt.title(f'{self.title} {threshold}')
        plt.xlabel('Year')
        plt.ylabel(self.title_var)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
        
        data, agg_data, p10, p90 = self.get_agg(df, cols)
        ymin = min(p10.min(), p90.min(), agg_data.min())
        ymax = max(p10.max(), p90.max(), agg_data.max())
        for i, ssp in enumerate(df.ssp.unique()):
            data, agg_data, p10, p90 = self.get_agg(df[df.ssp==ssp], cols)
            sns.lineplot(data=agg_data, label=ssp, color=self.colors[i], linewidth=0.8, alpha=alpha)
            plt.fill_between(agg_data.index, p10, p90, color=self.colors[i], alpha=0.1)
            plt.title(f'{self.title} {threshold}')
            plt.xlabel('Year')
            plt.ylabel(self.title_var)
            plt.ylim(ymin, ymax)
            plt.legend(bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()
            
    ##################################################
    ##    Threshold Comparison (1 ssp per plot)     ##
    ##################################################

    def severity_plotly(self, results, agg='mean', alpha=1):
        '''Plot severity comparison using plotly.'''
        ssps, thres = results.ssp.unique(), results.threshold.unique()
        fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes='all', shared_yaxes='all', 
                            subplot_titles=ssps, vertical_spacing=0.05)
                
        for i, ssp in enumerate(ssps, start=1):
            for j, t in enumerate(thres):
                df = results[(results.threshold==t) & (results.ssp==ssp)]
                data, cols = self.check_df(df)
                data, agg_data, p10, p90 = self.get_agg(data, cols)

                # Add error range
                fig.add_trace(go.Scatter(
                    x=(agg_data.index.tolist() + agg_data.index[::-1].tolist()),
                    y=p90.tolist() + p10[::-1].tolist(), showlegend=False, legendgroup=t,
                    fill='toself', fillcolor=self.colors[j % len(self.colors)], opacity=0.2, 
                    line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'), row=i, col=1)
                
                fig.add_trace(go.Scatter(
                    x=agg_data.index, y=agg_data, name=t,
                    line=dict(color=self.colors[j % len(self.colors)], width=0.8), mode='lines',
                    opacity=alpha, legendgroup=t, showlegend=(i == 1)), row=i, col=1)

        fig.update_layout(title=f'Severity Comparison: {self.title}', width=1000, height=200*len(ssps),
                          margin=dict(l=50, r=20, t=120, b=50),
                          paper_bgcolor='white', plot_bgcolor='white', legend_title='Thresholds',
                          legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))

        for i in range(1, len(ssps)+1):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1, 
                             title_text='Year' if i==len(ssps) else '')
            fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=0.1, row=i, col=1,
                             title_text=self.title_var if i==len(ssps)//2+1 else '')
        fig.show()

    def severity_sns(self, df, agg='mean', alpha=1):
        '''Plot severity comparison using seaborn.'''
        ssps, thres = df.ssp.unique(), df.threshold.unique()
        fig, axes = plt.subplots(len(ssps), 1, figsize=(15, 5*len(ssps)), sharex=True, sharey=True)
        fig.suptitle(f'Severity Comparison: {self.title}', fontsize=16, y=0.95)

        for i, ssp in enumerate(ssps):
            ax = axes[i] if len(ssps) > 1 else axes
            data, cols = self.check_df(df[(df.ssp==ssp)])
            data, agg_data, p10, p90 = self.get_agg(data, cols)
            sns.lineplot(x=data.index, y=agg_data, hue=data.threshold, palette=self.colors,
                         alpha=alpha, linewidth=alpha, ax=ax)
            for j, t in enumerate(thres):
                mask = data.threshold == t
                ax.fill_between(x=data[mask].index, y1=p10[mask], y2=p90[mask], color=self.colors[j], alpha=0.1)
            ax.set(title=ssp, ylabel=(self.title_var if i == len(ssps)//2 else ''), 
                   xlabel=('Year' if i == len(ssps)-1 else ''))
            ax.grid(True, color='lightgrey', linewidth=0.2)
            ax.get_legend().set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Thresholds', bbox_to_anchor=(0.5, 0.88), 
                   loc='lower center', ncol=len(df), frameon=False, fontsize=14, title_fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
    
    ##################################################
    ##     Variable Comparison (1 ssp per plot)     ##
    ##################################################

    def variable_comp(self, results, var:list, metrics:dict, agg='mean', alpha=1):
        '''Plot variable comparison.'''
        title = f'{self.center} - <variable> Per Year {self.month_title} Across CMIP6 Models'
        ssps, thres = results.ssp.unique(), results.threshold.unique()
        for t in thres:
            ymin, ymax = [], []
            df1 = results[results.threshold==t]
            for col in var:
                df, cols = self.check_df(df1, col)
                agg_data = df[cols].agg(agg, axis=1)
                ymin.append(min(agg_data)), ymax.append(max(agg_data))
            
            for ssp in ssps:
                df2 = df1[df1.ssp==ssp]
                for i, col in enumerate(var):
                    df3, cols = self.check_df(df2, col)
                    sns.lineplot(x=df3.date, y=df3[cols].agg(agg, axis=1),  
                                 label=metrics[col], color=self.colors[i], linewidth=0.5, alpha=alpha)
                plt.title(f'{title}; {ssp}; {t}')
                plt.xlabel('Year')
                plt.ylabel('')    
                plt.ylim(min(ymin), max(ymax))
                plt.legend(title='Variable', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                plt.show()
                
    ##################################################
    ##    Time Scale Comparison (1 ssp per plot)    ##
    ##################################################
    
    def scale_comp(self, results:pd.DataFrame(), metrics:dict, threshold:str='', agg:str='mean', alpha:float=1):
        for col, title_var in metrics.items():
            df, cols = self.check_df(results, col)
            agg_data = df[cols].agg(agg, axis=1)
            for i, ssp in enumerate(df.ssp.unique()):
                sns.lineplot(x=df.date, y=df[df.ssp==ssp][cols].agg(agg, axis=1), 
                             hue=df.scale, palette=self.colors, linewidth=0.8, alpha=alpha)
                plt.title(f'{self.title}; {ssp}; {threshold}')
                plt.xlabel('Year')
                plt.ylabel(title_var)   
                plt.ylim(min(agg_data), max(agg_data))
                plt.legend(title='SPI Time Scale', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                plt.show()
    
    ##################################################
    ##               Plot 1 Variable                ##
    ##################################################
    
    def all(self, results:pd.DataFrame(), plotly_:bool, agg:str='mean', alpha:float=1):
        '''Plot all types of plots.'''
        ts_plot = self.ts_plotly if plotly_ else self.ts_sns
        severity_plot = self.severity_plotly if plotly_ else self.severity_sns
        for t in results.threshold.unique():
            df = results[results.threshold == t].drop(columns='threshold')
            self.dist_sns(df, agg, alpha, t)
            ts_plot(df, agg, alpha, t)
        severity_plot(results, agg, alpha)







        






        






        






        






        


