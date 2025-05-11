# Standard Library Imports
import os
import json
import calendar
import warnings

# Third-Party Library Imports
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import linregress
import numpy as np

# Suppress Warnings
warnings.filterwarnings('ignore')

pio.templates.default = 'plotly_dark'
pio.templates[pio.templates.default].update({
    'layout': {
        #'autosize': True, 'width': 800, 'height': 300, 
        'font': {'size': 11},
        'hoverlabel': {'font_size': 11},
        'margin': dict(l=20, r=20, t=30, b=20),
        'xaxis': {'showgrid': True, 'gridwidth': 0.1, 'title': {'text': '', 'font': {'size': 11}}},
        'yaxis': {'showgrid': True, 'gridwidth': 0.1, 'title': {'text': '', 'font': {'size': 11}}},
        'title': {'text': '', 'font': {'size': 13}, 'x': 0.5, 'xanchor': 'center'},
        'annotations': [{'text': '', 'xref': "paper", 'yref': "paper", 'x': 0.5, 'y': -0.2,
                         'showarrow': False, 'font': {'size': 11}, 'xanchor': 'center', 'yanchor': 'bottom'}],
    },
    'data': {'scatter': [{'line': {'width': 0.8}}], 'scattergl': [{'line': {'width': 0.8}}]}
})

class Plot:
    ##################################################
    ##                   General                    ##
    ##################################################

    def __init__(self, center, months, var_title, var, title):
        self.center = center
        self.months = months
        self.var_title = var_title
        self.var = var
        self.colors = px.colors.qualitative.Plotly
        self.month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
        self.month_title = ' (' + ''.join(self.month_dict[month] for month in months) + ') ' if months and len(months)<12 else ''
        # self.title = f'{center} - {var_title} Per Year {self.month_title}Across CMIP6 Models'
        self.title = title + (self.month_title if months else '')
        self.y_labels = {'pr': 'Precipitation (mm)',
                         'rzsm': 'Root Zone Soil Moisture (m3/m3)',
                         'spi': 'SPI (JJA)',
                         'tasmax': 'Max Temperature (C)',
                        }

    ##################################################
    ##                   KDE Plot                   ##
    ##################################################

    def distribution(self, df, agg='mean', threshold=''):
        '''Plot distribution using seaborn.'''
        df = df.dropna(subset=f'{agg}{self.var}')
        hist_data = []
        group_labels = []
        for ssp, group in df.groupby('ssp'):
            hist_data.append(group[f'{agg}{self.var}'].dropna().values)
            group_labels.append(ssp)            
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=self.colors)
        fig.update_layout(title=f'{self.title} {threshold}', showlegend=True, 
                          xaxis_title=self.var_title.split('(')[0], yaxis_title='Density',
                          legend_traceorder='normal', autosize=False, width=1000, height=250)
        return fig

    ##################################################
    ##                  Time Series                 ##
    ##################################################

    def time_series(self, df, error_bar=False, agg='mean', threshold=''):
        vars_, ssps = df['variable'].unique(), df['ssp'].unique()
        color_map = {ssp: self.colors[i % len(self.colors)] for i, ssp in enumerate(sorted(ssps))}

        fig = make_subplots(rows=len(vars_), cols=1, shared_xaxes=True, vertical_spacing=0.01)

        for r, var in enumerate(vars_, 1):
            ddf = df[(df['variable'] == var)].sort_values('date')
            for ssp, group in ddf.groupby('ssp'):
                x, y, color = group['date'], group[agg], color_map[ssp]

                # Error band
                if error_bar and 'q90' in group and 'q10' in group:
                    fig.add_trace(go.Scatter(x=x.tolist() + x[::-1].tolist(),
                        y=group['q90'].tolist() + group['q10'][::-1].tolist(),
                        fill='toself', fillcolor=color, opacity=0.1, line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
                        showlegend=False, legendgroup=ssp), row=r, col=1)

                # Main line
                fig.add_trace(go.Scatter(x=x, y=y, name=ssp, line=dict(color=color), legendgroup=ssp,
                                         showlegend=(r == 1)), row=r, col=1)

                # Trend line
                t = (x - x.min()).dt.total_seconds() / (365.25 * 86400)
                if len(t) > 1:
                    slope, intercept, *_ = linregress(t, y)
                    fig.add_trace(go.Scatter(x=x, y=slope * t + intercept, name=f'{ssp}_trend',  line=dict(color=color, dash='dash'),
                                             showlegend=False, legendgroup=ssp), row=r, col=1)

            fig.update_yaxes(title_text=self.y_labels.get(var, var), row=r, col=1)

        fig.update_xaxes(title_text='Year', row=len(vars_), col=1)
        fig.update_layout(title=f'{self.title} {threshold}', height=250 * len(vars_), width=1000, showlegend=True)
        return fig

    
    ##################################################
    ##                   Severity                   ##
    ##################################################

    def severity(self, df, error_bar=False, agg='mean', alpha=1):
        '''Plot severity comparison using plotly.'''
        ssps, thres = sorted(filter(lambda x: 'ssp' in x, df['ssp'].unique())), df.threshold.unique()
        fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes='all', shared_yaxes='all', 
                            subplot_titles=ssps, vertical_spacing=0.03)

        for i, ssp in enumerate(ssps, start=1):
            for j, t in enumerate(thres):
                dd = df[(df.threshold == t) & (df.ssp.str.contains(f'{ssp}|historical'))].sort_values('date')
                x, y = dd.date, dd[f'{agg}{self.var}']
                color = self.colors[j % len(self.colors)]

                if error_bar:
                    fig.add_trace(go.Scatter(x=x.tolist() + x[::-1].tolist(), 
                        y=dd[f'q90{self.var}'].tolist() + dd[f'q10{self.var}'][::-1].tolist(),
                        showlegend=False, legendgroup=t, fill='toself', fillcolor=color, opacity=0.1,
                        line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip'), row=i, col=1)

                # Main line
                fig.add_trace(go.Scatter(x=x, y=y, name=t,line=dict(color=color), mode='lines',
                    opacity=alpha, legendgroup=t, showlegend=(i == 1)), row=i, col=1)

                # Trend line
                if len(x) > 1:
                    slope, intercept, *_ = linregress(x, y)
                    fig.add_trace(go.Scatter(x=x, y=slope * x + intercept, name=f'{t}_trend', line=dict(color=color, dash='dash'),
                                             showlegend=False, legendgroup=t), row=i, col=1)

        fig.update_layout(title=dict(text=f'Severity Comparison: {self.title}'), legend_title='Thresholds', autosize=False,
            width=1000, height=250 * len(ssps)
        )
        fig.update_annotations(font_size=11)
        for i in range(1, len(ssps)+1):
            fig.update_xaxes(row=i, col=1, title_text='Year' if i == len(ssps) else '')
            fig.update_yaxes(row=i, col=1, title_text=self.var_title if i == len(ssps) // 2 + 1 else '')
        return fig

    ##################################################
    ##     Metric Comparison (1 ssp per plot)     ##
    ##################################################

    def metric_comp(self, df, metrics:dict, agg='mean', alpha=1):
        '''Plot variable comparison using Plotly with optional error bands.'''
        ssps = sorted(filter(lambda x: 'ssp' in x, df['ssp'].unique()))

        fig = make_subplots(rows=len(ssps), cols=1, shared_xaxes='all', shared_yaxes='all', 
                            subplot_titles=ssps, vertical_spacing=0.03)
        for i, ssp in enumerate(ssps, 1):
            dd = df[(df.ssp.str.contains(f'{ssp}|historical'))]
            for v, (var, t) in enumerate(metrics.items()):
                fig.add_trace(go.Scatter(
                    x=dd.date, y=dd[f'{agg}{var}'], name=t, legendgroup=t, showlegend=(i==1), 
                    opacity=alpha, line=dict(color=self.colors[v]), mode='lines'), row=i, col=1)

        fig.update_layout(autosize=False, width=1000, height=250*len(ssps), legend_title='Variables',
            title=f'Variable Comparison: {self.center} - {agg.title()} Across CMIP6 Models {self.month_title}',)
        fig.update_annotations(font_size=11)
        for i in range(1, len(ssps)+1):
            fig.update_xaxes(row=i, col=1, title_text='Year' if i == len(ssps) else '')
            fig.update_yaxes(row=i, col=1, title_text='Value' if i == (len(ssps)//2)+1 else '')
        return fig

    