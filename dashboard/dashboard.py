'''
terrible. lotta redundant code
'''

import os
import json
import warnings
import pandas as pd
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plot import Plot

warnings.filterwarnings('ignore')

##################################################
##                 Initialize                   ##
##################################################
with open('events_vars.json', 'r') as file:
    VAR = json.load(file)

PATH = 'compound_results'

def load_data_files():
    '''Load data files into DataFrames'''
    files = sorted([os.path.join(PATH, f) for f in os.listdir(PATH) if f.endswith(('.csv', '.parquet'))])
    
    return {'var_aggs': {'_'.join(os.path.basename(f).split('_')[:2]): pd.read_csv(f) 
                        for f in files if '_variables.csv' in f},
            'metric_aggs': {'_'.join(os.path.basename(f).split('_')[:2]): pd.read_csv(f)
                           for f in files if '_metrics' in f},
            'daily_aggs': {'_'.join(os.path.basename(f).split('_')[:2]): pd.read_parquet(f)
                          for f in files if '_daily.parquet' in f}}

def metric_titles(event):
    return {'_day_total': f'Total {event} Days ',
            '_event_total': f'Total {event} ',
            '_sequence_total': f'Total {event} Sequences ',
            '_duration_mean': f'Average {event} Duration ',
            '_duration_max': f'Max {event} Duration '}

def variable_titles(event, var):
    return {'spi': (f'Average SPI ({var[event]['scale']} Day Accumulation; gamma_n={var[event]['gamma_n']})'),
            'tasmax': 'Average Max Temperature',
            'pr': (f'Average Precipitation ({var['CFE']['scale']} Day Accumulation)'
                  if event == 'CFE' else 'Average Precipitation'),
            'rzsm': 'Average Root Zone Soil Moisture'}

##################################################
##              Variable Analysis               ##
##################################################
class VariableAnalysis:
    def __init__(self):
        '''Initialize dashboard with loaded data'''
        self.data = load_data_files()
        self.app = JupyterDash(__name__)
        self._setup_layout()
        self._register_callbacks()

    def _setup_layout(self):
        '''Configure dashboard layout'''
        centers = sorted({f.split('_')[0] for f in self.data['var_aggs'].keys()})
        
        self.app.layout = html.Div([html.Div([
                dcc.Dropdown(id='center-dropdown', options=[{'label': c, 'value': c} for c in centers],
                    value=centers[0], style={'minWidth': '200px'}),
                dcc.Dropdown(id='event-dropdown', style={'minWidth': '200px'}),
                dcc.Dropdown(id='variable-dropdown', style={'minWidth': '200px'}),
                dcc.Checklist(id='error-bar-checkbox', value=[],
                    options=[{'label': 'Show Error Bars', 'value': 'show_error'}], style={'marginLeft': '20px'}
                )
            ], style={'display': 'flex', 'gap': '5px', 'justifyContent': 'space-between',
                      'alignItems': 'center'
            }),
            dcc.Graph(id='dist-plot'), dcc.Graph(id='time-series-plot')
        ])

    def _register_callbacks(self):
        '''Register Dash callbacks'''
        
        @self.app.callback(
            [Output('event-dropdown', 'options'), Output('event-dropdown', 'value')],
            [Input('center-dropdown', 'value')]
        )
        def update_event_dropdown(selected_center):
            events = sorted({k.split('_')[1] for k in self.data['var_aggs'].keys() 
                             if k.startswith(f'{selected_center}_')})
            return [{'label': e, 'value': e} for e in events], (events[0] if events else None)

        @self.app.callback(
            [Output('variable-dropdown', 'options'), Output('variable-dropdown', 'value')],
            [Input('center-dropdown', 'value'), Input('event-dropdown', 'value')]
        )
        def update_variable_dropdown(center, event):
            if not center or not event:
                return [], None
            try:
                variables = sorted({col.split('_')[-1] 
                    for col in self.data['var_aggs'][f'{center}_{event}'].columns  if '_' in col})
                return [{'label': v, 'value': v} for v in variables], (variables[0] if variables else None)
            except KeyError:
                return [], None

        @self.app.callback(
            [Output('dist-plot', 'figure'), Output('time-series-plot', 'figure')],
            [Input('center-dropdown', 'value'), Input('event-dropdown', 'value'),
             Input('variable-dropdown', 'value'), Input('error-bar-checkbox', 'value')]
        )
        def update_plots(center, event, variable, error_bar):
            if None in [center, event, variable]:
                return go.Figure(), go.Figure()
            try:
                data = self.data['var_aggs'][f'{center}_{event}']
                p = Plot(center, VAR[event]['months'], variable_titles(event, VAR)[variable], f'_{variable}')
                return p.distribution(data), p.time_series(data, bool(error_bar))
            except (KeyError, TypeError) as e:
                return go.Figure(), go.Figure()

    def run(self):
        self.app.run(mode='inline', port=8050, jupyter_height=700, jupyter_width='100%')

##################################################
##              Severity Analysis               ##
##################################################
class SeverityAnalysis:
    def __init__(self):
        '''Initialize metric analysis dashboard'''
        self.data = load_data_files()
        self.app = JupyterDash(__name__)
        self.metric_vars = list(metric_titles('').keys())
        self._setup_layout()
        self._register_callbacks()

    def _setup_layout(self):
        '''Configure dashboard layout'''
        centers = sorted({f.split('_')[0] for f in self.data['metric_aggs'].keys()})
        
        self.app.layout = html.Div([html.Div([
                dcc.Dropdown(id='center-dropdown', options=[{'label': c, 'value': c} for c in centers],
                    value=centers[0], style={'minWidth': '200px'}),
                dcc.Dropdown(id='event-dropdown', style={'minWidth': '200px'}),
                dcc.Dropdown(id='threshold-dropdown', style={'minWidth': '200px'}),
                dcc.Dropdown(id='variable-dropdown',options=[{'label': v, 'value': v} for v in self.metric_vars],
                    value=self.metric_vars[0],style={'minWidth': '200px'}
                ),
                dcc.Checklist(id='error-bar-checkbox', value=[],
                    options=[{'label': 'Show Error Bars', 'value': 'show_error'}], style={'marginLeft': '20px'})
            ], style={'display': 'flex', 'gap': '5px', 'justifyContent': 'space-between', 'alignItems': 'center'
            }),
            dcc.Graph(id='dist-plot'), dcc.Graph(id='time-series-plot')
        ])

    def _register_callbacks(self):
        '''Register Dash callbacks for metric analysis'''
        
        @self.app.callback(
            [Output('event-dropdown', 'options'), Output('event-dropdown', 'value')],
            [Input('center-dropdown', 'value')]
        )
        def update_event_dropdown(selected_center):
            events = sorted({k.split('_')[1] for k in self.data['metric_aggs'].keys() 
                           if k.startswith(f'{selected_center}_')})
            return [{'label': e, 'value': e} for e in events], (events[0] if events else None)

        @self.app.callback(
            [Output('threshold-dropdown', 'options'), Output('threshold-dropdown', 'value')],
            [Input('center-dropdown', 'value'), Input('event-dropdown', 'value')]
        )
        def update_threshold_dropdown(center, event):
            if not center or not event:
                return [], None
            try:
                thres = sorted(self.data['metric_aggs'][f'{center}_{event}'].threshold.unique())
                return [{'label': str(t), 'value': t} for t in thres], (thres[0] if thres else None)
            except KeyError:
                return [], None

        @self.app.callback(
            [Output('dist-plot', 'figure'), Output('time-series-plot', 'figure')],
            [Input('center-dropdown', 'value'), Input('event-dropdown', 'value'),
             Input('threshold-dropdown', 'value'), Input('variable-dropdown', 'value'),
             Input('error-bar-checkbox', 'value')]
        )
        def update_plots(center, event, threshold, variable, error_bar):
            if None in [center, event, threshold, variable]:
                return go.Figure(), go.Figure()
            try:
                data = self.data['metric_aggs'][f'{center}_{event}']
                filtered_data = data[data.threshold == threshold].drop(columns='threshold')
                titles = metric_titles(event)
                p = Plot(center, VAR[event]['months'], titles[variable], variable)
                return p.distribution(filtered_data, threshold=threshold), p.severity(data, bool(error_bar))
            except (KeyError, TypeError) as e:
                return go.Figure(), go.Figure()

    def run(self):
        self.app.run(mode='inline', port=8051, jupyter_height=1200, jupyter_width='100%')

##################################################
##             Metric Comparison              ##
##################################################
class MetricComparison:
    def __init__(self):
        '''Initialize metric analysis dashboard'''
        self.data = load_data_files()
        self.app = JupyterDash(__name__)
        self._setup_layout()
        self._register_callbacks()

    def _setup_layout(self):
        '''Configure dashboard layout'''
        centers = sorted({f.split('_')[0] for f in self.data['metric_aggs'].keys()})
        
        self.app.layout = html.Div([html.Div([
                dcc.Dropdown(id='center-dropdown', options=[{'label': c, 'value': c} for c in centers],
                    value=centers[0], style={'minWidth': '200px'}),
                dcc.Dropdown(id='event-dropdown', style={'minWidth': '200px'}),
                dcc.Dropdown(id='threshold-dropdown', style={'minWidth': '200px'}),                
            ], style={'display': 'flex', 'gap': '5px', 'justifyContent': 'space-between', 'alignItems': 'center'
            }),
            dcc.Graph(id='time-series-plot')
        ])

    def _register_callbacks(self):
        '''Register Dash callbacks for metric analysis'''
        
        @self.app.callback(
            [Output('event-dropdown', 'options'), Output('event-dropdown', 'value')],
            [Input('center-dropdown', 'value')]
        )
        def update_event_dropdown(selected_center):
            events = sorted({k.split('_')[1] for k in self.data['metric_aggs'].keys() 
                           if k.startswith(f'{selected_center}_')})
            return [{'label': e, 'value': e} for e in events], (events[0] if events else None)

        @self.app.callback(
            [Output('threshold-dropdown', 'options'), Output('threshold-dropdown', 'value')],
            [Input('center-dropdown', 'value'), Input('event-dropdown', 'value')]
        )
        def update_threshold_dropdown(center, event):
            if not center or not event:
                return [], None
            try:
                thres = sorted(self.data['metric_aggs'][f'{center}_{event}'].threshold.unique())
                return [{'label': str(t), 'value': t} for t in thres], (thres[0] if thres else None)
            except KeyError:
                return [], None

        @self.app.callback(
            Output('time-series-plot', 'figure'), [Input('center-dropdown', 'value'), 
            Input('event-dropdown', 'value'), Input('threshold-dropdown', 'value')]
        )
        def update_plots(center, event, threshold):
            if None in [center, event, threshold]:
                return go.Figure()
            try:
                data = self.data['metric_aggs'][f'{center}_{event}']
                filtered_data = data[data.threshold == threshold].drop(columns='threshold')
                p = Plot(center, VAR[event]['months'], '', '')
                return p.metric_comp(filtered_data, metric_titles(event))
            except (KeyError, TypeError) as e:
                return go.Figure()

    def run(self):
        self.app.run(mode='inline', port=8052, jupyter_height=800, jupyter_width='100%')


