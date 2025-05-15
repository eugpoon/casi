"""
Interactive dashboards for visualizing compound event data using CMIP6 ensemble means.

Usage:
    Run in a Jupyter Notebook to explore time series, severity, and threshold-based metrics.
"""

import pandas as pd, json, ipywidgets as widgets
from IPython.display import display, clear_output
from plot import Plot

# Load ensemble raw input and processed compound event results
mme_raw = pd.read_parquet('../data/compound_mme_raw.parquet')
mme_results = pd.read_parquet('../data/compound_mme_results.parquet')

# Load event configuration (thresholds, active months, etc.)
with open('../data/compound_events.json') as f:
    VAR = json.load(f)

metric_titles = lambda e: {
    '_day_total': f'Total {e} Days',
    '_event_total': f'Total {e}',
    '_sequence_total': f'Total {e} Sequences',
    '_duration_mean': f'Average {e} Duration',
    '_duration_max': f'Max {e} Duration'
}

# Styled button widget
def base_button(desc): 
    return widgets.Button(description=desc, layout=widgets.Layout(width='97%'),
                          style=dict(button_color='#525cb2ff', font_weight='bold', text_color='white'))

# Dashboard 1: Time series of climate variables
def dashboard_1():
    center, time, timeagg, modelagg = [widgets.Dropdown(options=opt, description=desc) for opt, desc in [
        (mme_raw['center'].unique(), 'Center:'), 
        (['Yearly', 'Monthly'], 'Time:'), 
        (['mean', 'median', 'sum'], 'Time Agg:'), 
        (['mean', 'median'], 'Model Agg:')
    ]]
    months = widgets.SelectMultiple(options=[str(i) for i in range(1, 13)], value=[str(i) for i in range(1, 13)], description='Months:')
    error = widgets.ToggleButton(value=False, description='Show Error Bars', button_style='info', layout=widgets.Layout(width='97%'), style={'text_color': 'white'})
    out, btn = widgets.Output(), base_button("Generate/Update Plot")

    def update(_=None):
        with out:
            clear_output(wait=True)
            df = mme_raw[mme_raw['center'] == center.value]
            if df.empty: return print("No data available.")
            m = [int(i) for i in months.value]
            df = df[df['date'].dt.month.isin(m)].assign(date=df['date'].dt.to_period(time.value[0]).dt.to_timestamp())
            df = df.groupby(['center', 'variable', 'ssp', 'date']).agg(timeagg.value).reset_index()
            Plot(center=center.value, months=m, var_title='', var='',
                 title=f'{center.value} – {time.value} {timeagg.value.title()} of Ensemble {modelagg.value.title()}'
            ).time_series(df, agg=modelagg.value, error_bar=error.value).show()

    btn.on_click(update)
    display(widgets.HBox([
        widgets.VBox([center, error, btn], layout=widgets.Layout(width='33%')),
        widgets.VBox([modelagg, time, timeagg], layout=widgets.Layout(width='33%')),
        widgets.VBox([months], layout=widgets.Layout(width='33%'))
    ]), out)

# Dashboard 2: Severity of a selected compound event using one metric
def dashboard_2():
    center, event = [widgets.Dropdown(options=mme_results[col].unique(), description=col.capitalize() + ':') for col in ['center', 'event']]
    model = widgets.Dropdown(options=['mean', 'median'], description='Model Agg:')
    metric = widgets.Dropdown(options=list(metric_titles('').keys()), description='Metric:')
    error, out, btn = widgets.ToggleButton(value=False, description='Show Error Bars', button_style='info', layout=widgets.Layout(width='97%'), style={'text_color': 'white'}), widgets.Output(), base_button("Generate/Update Plot")

    def update(_=None):
        with out:
            clear_output(wait=True)
            df = mme_results[(mme_results['center'] == center.value) & (mme_results['event'] == event.value)]
            if df.empty: return print("No data available.")
            Plot(center=center.value, months=VAR[event.value]['months'],
                 var_title=metric_titles(event.value)[metric.value], var=metric.value,
                 title=f'{center.value} – {model.value.title()} {metric_titles(event.value)[metric.value]} Per Year'
            ).severity(df, agg=model.value, error_bar=error.value).show()

    btn.on_click(update)
    display(widgets.HBox([
        widgets.VBox([error, btn], layout=widgets.Layout(width='33%')),
        widgets.VBox([center, event], layout=widgets.Layout(width='33%')),
        widgets.VBox([model, metric], layout=widgets.Layout(width='33%'))
    ]), out)

# Dashboard 3: Compare multiple metrics for one event and threshold level
def dashboard_3():
    center = widgets.Dropdown(options=mme_results['center'].unique(), description='Center:')
    event = widgets.Dropdown(options=mme_results['event'].unique(), description='Event:')
    threshold = widgets.Dropdown(description='Threshold:')
    model = widgets.Dropdown(options=['mean', 'median'], description='Model Agg:')
    btn, out = base_button("Generate/Update Plot"), widgets.Output()

    def update_thresh(*_):
        thresholds = mme_results[mme_results['event'] == event.value]['threshold'].unique()
        threshold.options = thresholds
        if thresholds.size > 0:
            threshold.value = thresholds[0]

    event.observe(update_thresh, names='value')
    update_thresh()  # initialize on first load

    def update(_=None):
        with out:
            clear_output(wait=True)
            df = mme_results[(mme_results['center'] == center.value) & (mme_results['event'] == event.value) &
                             (mme_results['threshold'] == threshold.value)]
            if df.empty:
                print("No data available.")
                return
            title = f'{center.value} – {model.value.title()} Per Year'
            Plot(center=center.value, months=VAR[event.value]['months'], var_title='', var='', title=title
                ).metric_comp(df, metrics=metric_titles(event.value), agg=model.value).show()

    btn.on_click(update)
    display(widgets.HBox([
        widgets.VBox([btn], layout=widgets.Layout(width='33%')),
        widgets.VBox([center, event], layout=widgets.Layout(width='33%')),
        widgets.VBox([model, threshold], layout=widgets.Layout(width='33%'))
    ]), out)