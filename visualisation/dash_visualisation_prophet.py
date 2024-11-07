import sys
import os
# Add the Modelling folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Modelling')))

# Add the data folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from dash import dash_table

from models import ETSModelWeekly, ProphetModelWeekly, MetricsCalculator

# Load your energy consumption data
data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
energy_consump_df = pd.read_csv(os.path.join(data_folder, 'total_energy_consumption_until2024_10.csv'))
  
energy_consump_df = pd.read_csv(os.path.join(data_folder, 'total_energy_consumption_until2024_10.csv'))
energy_consump_df['Datum-tijd tot'] = pd.to_datetime(energy_consump_df['Datum-tijd tot'], utc=True)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Short-Term (1 Day) THUAS Delft Building Energy Forecasting"),
    
    # Input to select start date
    html.Label('Select Start Date'),
    dcc.DatePickerSingle(
        id='start-date-picker',
        date='2023-02-01',
    ),
    
    # Forecast graph and metrics table (now first)
    html.Div([
        html.H2("Forecast"),
        dcc.Graph(id='forecast-graph'),
        html.Div(id='forecast-metrics-table', style={'text-align': 'center'}),
    ], style={'display': 'flex', 'flex-direction': 'column'}),

    # Validation graph and metrics table (now second)
    html.Div([
        html.H2("Validation"),
        dcc.Graph(id='model-validation-graph'),
        html.Div(id='validation-metrics-table', style={'text-align': 'center'}),
    ], style={'display': 'flex', 'flex-direction': 'column'}),
])

# Callback to update graphs and metrics based on inputs
@app.callback(
    [Output('model-validation-graph', 'figure'),
     Output('forecast-graph', 'figure'),
     Output('validation-metrics-table', 'children'),
     Output('forecast-metrics-table', 'children')],
    [Input('start-date-picker', 'date')]
)
def update_visualizations(start_date):
    start_date = pd.to_datetime(start_date)
    days_to_forecast = 1  # Hardcoded to 1 day
    
    # Load and run your models here
    # Initialize models
    ets_model = ETSModelWeekly(energy_consump_df, start_date, days_to_forecast)
    ets_pred_df = ets_model.run()
    
    prophet_model = ProphetModelWeekly(energy_consump_df, start_date, days_to_forecast)
    prophet_pred_df = prophet_model.run()
    
    # Validation metrics (on training data)
    train_df, test_df = ets_model.get_train_test_df()
    
    ets_model_df_val = ets_pred_df[0:len(train_df)]
    prophet_model_df_val = prophet_pred_df[0:len(train_df)]
    
    metrics_calculator = MetricsCalculator(train_df, ets_model_df_val)
    metrics_ets = metrics_calculator.calculate_metrics()
    
    metrics_calculator = MetricsCalculator(train_df, prophet_model_df_val)
    metrics_prophet = metrics_calculator.calculate_metrics()

    # Forecast metrics (on test data)
    ets_model_df_forecast = ets_pred_df[len(train_df):]
    prophet_model_df_forecast = prophet_pred_df[len(train_df):]
    
    metrics_calculator = MetricsCalculator(test_df, ets_model_df_forecast)
    forecast_metrics_ets = metrics_calculator.calculate_metrics()
    
    metrics_calculator = MetricsCalculator(test_df, prophet_model_df_forecast)
    forecast_metrics_prophet = metrics_calculator.calculate_metrics()

    # Create figures with prediction bands (Validation)
    validation_fig = go.Figure()
    validation_fig.add_trace(go.Scatter(x=train_df['Datum-tijd tot'], y=train_df['Elektriciteit Consumptie (kWh)'], mode='lines', name='Actual'))
    validation_fig.add_trace(go.Scatter(x=ets_model_df_val['Datum-tijd tot'], y=ets_model_df_val['Elektriciteit Consumptie (kWh)'], mode='lines', name='ETS'))
    validation_fig.add_trace(go.Scatter(x=prophet_model_df_val['Datum-tijd tot'], y=prophet_model_df_val['Elektriciteit Consumptie (kWh)'], mode='lines', name='Prophet'))
    
    # Add prediction bands for ETS
    validation_fig.add_trace(go.Scatter(
        x=ets_model_df_val['Datum-tijd tot'],
        y=ets_model_df_val['Consumption_Lower'],
        mode='lines',
        line=dict(width=0),
        name='ETS Lower Band',
        showlegend=False
    ))
    validation_fig.add_trace(go.Scatter(
        x=ets_model_df_val['Datum-tijd tot'],
        y=ets_model_df_val['Consumption_Upper'],
        mode='lines',
        fill='tonexty',
        line=dict(width=0),
        name='ETS Upper Band',
        showlegend=False
    ))
    
    # Create figures with prediction bands (Forecast)
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=test_df['Datum-tijd tot'], y=test_df['Elektriciteit Consumptie (kWh)'], mode='lines', name='Actual'))
    forecast_fig.add_trace(go.Scatter(x=ets_model_df_forecast['Datum-tijd tot'], y=ets_model_df_forecast['Elektriciteit Consumptie (kWh)'], mode='lines', name='ETS'))
    forecast_fig.add_trace(go.Scatter(x=prophet_model_df_forecast['Datum-tijd tot'], y=prophet_model_df_forecast['Elektriciteit Consumptie (kWh)'], mode='lines', name='Prophet'))
    
    # Add prediction bands for ETS in forecast
    forecast_fig.add_trace(go.Scatter(
        x=ets_model_df_forecast['Datum-tijd tot'],
        y=ets_model_df_forecast['Consumption_Lower'],
        mode='lines',
        line=dict(width=0),
        name='ETS Lower Band',
        showlegend=False
    ))
    forecast_fig.add_trace(go.Scatter(
        x=ets_model_df_forecast['Datum-tijd tot'],
        y=ets_model_df_forecast['Consumption_Upper'],
        mode='lines',
        fill='tonexty',
        line=dict(width=0),
        name='ETS Upper Band',
        showlegend=False
    ))

    # Add prediction bands for Prophet in forecast
    forecast_fig.add_trace(go.Scatter(
        x=prophet_model_df_forecast['Datum-tijd tot'],
        y=prophet_model_df_forecast['Consumption_Lower'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(98, 182, 98, 0.4)',  # Darker green with less transparency
        name='Prophet Lower Band',
        showlegend=False
    ))
    forecast_fig.add_trace(go.Scatter(
        x=prophet_model_df_forecast['Datum-tijd tot'],
        y=prophet_model_df_forecast['Consumption_Upper'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(98, 182, 98, 0.4)',  # Darker green with less transparency
        line=dict(width=0),
        name='Prophet Upper Band',
        showlegend=False
    ))

    # Create validation metrics table
    validation_table = dash_table.DataTable(
        columns=[
            {"name": "Model", "id": "model"},
            {"name": "RMSE", "id": "rmse"},
            {"name": "MAD", "id": "mad"},
            {"name": "R2", "id": "r2"},
            {"name": "MAPE", "id": "mape"}
        ],
        data=[
            {
                "model": "ETS",
                "rmse": f"{metrics_ets['RMSE']:.2f}",
                "mad": f"{metrics_ets['MAD']:.2f}",
                "r2": f"{metrics_ets['R2']:.2f}",
                "mape": f"{metrics_ets['MAPE']:.2f}"
            },
            {
                "model": "Prophet",
                "rmse": f"{metrics_prophet['RMSE']:.2f}",
                "mad": f"{metrics_prophet['MAD']:.2f}",
                "r2": f"{metrics_prophet['R2']:.2f}",
                "mape": f"{metrics_prophet['MAPE']:.2f}"
            }
        ],
        style_table={'width': '50%'},
        style_cell={'textAlign': 'center'}
    )

    # Create forecast metrics table
    forecast_table = dash_table.DataTable(
        columns=[
            {"name": "Model", "id": "model"},
            {"name": "RMSE", "id": "rmse"},
            {"name": "MAD", "id": "mad"},
            {"name": "R2", "id": "r2"},
            {"name": "MAPE", "id": "mape"}
        ],
        data=[
            {
                "model": "ETS",
                "rmse": f"{forecast_metrics_ets['RMSE']:.2f}",
                "mad": f"{forecast_metrics_ets['MAD']:.2f}",
                "r2": f"{forecast_metrics_ets['R2']:.2f}",
                "mape": f"{forecast_metrics_ets['MAPE']:.2f}"
            },
            {
                "model": "Prophet",
                "rmse": f"{forecast_metrics_prophet['RMSE']:.2f}",
                "mad": f"{forecast_metrics_prophet['MAD']:.2f}",
                "r2": f"{forecast_metrics_prophet['R2']:.2f}",
                "mape": f"{forecast_metrics_prophet['MAPE']:.2f}"
            }
        ],
        style_table={'width': '50%'},
        style_cell={'textAlign': 'center'}
    )

    return validation_fig, forecast_fig, validation_table, forecast_table

if __name__ == '__main__':
    app.run_server(
        debug=True,  # Set to False in production
        host='0.0.0.0',  # Required for Docker
        port=3030
    )
