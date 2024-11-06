'''
Created by: Amey Vasulkar
Email: anvasulkar@hhs.nl
Date: 23-10-2024

This file contains modules/functions for different models used for prediction.
'''
#%%
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os
import sys
class ETSModelWeekly:
    def __init__(self, energy_consump_df, start_date, days_to_forecast):
        self.energy_consump_df = energy_consump_df
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.days_to_forecast = days_to_forecast
        self.train_df = None
        self.test_df = None
        self.model = None
        self.model_fit = None

    def split_data(self):
        split_date = self.start_date + pd.DateOffset(days=self.days_to_forecast-1)
        self.train_df = self.energy_consump_df[self.energy_consump_df['Datum-tijd tot'] < split_date]
        self.test_df = self.energy_consump_df[self.energy_consump_df['Datum-tijd tot'] >= split_date].head(24 * self.days_to_forecast)

    def fit(self):
        # Get the absolute path to the directory containing this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the CSV file
        param_path = os.path.join(script_dir, 'ETS_param_values.csv')
        initial_param_df = pd.read_csv(param_path)
        initial_level = initial_param_df[initial_param_df['param_names'] == 'initial_level']['param_val'].values[0]
        initial_trend = initial_param_df[initial_param_df['param_names'] == 'initial_trend']['param_val'].values[0]
        initial_seasonal = initial_param_df['param_val'].iloc[5:].values

        self.model = ETSModel(self.train_df['Elektriciteit Consumptie (kWh)'],
                              initialization_method='known',
                              initial_level=initial_level,
                              initial_trend=initial_trend,
                              initial_seasonal=initial_seasonal,
                              seasonal_periods=24 * 7,
                              trend='add',
                              seasonal='additive')
        self.model_fit = self.model.fit()

    def predict(self):
        prediction = self.model_fit.get_prediction(start=1, end=len(self.test_df) + len(self.train_df))
        pred_df = prediction.summary_frame(alpha=0.80)
        total_time_array = np.concatenate((self.train_df['Datum-tijd tot'].values, self.test_df['Datum-tijd tot'].values))
        pred_df['Datum-tijd tot'] = total_time_array
        pred_df.rename(columns={
            'mean': 'Elektriciteit Consumptie (kWh)',
            'pi_lower': 'Consumption_Lower',
            'pi_upper': 'Consumption_Upper'
        }, inplace=True)
        return pred_df

    def run(self):
        self.split_data()
        self.fit()
        return self.predict()

    def get_train_test_df(self):
        return self.train_df, self.test_df


class ProphetModelWeekly:
    def __init__(self, energy_consump_df, start_date, days_to_forecast):
        self.energy_consump_df = energy_consump_df
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.days_to_forecast = days_to_forecast
        self.train_df = None
        self.test_df = None
        self.model = None
        self.forecast = None

    def split_data(self):
        split_date = self.start_date + pd.DateOffset(days=self.days_to_forecast-1)
        self.train_df = self.energy_consump_df[self.energy_consump_df['Datum-tijd tot'] < split_date]
        self.test_df = self.energy_consump_df[self.energy_consump_df['Datum-tijd tot'] >= split_date].head(24 * self.days_to_forecast)
        # Ensure time zone-aware to avoid warnings
        self.train_df['Datum-tijd tot'] = pd.to_datetime(self.train_df['Datum-tijd tot'], utc=True)
        self.test_df['Datum-tijd tot'] = pd.to_datetime(self.test_df['Datum-tijd tot'], utc=True)  
        # Now you can safely remove the time zone information
        self.train_df.loc[:, 'Datum-tijd tot'] = self.train_df['Datum-tijd tot'].dt.tz_convert(None)
        self.test_df.loc[:, 'Datum-tijd tot'] = self.test_df['Datum-tijd tot'].dt.tz_convert(None)

    def fit(self):
        self.train_df_prophet = self.train_df[['Datum-tijd tot', 'Elektriciteit Consumptie (kWh)']].rename(
            columns={'Datum-tijd tot': 'ds', 'Elektriciteit Consumptie (kWh)': 'y'}
        )
        self.model = Prophet(growth='linear',seasonality_mode='multiplicative',seasonality_prior_scale=1,
                             changepoint_prior_scale=0.01,interval_width=0.2,scaling='absmax',
                             yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=True)
        self.model.fit(self.train_df_prophet)

    def predict(self):
        future_df = self.model.make_future_dataframe(periods=24 * self.days_to_forecast, freq='h')
        self.forecast = self.model.predict(future_df)
        prophet_forecast = self.forecast
        prophet_forecast.reset_index(inplace=True)
        prophet_forecast.rename(columns={
            'ds': 'Datum-tijd tot',
            'yhat': 'Elektriciteit Consumptie (kWh)',
            'yhat_lower': 'Consumption_Lower',
            'yhat_upper': 'Consumption_Upper'
        }, inplace=True)
        return prophet_forecast

    def run(self):
        self.split_data()
        self.fit()
        return self.predict()

    def get_train_test_df(self):
        return self.train_df, self.test_df


class MetricsCalculator:
    def __init__(self, true_df, predicted_df):
        self.true_df = true_df
        self.predicted_df = predicted_df

    def calculate_rmse(self):
        #convert both true and predicted to same format in datetime utc
        # Ensure 'Datum-tijd tot' column is in datetime format and timezone-aware using .loc
        self.true_df['Datum-tijd tot'] = pd.to_datetime(self.true_df['Datum-tijd tot'], utc=True).dt.tz_convert(None)
        self.predicted_df['Datum-tijd tot'] = pd.to_datetime(self.predicted_df['Datum-tijd tot'], utc=True).dt.tz_convert(None)

        merged_df = pd.merge(self.true_df, self.predicted_df, on='Datum-tijd tot', suffixes=('_true', '_pred'))
        rmse = np.sqrt(mean_squared_error(merged_df['Elektriciteit Consumptie (kWh)_true'], merged_df['Elektriciteit Consumptie (kWh)_pred']))
        return rmse

    def calculate_mad(self):
        merged_df = pd.merge(self.true_df, self.predicted_df, on='Datum-tijd tot', suffixes=('_true', '_pred'))
        mad = np.median(np.abs(merged_df['Elektriciteit Consumptie (kWh)_true'] - merged_df['Elektriciteit Consumptie (kWh)_pred']))
        return mad

    #calculate R2 as well as correlation
    def calculate_r2(self):
        merged_df = pd.merge(self.true_df, self.predicted_df, on='Datum-tijd tot', suffixes=('_true', '_pred'))
        corr_matrix = np.corrcoef(merged_df['Elektriciteit Consumptie (kWh)_true'], merged_df['Elektriciteit Consumptie (kWh)_pred'])
        r2 = corr_matrix[0, 1] ** 2
        return r2

    #calculate mean absolute percentage error
    def calculate_mape(self):
        merged_df = pd.merge(self.true_df, self.predicted_df, on='Datum-tijd tot', suffixes=('_true', '_pred'))
        mape = np.mean(np.abs((merged_df['Elektriciteit Consumptie (kWh)_true'] - merged_df['Elektriciteit Consumptie (kWh)_pred']) / merged_df['Elektriciteit Consumptie (kWh)_true']))
        return mape*100
    def calculate_metrics(self):
        rmse = self.calculate_rmse()
        mad = self.calculate_mad()
        r2 = self.calculate_r2()
        mape = self.calculate_mape()
        return {'RMSE': rmse, 'MAD': mad, 'R2': r2, 'MAPE': mape}
#%%
if __name__ == '__main__':
    pass
    # Example usage
    # write code here to test the class
    #data folder
    cwd=os.getcwd()
    #get one level up
    data_folder=os.path.dirname(cwd)
    data_folder=os.path.join(data_folder,'data')
    sys.path.append(data_folder)
    merged_file=os.path.join(data_folder,'total_energy_consumption_until2024_10.csv')
    energy_consump_df=pd.read_csv(merged_file)
    energy_consump_df['Datum-tijd tot'] = pd.to_datetime(energy_consump_df['Datum-tijd tot'], utc=True)

    start_date = '2024-02-01'
    days_to_forecast = 1
    #ETS model
    ets_model = ETSModelWeekly(energy_consump_df, start_date, days_to_forecast)
    ets_pred_df = ets_model.run()
    #Prophet model
    prophet_model = ProphetModelWeekly(energy_consump_df, start_date, days_to_forecast)
    prophet_pred_df = prophet_model.run()
    #Metrics
    train_df, test_df = ets_model.get_train_test_df()
    #%%
    #Metrics on model i.e. training
    #divide the predictions on the length of train and test
    ets_model_df_val=ets_pred_df[0:len(train_df)]
    prophet_model_df_val=prophet_pred_df[0:len(train_df)]
    #model validation on training data
    metrics_calculator = MetricsCalculator(train_df, ets_model_df_val)
    metrics_ets = metrics_calculator.calculate_metrics()
    metrics_calculator = MetricsCalculator(train_df, prophet_model_df_val)
    metrics_prophet = metrics_calculator.calculate_metrics()
    print('Metrics on training data')
    print('ETS:', metrics_ets)
    print('Prophet:', metrics_prophet)
    #Metrics on model forecast
    ets_model_df_forecast=ets_pred_df[len(train_df):]
    prophet_model_df_forecast=prophet_pred_df[len(train_df):]
    metrics_calculator = MetricsCalculator(test_df, ets_model_df_forecast)
    metrics_ets = metrics_calculator.calculate_metrics()
    metrics_calculator = MetricsCalculator(test_df, prophet_model_df_forecast)
    metrics_prophet = metrics_calculator.calculate_metrics()
    print('Metrics on forecast data')
    print('ETS:', metrics_ets)
    print('Prophet:', metrics_prophet)


    # %%

