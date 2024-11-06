# Predicting Energy Consumption of THUAS Buildings

## Introduction
This repository has all the code and scripts for analyzing and making the models for predicting the energy consumption of THUAS building. 
We obtained the data from Baldiri Salcedo who is also actively involved in this project.

## Setting up the working environment

`python3 -m venv energythuasapp` - Create a virtual environment
`source energythuasapp/bin/activate` - Activate the virtual environment
`pip install --upgrade pip` - Upgrade pip
`pip install -r requirements.txt`


## Models

Currently we are thinking of multiple models to predict the energy consumption of THUAS buildings..
1. A direct application of the ARIMA model or exponential smoothing model to the time series data.
3. May be apply a proprietary facebook prophet model to the data?


For validation and checkeing we do a forecast of 1 day ahead and check with the actual data of 2024/2023. And then slowly with a sliding window approach get the data of the next day and check the forecast of the next day..