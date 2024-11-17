import matplotlib.pyplot as plt
import pandas as pd
from Model import train_and_forecast_arima
from Data import prepare_data

def main():
    # Prepare the dataset
    data = prepare_data()

    # Get a list of unique countries
    countries = data["Country"].unique()

    # Forecast for each country
    for country in countries:
        print(f"Forecasting GDP growth for {country}...")
        forecast = train_and_forecast_arima(country, data)
        print(f"Forecast for 2024-2030 for {country}:\n{forecast}\n")

if __name__ == "__main__":
    main()