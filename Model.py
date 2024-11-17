import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import Data

def trainModelForCountry(countryName):
    # Load and clean data
    headers, data = Data.load_and_clean_data()
    df = pd.DataFrame(data, columns=headers)

    # Filter data for the specific country
    country_df = df[df["Country"] == countryName].copy()

    # Ensure numeric columns are in correct format and handle NaNs
    for column in ["GDP per capita (current US$)", "Unemployment, total (% of total labor force) (modeled ILO estimate)",
                   "Inflation, consumer prices (annual %)", "Foreign direct investment, net inflows (% of GDP)",
                   "Trade (% of GDP)", "GDP growth (annual %)"]:
        country_df[column] = pd.to_numeric(country_df[column], errors='coerce')
        country_df[column].fillna(country_df[column].mean(), inplace=True)

    # Define features (x) and target (y) for GDP growth prediction
    x = country_df[["GDP per capita (current US$)", "Unemployment, total (% of total labor force) (modeled ILO estimate)",
                    "Inflation, consumer prices (annual %)", "Foreign direct investment, net inflows (% of GDP)",
                    "Trade (% of GDP)"]]
    y = country_df["GDP growth (annual %)"]

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Model trained for {countryName}.")
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)

    # Return the trained model and test data
    return model, x_test, y_test, y_pred

def train_and_forecast_arima(country, data, forecast_years=7):
    # Filter the data for the selected country
    country_data = data[data["Country"] == country]

    # Extract GDP growth data
    gdp_growth = country_data["GDP growth (annual %)"]
    gdp_growth.index = gdp_growth.index.to_period("Y")  # Set yearly frequency

    # Debugging: Print data information
    print(f"Country: {country}")
    print(f"GDP Growth Data:\n{gdp_growth}")
    print(f"Index Type: {type(gdp_growth.index)}\n")

    # Check for data sufficiency
    if len(gdp_growth) < 10:
        print(f"Insufficient data for {country}. Skipping forecast.\n")
        future_years = pd.period_range(start=gdp_growth.index[-1] + 1, periods=forecast_years, freq="Y")
        return pd.Series([None] * forecast_years)

    # Use auto_arima to find the best ARIMA order
    try:
        model_order = auto_arima(
            gdp_growth,
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'  # Ignore if model fails to converge
        ).order

        print(f"Best ARIMA order for {country}: {model_order}")

        # Handle trivial orders
        if model_order == (0, 0, 0):
            print(f"Trivial order detected for {country}. Using fallback order (1, 1, 1).")
            model_order = (1, 1, 1)

        # Train the ARIMA model
        model = ARIMA(gdp_growth, order=model_order)
        model_fit = model.fit()

    except Exception as e:
        print(f"ARIMA model failed for {country}: {e}\n")
        future_years = pd.period_range(start=gdp_growth.index[-1] + 1, periods=forecast_years, freq="Y")
        return pd.Series([None] * forecast_years, index=future_years)

    # Forecast for the next `forecast_years`
    future_years = pd.period_range(start=gdp_growth.index[-1] + 1, periods=forecast_years, freq="Y")
    try:
        forecast = model_fit.forecast(steps=forecast_years)
        forecast = pd.Series(forecast, index=future_years)
        print(f"Forecast for {country}:\n{forecast}\n")
    except Exception as e:
        print(f"Forecasting failed for {country}: {e}\n")
        forecast = pd.Series([None] * forecast_years, index=future_years)

    # Convert PeriodIndex to DatetimeIndex for plotting
    gdp_growth.index = gdp_growth.index.to_timestamp()
    future_years = future_years.to_timestamp()  # Convert future PeriodIndex to DatetimeIndex

    # Plot historical and forecast data
    plt.figure(figsize=(10, 6))
    plt.plot(gdp_growth.index, gdp_growth, label="Historical GDP Growth")
    if not forecast.isna().all():
        plt.plot(future_years, forecast, label="Forecast", color="orange")
    else:
        print(f"No forecast available for {country}.")
    plt.title(f"GDP Growth Forecast for {country}")
    plt.xlabel("Year")
    plt.ylabel("GDP Growth (annual %)")
    plt.legend()
    plt.grid()
    plt.show()

    return forecast

def find_best_arima(gdp_growth):
    model = auto_arima(
        gdp_growth,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        seasonal=False,
        stepwise=True
    )
    return model.order