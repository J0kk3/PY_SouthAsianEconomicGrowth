import csv
import numpy as np
import pandas as pd

def load_and_clean_data():
    data = []
    with open("Dataset/South_Asian_dataset.csv", mode="r") as file:
        csvReader = csv.reader(file)
        headers = next(csvReader)
        data = [row for row in csvReader]

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=headers)

    # Specify columns to fill missing values with the mean (numerical columns only)
    for col in ["GDP (current US$)", "GDP growth (annual %)", "GDP per capita (current US$)",
                "Inflation, consumer prices (annual %)", "Foreign direct investment, net inflows (% of GDP)",
                "Trade (% of GDP)", "Gini index"]:
        # Convert to numeric, coercing errors to NaN, then fill NaN with the column mean
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)

    return df.columns.tolist(), df.values.tolist()

def prepare_data():
    # Load the dataset
    df = pd.read_csv("Dataset/South_Asian_dataset.csv")

    # Extract relevant columns
    df = df[["Year", "Country", "GDP growth (annual %)"]]

    # Drop rows with missing GDP growth values
    df.dropna(subset=["GDP growth (annual %)"], inplace=True)

    # Ensure Year is in datetime format and set as index
    df["Year"] = pd.to_datetime(df["Year"], format='%Y')
    df.set_index("Year", inplace=True)

    # Ensure the data is sorted by Year for each country
    df.sort_values(by=["Country", "Year"], inplace=True)

    return df