import csv
import numpy as np

data = []
with open("Dataset/South_Asian_dataset.csv", mode="r") as file:
    csvReader = csv.reader(file)
    headers = next(csvReader)
    data = [row for row in csvReader]

# Specify the column indices to process (excluding Country, Year, and Unemployment)
for colIndex in [3, 4, 5, 7, 8, 9, 10]:
    # Collect non-missing values in the column to calculate the mean
    columnData = [float(row[colIndex]) for row in data if row[colIndex] != ""]
    if columnData: # Check if there's any non-missing data
        meanValue = np.mean(columnData)

        # Fill missing values with the mean
        for row in data:
            if row[colIndex] == "":
                # Convert to string to match data format in CSV
                row[colIndex] = str(meanValue)