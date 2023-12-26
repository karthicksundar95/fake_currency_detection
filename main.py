####  main script which controls the flow of the application

# importing necessary packages
import pandas as pd
import numpy as np
from pathlib import Path

# function to read the file

def read_input_csv_data(file_path):
    currency_data = pd.read_csv(file_path)
    print(">> inside the read_input_csv_data method <<")
    return currency_data

if __name__ ==  "__main__":
    data = read_input_csv_data("{}/BankNote_Authentication.csv".format(Path.cwd()))
    print(">> read completed <<")
    print(data.head())
    print(">> data print completed <<")