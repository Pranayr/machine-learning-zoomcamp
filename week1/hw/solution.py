import numpy as np
import pandas as pd
import os

def read_input_data(filename="week1/data/laptops.csv"):
    data = pd.read_csv(os.path.join(os.getcwd(), filename))
    return data


data = read_input_data()
print(f"Pandas version is: {pd.__version__}")
print(f"Number of records in dataset : {data.shape[0]}")

print(f"Number of laptop brands : {data.Brand.nunique()}")

missing_columns = data.isnull().sum()

# Count the number of columns with missing values
num_missing_columns = (missing_columns > 0).sum()

print(f"Number of Columns with missing values : {num_missing_columns}")

print(f'Dell laptop with max final price: {data[data.Brand=="Dell"]["Final Price"].max()}')

median_before = data.Screen.describe()["50%"]
screen_most_frequent_val = data.Screen.value_counts().index[0]
data.Screen = data.Screen.fillna(screen_most_frequent_val)
median_after = data.Screen.describe()["50%"]

print(f"Screen median before and after same: {median_before==median_after}")

innjoo_data = data.loc[data.Brand=="Innjoo", ["RAM", "Storage", "Screen"]]
x = innjoo_data.values
xTx = np.dot(x.T, x)
xTx_inv = np.linalg.inv(xTx)
y = np.array([1100, 1300, 800, 900, 1000, 1100])

w = np.dot(xTx_inv, x.T)

print(f"Result of whole above operation is {np.sum(np.dot(w, y))}")




