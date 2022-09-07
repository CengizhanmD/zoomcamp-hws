import numpy as np
import pandas as pd

#Q1
print("Numpy version is " + np.__version__ + "\n \n")

# Q2
data = pd.DataFrame(pd.read_csv('./data.csv'))
records =data.shape[0]
print("Number of records are " + str(records) + "\n \n")

# Q3
make_series = data["Make"]
print("Most popular car manufacturers are", make_series.value_counts()[0:3].index, "\n \n")

# Q4
audi = data[data.Make == "Audi"]
print("number of unique Audi cars are", audi.Model.nunique(), "\n \n")

# Q5
check_data = data.isna()
number_of_nans = check_data.sum()
print("Columns with missing values are \n", number_of_nans[number_of_nans != 0], "\n \n")

# Q6
median = data["Engine Cylinders"].median()
mode = data["Engine Cylinders"].mode()
data["Engine Cylinders"].fillna(value=mode)
median2 = data["Engine Cylinders"].median()
print("First median is:", median, " Second median is:", median2, "\n \n")

# Q7
lotus = data[data["Make"] == "Lotus"]
lotus = lotus.drop_duplicates(subset = ["Engine HP", "Engine Cylinders"])
lotus = lotus[["Engine HP", "Engine Cylinders"]]
X = lotus.to_numpy()
XTX = np.matmul(X.T, X)
inv = np.linalg.inv(XTX)
y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])
w = np.matmul(np.matmul(inv, X.T), y)
print("First element of w is:", w[0])