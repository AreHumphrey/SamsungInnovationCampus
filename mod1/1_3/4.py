import numpy as np
import pandas as pd

df = pd.read_csv('input.csv')

data = df.iloc[:, 1:]
n = data.shape[1]
if n == 0 or n == 1:
    print([1.0])
else:
    corr = data.corr()
    corr_values = corr.values
    np.fill_diagonal(corr_values, np.nan)
    abs_corr = np.abs(corr_values)
    max_vals = []
    for j in range(n):
        col = abs_corr[:, j]
        if np.all(np.isnan(col)):
            max_vals.append(0.0)
        else:
            max_vals.append(np.nanmax(col))
    result = [round(x, 2) for x in max_vals]
    print(result)