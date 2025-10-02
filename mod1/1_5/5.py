import numpy as np
import pandas as pd

df = pd.read_csv('input.csv')

data = df.iloc[:, 1:]
arr = data.values
n_rows, n_cols = arr.shape
counts = np.zeros(n_cols, dtype=int)

for i in range(n_rows):
    row = arr[i]
    max_val = row.max()
    max_cols = np.where(row == max_val)[0]
    counts[max_cols] += 1

print(np.argmax(counts))