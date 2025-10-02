import numpy as np
import pandas as pd

df = pd.read_csv('input.csv')

data = df.iloc[:, 1:]
print((data == 1).sum().sum())