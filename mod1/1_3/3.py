import numpy as np
import pandas as pd

df = pd.read_csv('input.csv')

print(np.any(pd.isnull(df)))