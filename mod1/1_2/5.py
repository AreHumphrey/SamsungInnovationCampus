import sys
import numpy as np

input_lines = [line.strip() for line in sys.stdin if line.strip()]

if not input_lines:
    exit()

matrix_data = []
for line in input_lines:
    row = [float(x) for x in line.split()]
    matrix_data.append(row)

A = np.array(matrix_data)

row_means = A.mean(axis=1, keepdims=True)

result = A - row_means

print(result)