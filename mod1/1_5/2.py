k = int(input())
matrix = []
while True:
    try:
        line = input().strip()
        if line:
            row = list(map(int, line.split()))
            matrix.append(row)
    except EOFError:
        break

rows = len(matrix)
cols = len(matrix[0])

block_rows = (rows + k - 1) // k
block_cols = (cols + k - 1) // k

result = []

for i in range(block_rows):
    row_result = []
    start_row = i * k
    end_row = min(start_row + k, rows)
    for j in range(block_cols):
        start_col = j * k
        end_col = min(start_col + k, cols)
        total = 0
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                total += matrix[r][c]
        row_result.append(total)
    result.append(row_result)

for row in result:
    print(' '.join(map(str, row)))