import sys


def main():
    lines = []
    for line in sys.stdin:
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    if not lines:
        print(0)
        return

    matrix = []
    for line in lines:
        row = list(map(int, line.split()))
        matrix.append(row)

    n_rows = len(matrix)
    n_cols = len(matrix[0])

    zero_columns_count = 0

    for j in range(n_cols):
        all_zeros = True
        for i in range(n_rows):
            if matrix[i][j] != 0:
                all_zeros = False
                break
        if all_zeros:
            zero_columns_count += 1

    print(zero_columns_count)


if __name__ == "__main__":
    main()