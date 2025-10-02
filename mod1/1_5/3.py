unique_rows = set()
while True:
    try:
        row = tuple(map(int, input().split()))
        unique_rows.add(row)
    except EOFError:
        break
print(len(unique_rows))