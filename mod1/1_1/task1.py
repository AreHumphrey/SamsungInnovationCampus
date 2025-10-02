test_cases = [
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4],
    [1, 2, 3],
    [1],
    [],
    ['a', 'b', 'c', 'd', 'e']
]

for i, A in enumerate(test_cases):
    result = A[-1:-4:1]
    print(f"Случай {i + 1}: A = {A}")
    print(f"  Срез A[-1:-4:1] = {result}")
    print(f"  Количество элементов: {len(result)}\n")