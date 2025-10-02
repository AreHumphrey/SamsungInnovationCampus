s = input().strip()
count = 0

for i in range(len(s) - 1):
    if s[i + 1] == 'a':
        count += 1

print(count)