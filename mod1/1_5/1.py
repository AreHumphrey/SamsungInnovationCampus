s = input().strip()

freq = {}

for i in range(len(s) - 1):
    pair = s[i:i+2]
    freq[pair] = freq.get(pair, 0) + 1

max_count = max(freq.values())

best_pair = ""
for pair,  count in freq.items():
    if count == max_count:
        if pair > best_pair:
            best_pair = pair

print(best_pair)