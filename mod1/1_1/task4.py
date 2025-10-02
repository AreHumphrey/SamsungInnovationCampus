x, y, x1, y1, x2, y2, r, s = map(float, input().split())

s = int(s)

in_rect = (x1 <= x <= x2) and (y1 <= y <= y2)

in_circle = (x*x + y*y <= r*r)

if s == 0:
    result = in_rect or in_circle
else:
    result = in_rect and in_circle

print(result)