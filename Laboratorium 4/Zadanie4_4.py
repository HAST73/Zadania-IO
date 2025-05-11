import numpy as np
from matplotlib import pyplot as plt

width = 80
height = 60
image = np.zeros((height, width, 3), dtype=np.uint8)

# Funkcja do rysowania punktu
def draw_point(image, x, y, color=(255, 255, 255)):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[image.shape[0] - 1 - y, x] = np.clip(np.round(color), 0, 255)

# Algorytm Bresenhama z interpolacja koloru
def draw_line(image, x1, y1, x2, y2, col1, col2):
    col1 = np.asarray(col1, dtype=float)
    col2 = np.asarray(col2, dtype=float)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = np.sign(x2 - x1)
    sy = np.sign(y2 - y1)
    x, y = x1, y1
    steps = max(dx, dy)

    for i in range(steps + 1):
        t = i / steps if steps != 0 else 0
        color = (1 - t) * col1 + t * col2
        draw_point(image, x, y, color)

        if dx > dy:
            x += sx
            if (i * dy * 2) % dx >= dx:
                y += sy
        else:
            y += sy
            if (i * dx * 2) % dy >= dy:
                x += sx

# Klasa wierzcholka z kolorem
class Vertex:
    def __init__(self, x, y, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.color = np.asarray(color, dtype=float)


# Funkcja do rysowania wypelnionego trojkata z interpolacja koloru
def draw_triangle(image, A, B, C):
    xmin = min(A.x, B.x, C.x)
    xmax = max(A.x, B.x, C.x)
    ymin = min(A.y, B.y, C.y)
    ymax = max(A.y, B.y, C.y)

    # Calkowite pole trojkata
    def area(p1, p2, p3):
        return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)

    total_area = area(A, B, C)

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            P = Vertex(x, y)
            a0 = area(B, C, P)
            a1 = area(C, A, P)
            a2 = area(A, B, P)

            if np.sign(a0) == np.sign(a1) == np.sign(a2):
                w0 = a0 / total_area
                w1 = a1 / total_area
                w2 = a2 / total_area
                color = w0 * A.color + w1 * B.color + w2 * C.color
                draw_point(image, x, y, color)


# Przykladowe rysowanie
# Linia z gradientem
draw_line(image, 5, 5, 70, 40, (255, 0, 0), (0, 255, 0))

# Trojkat z interpolacja koloru
v0 = Vertex(20, 10, color=(255, 0, 0))
v1 = Vertex(60, 15, color=(0, 255, 0))
v2 = Vertex(40, 40, color=(0, 0, 255))
draw_triangle(image, v0, v1, v2)

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title("Interpolacja koloru: linia + trojkat")
plt.axis('off')
plt.show()
