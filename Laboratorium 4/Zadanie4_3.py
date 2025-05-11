import numpy as np
from matplotlib import pyplot as plt

width = 80
height = 60
image = np.zeros((height, width, 3), dtype=np.uint8)

# Funkcja do rysowania punktu
def draw_point(image, x, y, color=(255, 255, 255)):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        image[image.shape[0] - 1 - y, x] = color

# Algorytm Bresenhama do rysowania linii
def draw_line(image, x1, y1, x2, y2, color=(255, 255, 255)):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = np.sign(x2 - x1)
    sy = np.sign(y2 - y1)
    x, y = x1, y1

    if dx > dy:
        err = dx // 2
        for _ in range(dx + 1):
            draw_point(image, x, y, color)
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
    else:
        err = dy // 2
        for _ in range(dy + 1):
            draw_point(image, x, y, color)
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy

# Klasa wierzcholka
class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Funkcja do rysowania wypelnionego trojkata
def draw_triangle(image, A, B, C, color=(255, 255, 255)):
    xmin = min(A.x, B.x, C.x)
    xmax = max(A.x, B.x, C.x)
    ymin = min(A.y, B.y, C.y)
    ymax = max(A.y, B.y, C.y)

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            # Obliczamy iloczyny wektorowe (pole trojkata ×2)
            ar1 = (x - A.x)*(B.y - A.y) - (y - A.y)*(B.x - A.x)
            ar2 = (x - B.x)*(C.y - B.y) - (y - B.y)*(C.x - B.x)
            ar3 = (x - C.x)*(A.y - C.y) - (y - C.y)*(A.x - C.x)

            if np.sign(ar1) == np.sign(ar2) == np.sign(ar3):
                draw_point(image, x, y, color)

# Przykladowe rysowanie
# Jednokolorowa linia
draw_line(image, 5, 5, 70, 40, color=(255, 255, 255))

# Trojkat
v0 = Vertex(20, 10)
v1 = Vertex(60, 15)
v2 = Vertex(40, 40)
draw_triangle(image, v0, v1, v2, color=(0, 0, 139))

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title("Jednokolorowa linia i trojkat (z wypelnieniem)")
plt.axis('off')
plt.show()
