import numpy as np
import matplotlib.pyplot as plt
import cv2

# Parametry obrazu
steps = 17
step = 255 // steps

# Tworzenie obrazu z czarnym obramowaniem
image = np.zeros((10, 122, 3), dtype=np.uint8)  # 10 wierszy (8+2), 122 kolumny (120+2)

# Tworzenie gradientu RGB
dummy = np.array([0, 0, 0], dtype=np.uint8)
idx = 1  # Zaczynamy od pierwszej kolumny wewnatrz obramowania

# Przejscia kolorow
# Od czarnego do niebieskiego
for i in range(steps):
    dummy[2] += step
    image[1:-1, idx] = dummy
    idx += 1
# Od niebieskiego do cyjanu
for i in range(steps):
    dummy[1] += step
    image[1:-1, idx] = dummy
    idx += 1
# Od cyjanu do zielonego
for i in range(steps):
    dummy[2] -= step
    image[1:-1, idx] = dummy
    idx += 1
# Od zielonego do zoltego
for i in range(steps):
    dummy[0] += step
    image[1:-1, idx] = dummy
    idx += 1
# Od zoltego do czerwonego
for i in range(steps):
    dummy[1] -= step
    image[1:-1, idx] = dummy
    idx += 1
# Od czerwonego do magenty
for i in range(steps):
    dummy[2] += step
    image[1:-1, idx] = dummy
    idx += 1
# Od magenty do bialego
for i in range(steps + 1):
    dummy[1] = min(255, dummy[1] + step)
    image[1:-1, idx] = dummy
    idx += 1

# Poprawny zapis PPM
ppm_header = f'P3\n{image.shape[1]} {image.shape[0]}\n255\n'
with open('tecza.ppm', 'w') as fh:
    fh.write(ppm_header)
    for row in image:
        for pixel in row:
            fh.write(f'{pixel[0]} {pixel[1]} {pixel[2]} ')
        fh.write('\n')

# Wczytanie i wyswietlenie
image_from_file = cv2.imread('tecza.ppm')
plt.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()