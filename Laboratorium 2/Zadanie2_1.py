import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Rozmiar obrazu
width, height = 6, 6

# Tworzenie jednolitego niebieskiego obrazu (pełny niebieski: R=0, G=0, B=255)
blue_image = np.full((height, width, 3), [0, 0, 255], dtype=np.uint8)

# Wyświetlenie obrazu
plt.imshow(blue_image)
plt.title("Wygenerowany niebieski obraz")
plt.axis("off")
plt.show()

# Nagłówki PPM
ppm_ascii_header = f'P3\n{width} {height}\n255\n'
ppm_binary_header = f'P6\n{width} {height}\n255\n'

# Zapis do formatu ASCII PPM (P3)
with open('blue-ascii.ppm', 'w') as fh:
    fh.write(ppm_ascii_header)
    blue_image.tofile(fh, sep=' ')
    fh.write('\n')

# Zapis do formatu binarnego PPM (P6)
with open('blue-binary.ppm', 'wb') as fh:
    fh.write(bytearray(ppm_binary_header, 'ascii'))
    blue_image.tofile(fh)

# Funkcja odczytu ASCII PPM
def read_ppm_ascii(file_path):
    with open(file_path, 'r') as file:
        file.readline()  # Pominięcie nagłówka P3
        width, height = map(int, file.readline().split())  # Odczyt wymiarów
        file.readline()  # Pominięcie wartości maksymalnej

        # Wczytanie pikseli
        pixels = []
        for line in file:
            pixels.extend(map(int, line.split()))

        image = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
        return image

# Funkcja odczytu binarnego PPM
def read_ppm_binary(file_path):
    with open(file_path, 'rb') as file:
        file.readline()  # Pominięcie nagłówka P6
        width, height = map(int, file.readline().split())  # Odczyt wymiarów
        file.readline()  # Pominięcie wartości maksymalnej

        pixels = np.fromfile(file, dtype=np.uint8)
        image = pixels.reshape((height, width, 3))
        return image

# Odczyt obrazów
image_from_ascii = read_ppm_ascii('blue-ascii.ppm')
image_from_binary = read_ppm_binary('blue-binary.ppm')

# Porównanie rozmiarów plików
ascii_size = os.path.getsize('blue-ascii.ppm')
binary_size = os.path.getsize('blue-binary.ppm')

print(f"Rozmiar pliku PPM ASCII (P3): {ascii_size} bajtów")
print(f"Rozmiar pliku PPM Binarnego (P6): {binary_size} bajtów")

# Wizualizacja odczytanych obrazów
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_from_ascii)
ax[0].set_title("Obraz z ASCII PPM")
ax[0].axis("off")

ax[1].imshow(image_from_binary)
ax[1].set_title("Obraz z Bin PPM")
ax[1].axis("off")

plt.show()

####################################################

# Wczytanie obrazu
image = cv2.imread("blue_photo.jpg")  # Wczytaj zdjęcie
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja z BGR na RGB

# Pobranie wymiarów
height, width, _ = image.shape

# Wyświetlenie oryginalnego obrazu
plt.imshow(image)
plt.title("Oryginalne zdjęcie")
plt.axis("off")
plt.show()

# Nagłówki PPM
ppm_ascii_header = f'P3\n{width} {height}\n255\n'
ppm_binary_header = f'P6\n{width} {height}\n255\n'

# Zapis do formatu ASCII PPM (P3)
with open('blue_photo_ascii.ppm', 'w') as fh:
    fh.write(ppm_ascii_header)
    np.savetxt(fh, image.reshape(-1, 3), fmt='%d', delimiter=' ')

# Zapis do formatu binarnego PPM (P6)
with open('blue_photo_binary.ppm', 'wb') as fh:
    fh.write(bytearray(ppm_binary_header, 'ascii'))
    image.tofile(fh)

# Funkcja odczytu ASCII PPM
def read_ppm_ascii(file_path):
    with open(file_path, 'r') as file:
        file.readline()  # Pominięcie nagłówka P3
        width, height = map(int, file.readline().split())  # Odczyt wymiarów
        file.readline()  # Pominięcie wartości maksymalnej

        # Wczytanie pikseli
        pixels = []
        for line in file:
            pixels.extend(map(int, line.split()))

        image = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
        return image

# Funkcja odczytu binarnego PPM
def read_ppm_binary(file_path):
    with open(file_path, 'rb') as file:
        file.readline()  # Pominięcie nagłówka P6
        width, height = map(int, file.readline().split())  # Odczyt wymiarów
        file.readline()  # Pominięcie wartości maksymalnej

        pixels = np.fromfile(file, dtype=np.uint8)
        image = pixels.reshape((height, width, 3))
        return image

# Odczyt obrazów
image_from_ascii = read_ppm_ascii('blue_photo_ascii.ppm')
image_from_binary = read_ppm_binary('blue_photo_binary.ppm')

# Porównanie rozmiarów plików
ascii_size = os.path.getsize('blue_photo_ascii.ppm')
binary_size = os.path.getsize('blue_photo_binary.ppm')

print(f"Rozmiar pliku PPM ASCII (P3): {ascii_size} bajtów")
print(f"Rozmiar pliku PPM Binarnego (P6): {binary_size} bajtów")

# Wizualizacja odczytanych obrazów
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_from_ascii)
ax[0].set_title("Obraz z ASCII PPM")
ax[0].axis("off")

ax[1].imshow(image_from_binary)
ax[1].set_title("Obraz z Bin PPM")
ax[1].axis("off")

plt.show()
