import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)

# Wczytanie obrazu i konwersja kolorow z BGR (domyslny format OpenCV) na RGB
image = cv.cvtColor(cv.imread("images/mountains.jpg"), cv.COLOR_BGR2RGB)

# Przeksztalcenie obrazu na tablice float32 i normalizacja wartosci do zakresu [0,1]
normalizedPixels = np.float32(image.reshape(-1,3))
normalizedPixels /= 255.

# Macierz transformacji do efektu sepii
color_matrix = [
    [0.393, 0.769, 0.189],
    [0.349, 0.689, 0.164],
    [0.272, 0.534, 0.131],
]
# Konwersja listy na macierz NumPy
transform = np.asarray(color_matrix)
# Lista do przechowywania nowych wartosci pikseli
newPixels = []
# Iteracja po kazdym pikselu obrazu
for row in normalizedPixels:
    col = np.asarray(row.reshape(3,1)) # Przeksztalcenie do postaci kolumnowej
    transformedPixel = np.matmul(transform,col) # Zastosowanie transformacji (mnozenie macierzy)
    transformedPixel = transformedPixel.reshape(1,3) # Powrot do postaci 1x3
    newPixels.append(transformedPixel) # Dodanie przeksztalconego piksela do nowej listy

newPixels = np.clip(newPixels, 0.0, 1.0) # Ograniczenie wartosci do zakresu [0,1] (zapobieganie przekroczeniu maksymalnej wartosci)
newPixels = np.asarray(newPixels).reshape(image.shape) # Konwersja listy do macierzy NumPy i przywrocenie oryginalnych wymiarow obrazu

fig, ax = plt.subplots(1,2) # Tworzenie figur i osi do wyswietlania obrazow

# Wyswietlenie oryginalnego obrazu
ax[0].imshow(image)
ax[0].set_title("Oryginalny obraz")
ax[0].axis("off") # Usuniecie osi dla pierwszego obrazka
# Wyswietlenie obrazu po przeksztalceniu kolorow
ax[1].imshow(newPixels)
ax[1].set_title("Przeksztalcony obraz")
ax[1].axis("off") # Usuniecie osi dla drugiego obrazka

plt.tight_layout() # Dopasowanie automatyczne obrazkow do rozmiaru ekranu
plt.show()
