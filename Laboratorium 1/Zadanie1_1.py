import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)

# Wczytanie obrazu i konwersja kolorow z BGR (domyslny format OpenCV) na RGB
image = cv.cvtColor(cv.imread("images/mountains.jpg"), cv.COLOR_BGR2RGB)

# Definicja macierzy filtru gornoprzepustowego (detektora krawedzi)
filter_matrix = [
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]

filter_matrix = np.asarray(filter_matrix)
# Zastosowanie filtru gornoprzepustowego do obrazu w celu wykrycia krawedzi
filtered_image = cv.filter2D(image, -1, filter_matrix)
fig, ax = plt.subplots(1,2) # Tworzenie figur i osi do wyswietlania obrazow

# Wyswietlenie oryginalnego obrazu
ax[0].imshow(image)
ax[0].set_title("Oryginalny obraz")
ax[0].axis("off") # Usuniecie osi dla pierwszego obrazka
# Wyswietlenie przefiltrowanego obrazu (z detekcja krawedzi)
ax[1].imshow(filtered_image)
ax[1].set_title("Przefiltrowany obraz")
ax[1].axis("off") # Usuniecie osi dla drugiego obrazka

plt.tight_layout() # Dopasowanie automatyczne obrazkow do rozmiaru ekranu
plt.show()
