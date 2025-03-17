import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)

# Wczytanie obrazu i konwersja kolorow z BGR (domyslny format OpenCV) na RGB
image = cv.cvtColor(cv.imread("images/mountains.jpg"), cv.COLOR_BGR2RGB)

# Przeksztalcenie obrazu do postaci listy pikseli
YCC = np.asarray(image.reshape(-1,3))
YCC = YCC.astype(np.uint8)

# Macierz transformacji RGB -> YCbCr
rgbToYcrcb = [
    [0.229, 0.587, 0.114],
    [0.500, -0.418, -0.082],
    [-0.168, -0.331, 0.500]
]

rgbToYcrcb = np.asarray(rgbToYcrcb)

shift = [
    [0],
    [128],
    [128],
]
shift = np.asarray(shift)

# Przeksztalcenie kazdego piksela
YCCImage = []
for row in YCC:
    col = np.asarray(row.reshape(3, 1))
    transformedPixel = np.matmul(rgbToYcrcb, col) + shift  # Mnozenie macierzy i dodanie przesuniecia
    YCCImage.append(transformedPixel.reshape(1, 3))

# Ograniczenie wartosci do zakresu [0, 255] i przywrocenie wymiarow obrazu
YCCImage = np.clip(YCCImage, 0, 255)
YCCImage = np.asarray(YCCImage).reshape(image.shape)
YCCImage = YCCImage.astype(np.uint8)

# Konwersja odwrotna YCbCr -> RGB
image_reconstructed = cv.cvtColor(YCCImage, cv.COLOR_YCrCb2RGB)

# Tworzenie wykresow
fig, ax = plt.subplots(2, 3)

# Oryginalny obraz RGB
ax[0, 0].imshow(image)
ax[0, 0].set_title("Oryginalny obraz")
ax[0, 0].axis("off") # Usuniecie osi dla pierwszego obrazka

# Skladowe Y, Cb, Cr w odcieniach szarosci
ax[0, 1].imshow(YCCImage[:, :, 0], cmap="gray")
ax[0, 1].set_title("Składowa Y")
ax[0, 1].axis("off") # Usuniecie osi dla drugiego obrazka

ax[0, 2].imshow(YCCImage[:, :, 1], cmap="gray")
ax[0, 2].set_title("Składowa Cb")
ax[0, 2].axis("off") # Usuniecie osi dla trzeciego obrazka

ax[1, 0].imshow(YCCImage[:, :, 2], cmap="gray")
ax[1, 0].set_title("Składowa Cr")
ax[1, 0].axis("off") # Usuniecie osi dla czwartego obrazka

# Obraz po konwersji odwrotnej
ax[1, 1].imshow(image_reconstructed)
ax[1, 1].set_title("Obraz po konwersji odwrotnej")
ax[1, 1].axis("off") # Usuniecie osi dla piatego obrazka

# Usunięcie pustej osi
ax[1, 2].axis("off")

plt.tight_layout() # Dopasowanie automatyczne obrazkow do rozmiaru ekranu
plt.show()
