import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)

# === KODER: Konwersja obrazu do YCbCr ===
image = cv.cvtColor(cv.imread("images/mountains.jpg"), cv.COLOR_BGR2RGB)
YCC = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
Y, Cb, Cr = cv.split(YCC)

# === Downsampling (zmniejszenie rozdzielczosci Cb i Cr w obu kierunkach 4:2:0) ===
Cb_down = cv.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2), interpolation=cv.INTER_AREA)
Cr_down = cv.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2), interpolation=cv.INTER_AREA)

# === DEKODER: Upsampling (przywrocenie do pierwotnego rozmiaru) ===
Cb_up = cv.resize(Cb_down, (Cb.shape[1], Cb.shape[0]), interpolation=cv.INTER_LINEAR)
Cr_up = cv.resize(Cr_down, (Cr.shape[1], Cr.shape[0]), interpolation=cv.INTER_LINEAR)

# === Skladanie obrazu po transmisji ===
YCC_transmitted = cv.merge([Y, Cb_up, Cr_up])
image_reconstructed = cv.cvtColor(YCC_transmitted, cv.COLOR_YCrCb2RGB)

# === Tworzenie wykresow ===
fig, ax = plt.subplots(3, 3)

# Oryginalny obraz
ax[0, 0].imshow(image)
ax[0, 0].set_title("Oryginalny obraz")
ax[0, 0].axis("off")

# Skladowa Y
ax[2, 1].imshow(Y, cmap="gray")
ax[2, 1].set_title("Składowa Y")
ax[2, 1].axis("off")

# Skladowa Cb przed downsamplowaniem
ax[0, 2].imshow(Cb, cmap="gray")
ax[0, 2].set_title("Składowa Cb (oryginalna)")
ax[0, 2].axis("off")

# Skladowa Cr przed downsamplowaniem
ax[0, 1].imshow(Cr, cmap="gray")
ax[0, 1].set_title("Składowa Cr (oryginalna)")
ax[0, 1].axis("off")

# Skladowa Cb po downsamplingu i upsamplingu
ax[1, 2].imshow(Cb_up, cmap="gray")
ax[1, 2].set_title("Składowa Cb (po transmisji)")
ax[1, 2].axis("off")

# Skladowa Cr po downsamplingu i upsamplingu
ax[1, 1].imshow(Cr_up, cmap="gray")
ax[1, 1].set_title("Składowa Cr (po transmisji)")
ax[1, 1].axis("off")

# Obraz po konwersji odwrotnej
ax[1, 0].imshow(image_reconstructed)
ax[1, 0].set_title("Obraz po transmisji")
ax[1, 0].axis("off")

# Usuniecie pustych osi
ax[2, 0].axis("off")
ax[2, 2].axis("off")

plt.tight_layout() # Dopasowanie automatyczne obrazkow do rozmiaru ekranu
plt.show()
