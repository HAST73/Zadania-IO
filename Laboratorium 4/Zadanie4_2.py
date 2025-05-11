import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie obrazu i konwersja do RGB (dla matplotliba)
image = cv2.imread('modern.jpg')
image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Konfiguracja: liczba poziomow kolorow
tones = 8  # np. 2 (cz-b), 4, 8, 16

# Przygotowanie danych wyjsciowych
output = np.copy(image_color).astype(int)
reduced = np.copy(image_color).astype(int)

# Funkcja kwantyzacji do palety
def find_closest_palette_color(pix, k=2):
    return round((k - 1) * pix / 255) * 255 / (k - 1)

# Redukcja kolorow (bez ditheringu) — przetwarzamy caly obraz
for y in range(reduced.shape[0]):
    for x in range(reduced.shape[1]):
        for i in range(3):  # R, G, B
            reduced[y, x, i] = find_closest_palette_color(reduced[y, x, i], tones)

# Dithering (Floyd–Steinberg)
for y in range(output.shape[0] - 1):
    for x in range(1, output.shape[1] - 1):
        oldpixel = output[y, x].copy()
        newpixel = np.array([
            find_closest_palette_color(oldpixel[0], tones),
            find_closest_palette_color(oldpixel[1], tones),
            find_closest_palette_color(oldpixel[2], tones)
        ])
        output[y, x] = newpixel
        quant_error = oldpixel - newpixel
        for i in range(3):
            output[y, x + 1, i] += quant_error[i] * 7 / 16
            output[y + 1, x - 1, i] += quant_error[i] * 3 / 16
            output[y + 1, x, i] += quant_error[i] * 5 / 16
            output[y + 1, x + 1, i] += quant_error[i] * 1 / 16

# Przyciecie wartosci i konwersja do uint8
output = np.clip(output, 0, 255).astype(np.uint8)
reduced = np.clip(reduced, 0, 255).astype(np.uint8)

# Wyswietlenie obrazow
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(reduced, interpolation='nearest')
ax[0].set_title(f'Redukcja kolorów ({tones} tonow)')
ax[0].axis('off')

ax[1].imshow(output, interpolation='nearest')
ax[1].set_title(f'Dithering Floyda–Steinberga ({tones} tonow)')
ax[1].axis('off')

plt.tight_layout()
plt.show()

# Histogram RGB
colors = ('r', 'g', 'b')
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

for i, col in enumerate(colors):
    hist_reduced = cv2.calcHist([reduced], [i], None, [256], [0, 256])
    hist_dithered = cv2.calcHist([output], [i], None, [256], [0, 256])

    ax[0].plot(hist_reduced, color=col)
    ax[1].plot(hist_dithered, color=col)

ax[0].set_title('Histogram RGB – obraz po redukcji')
ax[0].set_xlabel('Wartosc koloru [0-255]')
ax[0].set_ylabel('Liczba pikseli')

ax[1].set_title('Histogram RGB – po ditheringu')
ax[1].set_xlabel('Wartosc koloru [0-255]')
ax[1].set_ylabel('Liczba pikseli')

plt.tight_layout()
plt.show()
