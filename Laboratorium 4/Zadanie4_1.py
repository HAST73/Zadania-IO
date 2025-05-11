import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie dwoch roznych obrazow
image1 = cv2.imread('himalayas.jpg')
image2 = cv2.imread('himalayas-high-resolution.jpg')

# Konwersja do skali szarosci
image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Kopie obrazow jako int do obliczen
output1 = image_gray1.astype(np.int32)
output2 = image_gray2.astype(np.int32)

# Funkcja kwantyzacji
def find_closest_grey_color(pixel):
    return round(pixel / 255) * 255

# Dithering na pierwszym obrazie
for y in range(output1.shape[0] - 1):
    for x in range(1, output1.shape[1] - 1):
        oldpixel = output1[y, x]
        newpixel = find_closest_grey_color(oldpixel)
        output1[y, x] = newpixel
        quant_error = oldpixel - newpixel

        output1[y, x + 1] += quant_error * 7 / 16
        output1[y + 1, x - 1] += quant_error * 3 / 16
        output1[y + 1, x] += quant_error * 5 / 16
        output1[y + 1, x + 1] += quant_error * 1 / 16

# Dithering na drugim obrazie
for y in range(output2.shape[0] - 1):
    for x in range(1, output2.shape[1] - 1):
        oldpixel = output2[y, x]
        newpixel = find_closest_grey_color(oldpixel)
        output2[y, x] = newpixel
        quant_error = oldpixel - newpixel

        output2[y, x + 1] += quant_error * 7 / 16
        output2[y + 1, x - 1] += quant_error * 3 / 16
        output2[y + 1, x] += quant_error * 5 / 16
        output2[y + 1, x + 1] += quant_error * 1 / 16

# Przyciecie i konwersja do uint8
output1 = np.clip(output1, 0, 255).astype(np.uint8)
output2 = np.clip(output2, 0, 255).astype(np.uint8)

# Wyswietlenie obu wynikow obok siebie
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(output1, cmap='gray')
plt.title('Dithering obrazu o obnizonej rozdzielczosci')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output2, cmap='gray')
plt.title('Dithering obrazu o zwiekszonej rozdzielczosci')
plt.axis('off')

plt.tight_layout()
plt.show()

# Histogram
histr = cv2.calcHist([output2], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 6))
plt.plot(histr, label='himalayas-high-resolution.jpg')
plt.xlim([0, 256])
plt.xlabel('Wartosc skladowej koloru [0-255]')
plt.ylabel('Liczba pikseli')
plt.title('Histogram jasnosci po ditheringu')
plt.legend()
plt.grid(True)
plt.show()
