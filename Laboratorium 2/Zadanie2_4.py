import numpy as np
import struct
import zlib
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os

# Parametry obrazu
temp_steps = 17
temp_step = 255 // temp_steps
width, height = 122, 10

# Tworzenie obrazu tęczy
image = np.zeros((height, width, 3), dtype=np.uint8)
dummy = np.array([0, 0, 0], dtype=np.uint8)
idx = 1

for i in range(temp_steps):
    dummy[2] += temp_step
    image[1:-1, idx] = dummy
    idx += 1
for i in range(temp_steps):
    dummy[1] += temp_step
    image[1:-1, idx] = dummy
    idx += 1
for i in range(temp_steps):
    dummy[2] -= temp_step
    image[1:-1, idx] = dummy
    idx += 1
for i in range(temp_steps):
    dummy[0] += temp_step
    image[1:-1, idx] = dummy
    idx += 1
for i in range(temp_steps):
    dummy[1] -= temp_step
    image[1:-1, idx] = dummy
    idx += 1
for i in range(temp_steps):
    dummy[2] += temp_step
    image[1:-1, idx] = dummy
    idx += 1
for i in range(temp_steps + 1):
    dummy[1] = min(255, dummy[1] + temp_step)
    image[1:-1, idx] = dummy
    idx += 1


# Funkcja pomocnicza do paddingu
def pad_image(img, block_size=8):
    h, w = img.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)


# Konwersja na YCrCb
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(image_ycrcb)

# Dodanie paddingu
Y = pad_image(Y)
Cr = pad_image(Cr)
Cb = pad_image(Cb)

# Próbkowanie chrominancji
sampling_factors = [1, 2, 4]

for factor in sampling_factors:
    Cr_sub = cv2.resize(Cr, (Cr.shape[1] // factor, Cr.shape[0] // factor), interpolation=cv2.INTER_AREA)
    Cb_sub = cv2.resize(Cb, (Cb.shape[1] // factor, Cb.shape[0] // factor), interpolation=cv2.INTER_AREA)

    # DCT i kwantyzacja
    QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 48, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

    h, w = Y.shape
    compressed_Y = np.zeros_like(Y, dtype=np.int32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = Y[i:i + 8, j:j + 8].astype(np.float32) - 128
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / QY)
            compressed_Y[i:i + 8, j:j + 8] = quantized

    # Odtwarzanie
    reconstructed_Y = np.zeros_like(compressed_Y, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = compressed_Y[i:i + 8, j:j + 8]
            dequantized = block * QY
            idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho') + 128
            reconstructed_Y[i:i + 8, j:j + 8] = idct_block

    Y_recon = np.clip(reconstructed_Y, 0, 255).astype(np.uint8)
    Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_CUBIC)
    Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_CUBIC)

    reconstructed = cv2.merge((Y_recon, Cr_up, Cb_up))
    final_image = cv2.cvtColor(reconstructed, cv2.COLOR_YCrCb2BGR)
    final_image = final_image[:image.shape[0], :image.shape[1]]

    cv2.imwrite(f'jpeg_{factor}x.jpg', final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    size = os.path.getsize(f'jpeg_{factor}x.jpg')
    print(f'Rozmiar dla próbkowania {factor}x: {size} bajtów')

    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Oryginał')
    plt.subplot(122), plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)), plt.title(f'Próbkowanie {factor}x')
    plt.show()