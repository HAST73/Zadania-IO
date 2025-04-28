import numpy as np
import zlib
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os

# Parametry obrazu
steps = 17
step = 255 // steps
width, height = 122, 10

# Tworzenie obrazu teczy
image = np.zeros((height, width, 3), dtype=np.uint8)
dummy = np.array([0, 0, 0], dtype=np.uint8)
idx = 1

# Przejscia kolorow

# Od czarnego do niebieskiego
for i in range(steps):
    dummy[0] += step  # B
    image[1:-1, idx] = dummy
    idx += 1

# Od niebieskiego do cyjanu (Zwiekszaj G)
for i in range(steps):
    dummy[1] += step  # G
    image[1:-1, idx] = dummy
    idx += 1

# Od cyjanu do zielonego (Zmniejszaj B)
for i in range(steps):
    dummy[0] -= step  # B
    image[1:-1, idx] = dummy
    idx += 1

# Od zielonego do zoltego (Zwiekszaj R)
for i in range(steps):
    dummy[2] += step  # R
    image[1:-1, idx] = dummy
    idx += 1

# Od zoltego do czerwonego (Zmniejszaj G)
for i in range(steps):
    dummy[1] -= step  # G
    image[1:-1, idx] = dummy
    idx += 1

# Od czerwonego do magenty (Zwiekszaj B)
for i in range(steps):
    dummy[0] += step  # B
    image[1:-1, idx] = dummy
    idx += 1

# Od magenty do bialego (Zwiekszaj G)
for i in range(steps + 1):
    dummy[1] = min(255, dummy[1] + step)  # G
    image[1:-1, idx] = dummy
    idx += 1

# Funkcja pomocnicza do paddingu
def pad_image(img, block_size=8):
    h, w = img.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

# Funkcja generowania macierzy kwantyzacji dla danego QF
def get_quantization_matrix(QF):
    Q50 = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    if QF < 50 and QF >= 1:
        scale = 5000 / QF
    elif QF <= 100:
        scale = 200 - 2 * QF
    else:
        scale = 1

    Q = np.floor((Q50 * scale + 50) / 100)
    Q[Q == 0] = 1
    return Q.astype(np.uint8)

# Funkcja ZigZag dla bloku 8x8
def zigzag(block):
    zigzag_order = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
    ]
    return np.array([block[i,j] for i,j in zigzag_order])

# Konwersja na YCrCb i rozdzielenie kanalow
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(image_ycrcb)

# Padding kanalow
Y = pad_image(Y)
Cr = pad_image(Cr)
Cb = pad_image(Cb)

# Testowanie roznych QF
QF_list = [10, 30, 50, 70, 90]

for QF in QF_list:
    print(f"\n===== Przetwarzanie dla QF={QF} =====")

    Q = get_quantization_matrix(QF)

    h, w = Y.shape
    compressed_Y = np.zeros_like(Y, dtype=np.int32)

    # DCT + Kwantyzacja
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = Y[i:i+8, j:j+8].astype(np.float32) - 128
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / Q)
            compressed_Y[i:i+8, j:j+8] = quantized

    # Odtwarzanie obrazu: dekwantyzacja + IDCT
    reconstructed_Y = np.zeros_like(compressed_Y, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = compressed_Y[i:i+8, j:j+8]
            dequantized = block * Q
            idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho') + 128
            reconstructed_Y[i:i+8, j:j+8] = idct_block

    Y_recon = np.clip(reconstructed_Y, 0, 255).astype(np.uint8)

    # Skladanie obrazu
    reconstructed = cv2.merge((Y_recon, Cr, Cb))
    final_image = cv2.cvtColor(reconstructed, cv2.COLOR_YCrCb2BGR)
    final_image = final_image[:image.shape[0], :image.shape[1]]

    # ZigZag + Kompresja
    zigzag_all = []

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = compressed_Y[i:i+8, j:j+8]
            zigzag_all.append(zigzag(block))

    zigzag_flat = np.concatenate(zigzag_all).astype(np.int16)

    compressed_data = zlib.compress(zigzag_flat.tobytes(), level=9)  # max compression
    print(f"Rozmiar po kompresji (zlib) dla QF={QF}: {len(compressed_data)} bajtow")

    # Zapis pliku JPEG (dla porownania)
    filename = f'jpeg_QF{QF}.jpg'
    cv2.imwrite(filename, final_image, [cv2.IMWRITE_JPEG_QUALITY, QF])
    size = os.path.getsize(filename)
    print(f"Rozmiar pliku JPEG dla QF={QF}: {size} bajtow")

    # Wyswietlenie porownania
    plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Oryginal')
    plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)), plt.title(f'JPEG QF={QF}')
    plt.axis('off')
    plt.show()