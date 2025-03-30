import numpy as np
import struct
import zlib
import cv2
import matplotlib.pyplot as plt

# Parametry obrazu
steps = 17
step = 255 // steps
width, height = 122, 10

# Tworzenie obrazu tęczy
image = np.zeros((height, width, 3), dtype=np.uint8)
dummy = np.array([0, 0, 0], dtype=np.uint8)
idx = 1  # Start od pierwszej kolumny

# Przejścia kolorów
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

# Od zielonego do żółtego
for i in range(steps):
    dummy[0] += step
    image[1:-1, idx] = dummy
    idx += 1

# Od żółtego do czerwonego
for i in range(steps):
    dummy[1] -= step
    image[1:-1, idx] = dummy
    idx += 1

# Od czerwonego do magenty
for i in range(steps):
    dummy[2] += step
    image[1:-1, idx] = dummy
    idx += 1

# Od magenty do białego
for i in range(steps + 1):
    dummy[1] = min(255, dummy[1] + step)
    image[1:-1, idx] = dummy
    idx += 1

# Przygotowanie danych obrazu dla PNG (każdy wiersz z filtrem 0)
raw_data = b''.join(
    b'\x00' + image[row].tobytes() for row in range(height)
)

# Nagłówek PNG
png_signature = b'\x89PNG\r\n\x1a\n'

# Budowa chunk'a IHDR
ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
ihdr_chunk = (
    struct.pack('>I', len(ihdr_data)) +
    b'IHDR' +
    ihdr_data +
    struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data))
)

# Budowa chunk'a IDAT
compressed_data = zlib.compress(raw_data)
idat_chunk = (
    struct.pack('>I', len(compressed_data)) +
    b'IDAT' +
    compressed_data +
    struct.pack('>I', zlib.crc32(b'IDAT' + compressed_data))
)

# Budowa chunk'a IEND
iend_chunk = (
    struct.pack('>I', 0) +
    b'IEND' +
    b'' +
    struct.pack('>I', zlib.crc32(b'IEND' + b''))
)

# Zapis do pliku
with open('lab4.png', 'wb') as f:
    f.write(png_signature)
    f.write(ihdr_chunk)
    f.write(idat_chunk)
    f.write(iend_chunk)

# Weryfikacja obrazu
image_from_file = cv2.imread('lab4.png')
plt.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()