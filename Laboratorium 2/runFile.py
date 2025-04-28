import os
import struct
import zlib
from scipy.fftpack import dct, idct
import cv2
import numpy as np
from matplotlib import pyplot as plt


def Zadanie2_1():
    print("Wykonuję Zadanie 2.1")
    # Rozmiar obrazu
    width, height = 6, 6

    # Tworzenie jednolitego niebieskiego obrazu (pelny niebieski: R=0, G=0, B=255)
    blue_image = np.full((height, width, 3), [0, 0, 255], dtype=np.uint8)

    # Wyswietlenie obrazu
    plt.imshow(blue_image)
    plt.title("Wygenerowany niebieski obraz")
    plt.axis("off")
    plt.show()

    # Naglowki PPM
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
            file.readline()  # Pominiecie naglowka P3
            width, height = map(int, file.readline().split())  # Odczyt wymiarow
            file.readline()  # Pominiecie wartosci maksymalnej

            # Wczytanie pikseli
            pixels = []
            for line in file:
                pixels.extend(map(int, line.split()))

            image = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
            return image

    # Funkcja odczytu binarnego PPM
    def read_ppm_binary(file_path):
        with open(file_path, 'rb') as file:
            file.readline()  # Pominiecie naglowka P6
            width, height = map(int, file.readline().split())  # Odczyt wymiarow
            file.readline()  # Pominiecie wartosci maksymalnej

            pixels = np.fromfile(file, dtype=np.uint8)
            image = pixels.reshape((height, width, 3))
            return image

    # Odczyt obrazow
    image_from_ascii = read_ppm_ascii('blue-ascii.ppm')
    image_from_binary = read_ppm_binary('blue-binary.ppm')

    # Porownanie rozmiarow plikow
    ascii_size = os.path.getsize('blue-ascii.ppm')
    binary_size = os.path.getsize('blue-binary.ppm')

    print(f"Rozmiar pliku PPM ASCII (P3): {ascii_size} bajtów")
    print(f"Rozmiar pliku PPM Binarnego (P6): {binary_size} bajtów")

    # Wizualizacja odczytanych obrazow
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
    image = cv2.imread("blue_photo.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pobranie wymiarow
    height, width, _ = image.shape

    # Wyswietlenie oryginalnego obrazu
    plt.imshow(image)
    plt.title("Oryginalne zdjęcie")
    plt.axis("off")
    plt.show()

    # Naglowki PPM
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
            file.readline()  # Pominiecie naglowka P3
            width, height = map(int, file.readline().split())  # Odczyt wymiarow
            file.readline()  # Pominiecie wartosci maksymalnej

            # Wczytanie pikseli
            pixels = []
            for line in file:
                pixels.extend(map(int, line.split()))

            image = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
            return image

    # Funkcja odczytu binarnego PPM
    def read_ppm_binary(file_path):
        with open(file_path, 'rb') as file:
            file.readline()  # Pominiecie naglowka P6
            width, height = map(int, file.readline().split())  # Odczyt wymiarow
            file.readline()  # Pominiecie wartosci maksymalnej

            pixels = np.fromfile(file, dtype=np.uint8)
            image = pixels.reshape((height, width, 3))
            return image

    # Odczyt obrazow
    image_from_ascii = read_ppm_ascii('blue_photo_ascii.ppm')
    image_from_binary = read_ppm_binary('blue_photo_binary.ppm')

    # Porownanie rozmiarow plikow
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


def Zadanie2_2():
    print("Wykonuję Zadanie 2.2")
    # Parametry obrazu
    steps = 17
    step = 255 // steps

    # Tworzenie obrazu z czarnym obramowaniem
    image = np.zeros((10, 122, 3), dtype=np.uint8)  # 10 wierszy (8+2), 122 kolumny (120+2)

    # Tworzenie gradientu RGB
    dummy = np.array([0, 0, 0], dtype=np.uint8)
    idx = 1  # Zaczynamy od pierwszej kolumny wewnatrz obramowania

    # Przejscia kolorow
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
    # Od zielonego do zoltego
    for i in range(steps):
        dummy[0] += step
        image[1:-1, idx] = dummy
        idx += 1
    # Od zoltego do czerwonego
    for i in range(steps):
        dummy[1] -= step
        image[1:-1, idx] = dummy
        idx += 1
    # Od czerwonego do magenty
    for i in range(steps):
        dummy[2] += step
        image[1:-1, idx] = dummy
        idx += 1
    # Od magenty do bialego
    for i in range(steps + 1):
        dummy[1] = min(255, dummy[1] + step)
        image[1:-1, idx] = dummy
        idx += 1

    # Poprawny zapis PPM
    ppm_header = f'P3\n{image.shape[1]} {image.shape[0]}\n255\n'
    with open('tecza.ppm', 'w') as fh:
        fh.write(ppm_header)
        for row in image:
            for pixel in row:
                fh.write(f'{pixel[0]} {pixel[1]} {pixel[2]} ')
            fh.write('\n')

    # Wczytanie i wyswietlenie
    image_from_file = cv2.imread('tecza.ppm')
    plt.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def Zadanie2_3():
    print("Wykonuję Zadanie 2.3")
    # Parametry obrazu
    steps = 17
    step = 255 // steps
    width, height = 122, 10

    # Tworzenie obrazu teczy
    image = np.zeros((height, width, 3), dtype=np.uint8)
    dummy = np.array([0, 0, 0], dtype=np.uint8)
    idx = 1  # Start od pierwszej kolumny

    # Przejscia kolorow

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

    # Od zielonego do zoltego
    for i in range(steps):
        dummy[0] += step
        image[1:-1, idx] = dummy
        idx += 1

    # Od zoltego do czerwonego
    for i in range(steps):
        dummy[1] -= step
        image[1:-1, idx] = dummy
        idx += 1

    # Od czerwonego do magenty
    for i in range(steps):
        dummy[2] += step
        image[1:-1, idx] = dummy
        idx += 1

    # Od magenty do bialego
    for i in range(steps + 1):
        dummy[1] = min(255, dummy[1] + step)
        image[1:-1, idx] = dummy
        idx += 1

    # Przygotowanie danych obrazu dla PNG (kazdy wiersz z filtrem 0)
    raw_data = b''.join(
        b'\x00' + image[row].tobytes() for row in range(height)
    )

    # Naglowek PNG
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
    with open('tecza.png', 'wb') as f:
        f.write(png_signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)

    # Weryfikacja obrazu
    image_from_file = cv2.imread('tecza.png')
    plt.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def Zadanie2_4():
    print("Wykonuję Zadanie 2.4")
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

    # Od niebieskiego do cyjanu (Zwiększaj G)
    for i in range(steps):
        dummy[1] += step  # G
        image[1:-1, idx] = dummy
        idx += 1

    # Od cyjanu do zielonego (Zmniejszaj B)
    for i in range(steps):
        dummy[0] -= step  # B
        image[1:-1, idx] = dummy
        idx += 1

    # Od zielonego do zoltego (Zwiększaj R)
    for i in range(steps):
        dummy[2] += step  # R
        image[1:-1, idx] = dummy
        idx += 1

    # Od zoltego do czerwonego (Zmniejszaj G)
    for i in range(steps):
        dummy[1] -= step  # G
        image[1:-1, idx] = dummy
        idx += 1

    # Od czerwonego do magenty (Zwiększaj B)
    for i in range(steps):
        dummy[0] += step  # B
        image[1:-1, idx] = dummy
        idx += 1

    # Od magenty do bialego (Zwiększaj G)
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

    # Konwersja na YCrCb
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(image_ycrcb)

    # Dodanie paddingu
    Y = pad_image(Y)
    Cr = pad_image(Cr)
    Cb = pad_image(Cb)

    # Probkowanie chrominancji
    sampling_factors = [1, 2, 4]

    for factor in sampling_factors:
        # Zmniejszenie rozdzielczosci kanalow Cr i Cb
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

        # Przeprowadzanie transformacji DCT i kwantyzacji dla bloku 8x8
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = Y[i:i + 8, j:j + 8].astype(np.float32) - 128
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                quantized = np.round(dct_block / QY)
                compressed_Y[i:i + 8, j:j + 8] = quantized

        # Odtwarzanie obrazu po dekwantyzacji i odwrotnej DCT
        reconstructed_Y = np.zeros_like(compressed_Y, dtype=np.float32)

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = compressed_Y[i:i + 8, j:j + 8]
                dequantized = block * QY
                idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho') + 128
                reconstructed_Y[i:i + 8, j:j + 8] = idct_block

        Y_recon = np.clip(reconstructed_Y, 0, 255).astype(np.uint8)

        # Przywrocenie pelnej rozdzielczosci chrominancji
        Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_CUBIC)
        Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_CUBIC)

        # Skladanie koncowego obrazu
        reconstructed = cv2.merge((Y_recon, Cr_up, Cb_up))
        final_image = cv2.cvtColor(reconstructed, cv2.COLOR_YCrCb2BGR)
        final_image = final_image[:image.shape[0], :image.shape[1]]

        # Zapis obrazu jako plik JPEG
        cv2.imwrite(f'jpeg_{factor}x.jpg', final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        size = os.path.getsize(f'jpeg_{factor}x.jpg')
        print(f'Rozmiar dla próbkowania {factor}x: {size} bajtów')

        # Wizualizacja porownania oryginalu i skompresowanego obrazu
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Oryginał')
        plt.axis('off')
        plt.subplot(122), plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)), plt.title(f'Próbkowanie {factor}x')
        plt.axis('off')
        plt.show()


def main():
    while True:
        print("\nWybierz zadanie do uruchomienia:")
        print("1. Zadanie2_1")
        print("2. Zadanie2_2")
        print("3. Zadanie2_3")
        print("4. Zadanie2_4")
        print("0. Wyjście")

        choice = input("Podaj numer: ")

        if choice == "1":
            Zadanie2_1()
        elif choice == "2":
            Zadanie2_2()
        elif choice == "3":
            Zadanie2_3()
        elif choice == "4":
            Zadanie2_4()
        elif choice == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")


if __name__ == "__main__":
    main()

