import numpy as np
import binascii
import cv2 as cv
import math
import zlib
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os



def Zadanie2_5():
    print("Wykonuję Zadanie 2_5")
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
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
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
            (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
            (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
            (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
            (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
            (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
            (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
            (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
            (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
        ]
        return np.array([block[i, j] for i, j in zigzag_order])

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
                block = Y[i:i + 8, j:j + 8].astype(np.float32) - 128
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                quantized = np.round(dct_block / Q)
                compressed_Y[i:i + 8, j:j + 8] = quantized

        # Odtwarzanie obrazu: dekwantyzacja + IDCT
        reconstructed_Y = np.zeros_like(compressed_Y, dtype=np.float32)

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = compressed_Y[i:i + 8, j:j + 8]
                dequantized = block * Q
                idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho') + 128
                reconstructed_Y[i:i + 8, j:j + 8] = idct_block

        Y_recon = np.clip(reconstructed_Y, 0, 255).astype(np.uint8)

        # Skladanie obrazu
        reconstructed = cv2.merge((Y_recon, Cr, Cb))
        final_image = cv2.cvtColor(reconstructed, cv2.COLOR_YCrCb2BGR)
        final_image = final_image[:image.shape[0], :image.shape[1]]

        # ZigZag + Kompresja
        zigzag_all = []

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = compressed_Y[i:i + 8, j:j + 8]
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
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Oryginal')
        plt.axis('off')
        plt.subplot(122), plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)), plt.title(f'JPEG QF={QF}')
        plt.axis('off')
        plt.show()


def Zadanie3_1():
    print("Wykonuję Zadanie 3_1")
    # Ustawienie rozmiaru wykresu
    plt.rcParams["figure.figsize"] = (18, 10)

    def encode_as_binary_array(msg):
        """Zakoduj wiadomosc jako ciag binarny"""
        msg = msg.encode("utf-8")
        msg = msg.hex()
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
        msg = ["{:08b}".format(int(el, base=16)) for el in msg]
        return "".join(msg)

    def decode_from_binary_array(array):
        """Dekoduj ciag binarny na wiadomosc UTF-8"""
        array = [array[i:i + 8] for i in range(0, len(array), 8)]
        if len(array[-1]) != 8:
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))
        array = ["{:02x}".format(int(el, 2)) for el in array]
        array = "".join(array)
        result = binascii.unhexlify(array)
        return result.decode("utf-8", errors="replace")

    def load_image(path, pad=False):
        """Wczytaj obrazek

        Jesli 'pad' jest ustawiony, obrazek zostanie uzupelniony zerami do wielokrotnosci 8 pikseli
        """
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if pad:
            y_pad = 8 - (image.shape[0] % 8)
            x_pad = 8 - (image.shape[1] % 8)
            image = np.pad(
                image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
        return image

    def save_image(path, image):
        """Zapisz obrazek na dysku"""
        plt.imsave(path, image)

    def clamp(n, minn, maxn):
        """Ogranicz wartosc 'n' do zakresu od 'minn' do 'maxn'"""
        return max(min(maxn, n), minn)

    def hide_message(image, message, nbits=1):
        """Ukryj wiadomosc w obrazie (LSB - najmlodsze bity)

        nbits: liczba najmlodszych bitow uzywanych do ukrycia wiadomosci
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        if len(message) > len(image) * nbits:
            raise ValueError("Wiadomosc jest za dluga :(")

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
        for i, chunk in enumerate(chunks):
            byte = "{:08b}".format(image[i])
            new_byte = byte[:-nbits] + chunk
            image[i] = int(new_byte, 2)

        return image.reshape(shape)

    def reveal_message(image, nbits=1, length=0):
        """Odczytaj ukryta wiadomosc z obrazu

        nbits: liczba najmlodszych bitow do odczytania.
        length: dlugosc wiadomosci w bitach.
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        length_in_pixels = math.ceil(length / nbits)
        if len(image) < length_in_pixels or length_in_pixels <= 0:
            length_in_pixels = len(image)

        message = ""
        i = 0
        while i < length_in_pixels:
            byte = "{:08b}".format(image[i])
            message += byte[-nbits:]
            i += 1

        mod = length % -nbits
        if mod != 0:
            message = message[:mod]
        return message

    # Ustalenie sciezki do obrazu oraz wiadomosc, ktora chcemy ukryc
    image_path = 'kosmos.jpg'
    original_image = load_image(image_path)

    # Wiadomosc do ukrycia
    message = "Ukryta wiadomosc" * 1
    n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości

    # Zakodowanie wiadomosci jako ciag binarny
    binary_message = encode_as_binary_array(message)

    # Ukrycie wiadomosci w obrazie
    image_with_message = hide_message(original_image, binary_message, n)

    # Zapisanie obrazow w formatach PNG i JPG
    save_image("image_with_message.png", image_with_message)
    save_image("image_with_message.jpg", image_with_message)

    # Wczytanie obrazkow w formatach PNG i JPG
    image_with_message_png = load_image("image_with_message.png")
    image_with_message_jpg = load_image("image_with_message.jpg")

    # Odczytanie ukrytej wiadomosci z obrazkow
    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=n, length=len(binary_message)))
    secret_message_jpg = decode_from_binary_array(
        reveal_message(image_with_message_jpg, nbits=n, length=len(binary_message)))

    # Wyswietlanie ukrytej wiadomosci
    print(f"Tajna wiadomosc z PNG: {secret_message_png}")
    print(f"Tajna wiadomosc z JPG: {secret_message_jpg}")

    # Wyswietlenie obrazkow w jednym wykresie
    f, ar = plt.subplots(2, 2)

    # Oryginalny obraz
    ar[0, 0].imshow(original_image)
    ar[0, 0].set_title("Original image")
    ar[0, 0].axis('off')

    # Obraz z ukryta wiadomoscia
    ar[0, 1].imshow(image_with_message)
    ar[0, 1].set_title("Image with hidden message")
    ar[0, 1].axis('off')

    # Obraz PNG
    ar[1, 0].imshow(image_with_message_png)
    ar[1, 0].set_title("PNG image")
    ar[1, 0].axis('off')

    # Obraz JPG
    ar[1, 1].imshow(image_with_message_jpg)
    ar[1, 1].set_title("JPG image")
    ar[1, 1].axis('off')

    # Wyswietlanie wykresu
    plt.show()


def Zadanie3_2():
    print("Wykonuję Zadanie 3.2")
    # Ustawienie rozmiaru wykresu
    plt.rcParams["figure.figsize"] = (18, 10)

    def encode_as_binary_array(msg):
        """Zakoduj wiadomosc jako ciag binarny"""
        msg = msg.encode("utf-8")
        msg = msg.hex()
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
        msg = ["{:08b}".format(int(el, base=16)) for el in msg]
        return "".join(msg)

    def decode_from_binary_array(array):
        """Dekoduj ciag binarny na wiadomosc UTF-8"""
        array = [array[i:i + 8] for i in range(0, len(array), 8)]
        if len(array[-1]) != 8:
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))
        array = ["{:02x}".format(int(el, 2)) for el in array]
        array = "".join(array)
        result = binascii.unhexlify(array)
        return result.decode("utf-8", errors="replace")

    def load_image(path, pad=False):
        """Wczytaj obrazek

        Jesli 'pad' jest ustawiony, obrazek zostanie uzupelniony zerami do wielokrotnosci 8 pikseli
        """
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if pad:
            y_pad = 8 - (image.shape[0] % 8)
            x_pad = 8 - (image.shape[1] % 8)
            image = np.pad(
                image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
        return image

    def save_image(path, image):
        """Zapisz obrazek na dysku"""
        plt.imsave(path, image)

    def clamp(n, minn, maxn):
        """Ogranicz wartosc 'n' do zakresu od 'minn' do 'maxn'"""
        return max(min(maxn, n), minn)

    def hide_message(image, message, nbits=1):
        """Ukryj wiadomosc w obrazie (LSB - najmlodsze bity)

        nbits: liczba najmlodszych bitow uzywanych do ukrycia wiadomosci
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        if len(message) > len(image) * nbits:
            raise ValueError("Wiadomosc jest za dluga :(")

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
        for i, chunk in enumerate(chunks):
            byte = "{:08b}".format(image[i])
            new_byte = byte[:-nbits] + chunk
            image[i] = int(new_byte, 2)

        return image.reshape(shape)

    def reveal_message(image, nbits=1, length=0):
        """Odczytaj ukryta wiadomosc z obrazu

        nbits: liczba najmlodszych bitow do odczytania.
        length: dlugosc wiadomosci w bitach.
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        length_in_pixels = math.ceil(length / nbits)
        if len(image) < length_in_pixels or length_in_pixels <= 0:
            length_in_pixels = len(image)

        message = ""
        i = 0
        while i < length_in_pixels:
            byte = "{:08b}".format(image[i])
            message += byte[-nbits:]
            i += 1

        mod = length % -nbits
        if mod != 0:
            message = message[:mod]
        return message

    # Wczytanie obrazu
    original_image = load_image("kosmos.jpg")
    sizes = original_image.shape
    length = sizes[0] * sizes[1] * sizes[2]

    # Przygotowanie duzej wiadomosci (~80% obrazka)
    hidden_message_text = "TajnyTekst" * (length // 100)

    # Zakodowanie wiadomosci
    twos = encode_as_binary_array(hidden_message_text)

    # Wygenerowanie 8 obrazkow
    images_with_messages = [hide_message(original_image, twos, n) for n in range(1, 9)]

    # Liczenie MSE bez petli po pikselach
    MSE = np.array([np.mean((original_image.astype(float) - img.astype(float)) ** 2) for img in images_with_messages])

    # Rysowanie obrazow
    fig, ax = plt.subplots(4, 2, figsize=(12, 16))
    fig.tight_layout(h_pad=5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(images_with_messages[i])
        axi.set_title(f"nbits={i + 1}\nMSE={MSE[i]:.2f}")
        axi.axis('off')

    # Rysowanie wykresu
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 9), MSE, marker='o')
    plt.title("MSE vs nbits")
    plt.xlabel("nbits")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()


def Zadanie3_3():
    print("Wykonuję Zadanie 3.3")
    # Ustawienie rozmiaru wykresu
    plt.rcParams["figure.figsize"] = (18, 10)

    def encode_as_binary_array(msg):
        """Zakoduj wiadomosc jako ciag binarny"""
        msg = msg.encode("utf-8")
        msg = msg.hex()
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
        msg = ["{:08b}".format(int(el, base=16)) for el in msg]
        return "".join(msg)

    def decode_from_binary_array(array):
        """Dekoduj ciag binarny na wiadomosc UTF-8"""
        array = [array[i:i + 8] for i in range(0, len(array), 8)]
        if len(array[-1]) != 8:
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))
        array = ["{:02x}".format(int(el, 2)) for el in array]
        array = "".join(array)
        result = binascii.unhexlify(array)
        return result.decode("utf-8", errors="replace")

    def load_image(path, pad=False):
        """Wczytaj obrazek

        Jesli 'pad' jest ustawiony, obrazek zostanie uzupelniony zerami do wielokrotnosci 8 pikseli
        """
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if pad:
            y_pad = 8 - (image.shape[0] % 8)
            x_pad = 8 - (image.shape[1] % 8)
            image = np.pad(
                image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
        return image

    def save_image(path, image):
        """Zapisz obrazek na dysku"""
        plt.imsave(path, image)

    def clamp(n, minn, maxn):
        """Ogranicz wartosc 'n' do zakresu od 'minn' do 'maxn'"""
        return max(min(maxn, n), minn)

    def hide_message(image, message, nbits=1, spos=0):
        """Ukryj wiadomosc w obrazie (LSB - najmlodsze bity)

        nbits: liczba najmlodszych bitow uzywanych do ukrycia wiadomosci
        spos: indeks startowy w obrazie od ktorego zaczynamy ukrywanie
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        length = len(message)
        image = np.copy(image).flatten()
        if length > (len(image) - spos) * nbits:
            raise ValueError("Wiadomosc jest za dluga :(")

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
        for i, chunk in enumerate(chunks):
            byte = "{:08b}".format(image[i + spos])
            new_byte = byte[:-nbits] + chunk
            image[spos + i] = int(new_byte, 2)

        return image.reshape(shape)

    def reveal_message(image, nbits=1, length=0, spos=0):
        """Odczytaj ukryta wiadomosc z obrazu

        nbits: liczba najmlodszych bitow do odczytania
        length: dlugosc wiadomosci w bitach
        spos: indeks startowy w obrazie, od ktorego zaczynamy odczyt
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        length_in_pixels = math.ceil(length / nbits)
        if len(image) < length_in_pixels or length_in_pixels <= 0:
            length_in_pixels = len(image) - spos

        message = ""
        i = 0
        while i < length_in_pixels:
            byte = "{:08b}".format(image[i + spos])
            message += byte[-nbits:]
            i += 1

        mod = length % -nbits
        if mod != 0:
            message = message[:mod]
        return message

    # Wczytanie oryginalnego obrazu
    original_image = load_image("kosmos.jpg", pad=True)

    # Wiadomosc, ktora chcemy ukryc
    message = "Ukryta wiadomosc" * 1000
    binary = encode_as_binary_array(message)

    # Ustawienia kodowania
    KodB = 7  # Ile bitow na piksel

    # Ukrycie wiadomosci w obrazie
    image_with_message = hide_message(original_image, binary, KodB, spos=1000000)

    # Zapis obrazu z ukryta wiadomoscia
    save_image("pos_message.png", image_with_message)

    # Wczytanie z powrotem zapisanego obrazu
    image_with_message = load_image("pos_message.png")

    # Odkrycie wiadomosci
    secret_message = decode_from_binary_array(
        reveal_message(image_with_message, KodB, length=len(binary), spos=1000000)
    )

    # Wyswietlenie wiadomosci
    print(secret_message)

    # Wyswietlenie obrazow
    f, ar = plt.subplots(1, 2)
    ar[0].imshow(original_image)
    ar[0].set_title("Original image")
    ar[1].imshow(image_with_message)
    ar[1].set_title("Image with hidden message")
    plt.show()

def Zadanie3_4():
    print("Wykonuję Zadanie 3.4")
    # Ustawienie rozmiaru wykresu
    plt.rcParams["figure.figsize"] = (18, 10)

    def encode_as_binary_array(msg):
        """Zakoduj wiadomosc jako ciag binarny"""
        msg = msg.encode("utf-8")
        msg = msg.hex()
        msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
        msg = ["{:08b}".format(int(el, base=16)) for el in msg]
        return "".join(msg)

    def decode_from_binary_array(array):
        """Dekoduj ciag binarny na wiadomosc UTF-8"""
        array = [array[i:i + 8] for i in range(0, len(array), 8)]
        if len(array[-1]) != 8:
            array[-1] = array[-1] + "0" * (8 - len(array[-1]))
        array = ["{:02x}".format(int(el, 2)) for el in array]
        array = "".join(array)
        result = binascii.unhexlify(array)
        return result.decode("utf-8", errors="replace")

    def load_image(path, pad=False):
        """Wczytaj obrazek

        Jesli 'pad' jest ustawiony, obrazek zostanie uzupelniony zerami do wielokrotnosci 8 pikseli
        """
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if pad:
            y_pad = 8 - (image.shape[0] % 8)
            x_pad = 8 - (image.shape[1] % 8)
            image = np.pad(
                image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
        return image

    def save_image(path, image):
        """Zapisz obrazek na dysku"""
        plt.imsave(path, image)

    def clamp(n, minn, maxn):
        """Ogranicz wartosc 'n' do zakresu od 'minn' do 'maxn'"""
        return max(min(maxn, n), minn)

    def hide_message(image, message, nbits=1):
        """Ukryj wiadomosc w obrazie (LSB - najmlodsze bity)

        nbits: liczba najmlodszych bitow uzywanych do ukrycia wiadomosci
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        if len(message) > len(image) * nbits:
            raise ValueError("Wiadomosc jest za dluga :(")

        chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
        for i, chunk in enumerate(chunks):
            byte = "{:08b}".format(image[i])
            new_byte = byte[:-nbits] + chunk
            image[i] = int(new_byte, 2)

        return image.reshape(shape)

    def reveal_message(image, nbits=1, length=0):
        """Odczytaj ukryta wiadomosc z obrazu

        nbits: liczba najmlodszych bitow do odczytania.
        length: dlugosc wiadomosci w bitach.
        """
        nbits = clamp(nbits, 1, 8)
        shape = image.shape
        image = np.copy(image).flatten()
        length_in_pixels = math.ceil(length / nbits)
        if len(image) < length_in_pixels or length_in_pixels <= 0:
            length_in_pixels = len(image)

        message = ""
        i = 0
        while i < length_in_pixels:
            byte = "{:08b}".format(image[i])
            message += byte[-nbits:]
            i += 1

        mod = length % -nbits
        if mod != 0:
            message = message[:mod]
        return message

    def hide_image(image, secret_image_path, nbits=1):
        """Ukryj obraz wewnatrz innego obrazu"""
        with open(secret_image_path, "rb") as file:
            secret_img = file.read()
            secret_img = secret_img.hex()
            secret_img = [secret_img[i:i + 2] for i in range(0, len(secret_img), 2)]
            secret_img = ["{:08b}".format(int(el, base=16)) for el in secret_img]
            secret_img = "".join(secret_img)
        encoded_image = hide_message(image, secret_img, nbits)
        return encoded_image, len(secret_img)

    def recover_image(image_with_secret, length, nbits, output_path):
        """Odzyskaj ukryty obraz z innego obrazu i zapisz go"""
        secret_bits = reveal_message(image_with_secret, nbits=nbits, length=length)
        secret_bytes = [secret_bits[i:i + 8] for i in range(0, len(secret_bits), 8)]
        secret_bytes = [int(b, 2) for b in secret_bytes]
        secret_bytes = bytes(secret_bytes)

        with open(output_path, "wb") as f:
            f.write(secret_bytes)

    # Zaladowanie obrazu bazowego
    carrier_image = load_image("kosmos.jpg")

    # Zaladowanie obrazka, ktory chcemy ukryc
    secret_image = open("star.jpg", "rb").read()

    # Ukrycie obrazka w obrazie
    image_with_secret, secret_length = hide_image(carrier_image, "star.jpg", nbits=2)

    # Odzyskanie ukrytego obrazka z pamieci
    secret_bits = reveal_message(image_with_secret, nbits=2, length=secret_length)
    secret_bytes = [secret_bits[i:i + 8] for i in range(0, len(secret_bits), 8)]
    secret_bytes = [int(b, 2) for b in secret_bytes]
    secret_bytes = bytes(secret_bytes)

    # Proba odczytania odzyskanego obrazu
    import io
    from PIL import Image

    recovered_image = Image.open(io.BytesIO(secret_bytes))
    recovered_image = np.array(recovered_image)

    # Wyswietlenie wszystkiego razem

    fig, axs = plt.subplots(2, 2)

    # Oryginalny nosnik
    axs[0, 0].imshow(carrier_image)
    axs[0, 0].set_title("Carrier image cosmos")
    axs[0, 0].axis('off')

    # Oryginalny ukrywany obrazek
    original_secret_img = Image.open("star.jpg")
    original_secret_img = np.array(original_secret_img)
    axs[0, 1].imshow(original_secret_img)
    axs[0, 1].set_title("Image to hide star")
    axs[0, 1].axis('off')

    # Obraz z ukrytym obrazem
    axs[1, 0].imshow(image_with_secret)
    axs[1, 0].set_title("Image with hidden star")
    axs[1, 0].axis('off')

    # Odzyskany obraz
    axs[1, 1].imshow(recovered_image)
    axs[1, 1].set_title("Recovered star image")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    while True:
        print("\nWybierz zadanie do uruchomienia:")
        print("1. Zadanie2_5")
        print("2. Zadanie3_1")
        print("3. Zadanie3_2")
        print("4. Zadanie3_3")
        print("5. Zadanie3_4")
        print("0. Wyjście")

        choice = input("Podaj numer: ")

        if choice == "1":
            Zadanie2_5()
        elif choice == "2":
            Zadanie3_1()
        elif choice == "3":
            Zadanie3_2()
        elif choice == "4":
            Zadanie3_3()
        elif choice == "5":
            Zadanie3_4()
        elif choice == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")


if __name__ == "__main__":
    main()

