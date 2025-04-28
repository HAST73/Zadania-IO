from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

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
original_image = load_image("images/kosmos.jpg", pad=True)

# Wiadomosc, ktora chcemy ukryc
message = "Ukryta wiadomosc" * 1000
binary = encode_as_binary_array(message)

# Ustawienia kodowania
KodB = 7  # Ile bitow na piksel

# Ukrycie wiadomosci w obrazie
image_with_message = hide_message(original_image, binary, KodB, spos=1000000)

# Zapis obrazu z ukryta wiadomoscia
save_image("images/pos_message.png", image_with_message)

# Wczytanie z powrotem zapisanego obrazu
image_with_message = load_image("images/pos_message.png")

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
