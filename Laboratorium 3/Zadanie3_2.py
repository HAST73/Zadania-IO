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
original_image = load_image("images/kosmos.jpg")
sizes = original_image.shape
length = sizes[0] * sizes[1] * sizes[2]

# Przygotowanie duzej wiadomosci (~80% obrazka)
hidden_message_text = "TajnyTekst" * (length // 100)

# Zakodowanie wiadomosci
twos = encode_as_binary_array(hidden_message_text)

# Wygenerowanie 8 obrazkow
images_with_messages = [hide_message(original_image, twos, n) for n in range(1,9)]

# Liczenie MSE bez petli po pikselach
MSE = np.array([np.mean((original_image.astype(float) - img.astype(float))**2) for img in images_with_messages])

# Rysowanie obrazow
fig, ax = plt.subplots(4, 2, figsize=(12, 16))
fig.tight_layout(h_pad=5)
for i, axi in enumerate(ax.flat):
    axi.imshow(images_with_messages[i])
    axi.set_title(f"nbits={i+1}\nMSE={MSE[i]:.2f}")
    axi.axis('off')

# Rysowanie wykresu
plt.figure(figsize=(8,5))
plt.plot(range(1,9), MSE, marker='o')
plt.title("MSE vs nbits")
plt.xlabel("nbits")
plt.ylabel("MSE")
plt.grid(True)
plt.show()