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
carrier_image = load_image("images/kosmos.jpg")

# Zaladowanie obrazka, ktory chcemy ukryc
secret_image = open("images/star.jpg", "rb").read()

# Ukrycie obrazka w obrazie
image_with_secret, secret_length = hide_image(carrier_image, "images/star.jpg", nbits=2)

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
original_secret_img = Image.open("images/star.jpg")
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