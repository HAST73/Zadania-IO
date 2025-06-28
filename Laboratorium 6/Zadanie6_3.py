import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie obrazu w odcieniach szarosci
img = cv2.imread('test3.png', 0)
if img is None:
    print("Nie znaleziono pliku 'test3.png'. Upewnij sie, ze plik istnieje w katalogu.")
    exit()

# Binarizacja obrazu (prog = 127)
_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Definicje elementow strukturalnych
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_vert7 = np.array([[0], [1], [1], [1], [1], [1], [0]], dtype=np.uint8)
kernel_snowflake = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0]
], dtype=np.uint8)

# Slownik z nazwami i strukturami
kernels = {
    "1": ("Krzyz 3x3", kernel_cross),
    "2": ("Kwadrat 3x3", kernel_square),
    "3": ("Pionowy pasek 7x1", kernel_vert7),
    "4": ("Sniezynka 7x7", kernel_snowflake)
}

# Funkcja do przetwarzania i wyswietlania
def show_closing(name, kernel):
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    diff = cv2.absdiff(closed, bin_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bin_img, cmap='gray')
    axes[0].set_title(f'Oryginal ({name})')
    axes[0].axis('off')

    axes[1].imshow(closed, cmap='gray')
    axes[1].set_title('Domkniecie')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='gray')
    axes[2].set_title('Roznica (Domkniecie - Oryginal)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Menu w pętli
while True:
    print("\nWybierz element strukturalny do domkniecia:")
    print("1. Krzyz 3x3")
    print("2. Kwadrat 3x3")
    print("3. Pionowy pasek 7x1")
    print("4. Sniezynka 7x7")
    print("0. Wyjscie")

    choice = input("Podaj numer (0-4): ").strip()

    if choice == "0":
        print("Zamykanie programu...")
        break
    elif choice in kernels:
        name, kernel = kernels[choice]
        show_closing(name, kernel)
    else:
        print("Niepoprawny wybor, sprobuj ponownie.")
