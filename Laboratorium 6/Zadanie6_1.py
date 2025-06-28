import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie obrazu w odcieniach szarosci
img = cv2.imread('test2.png', 0)
if img is None:
    print("Nie znaleziono pliku 'test2.png'. Upewnij sie, że plik istnieje w tym samym katalogu.")
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

# Slownik z dostepnymi elementami strukturalnymi
kernels = {
    "1": ("Krzyz 3x3", kernel_cross),
    "2": ("Kwadrat 3x3", kernel_square),
    "3": ("Pionowy pasek 7x1", kernel_vert7),
    "4": ("Sniezynka 7x7", kernel_snowflake)
}

# Funkcja do wyswietlania obrazow
def show_morphology(kernel_name, kernel):
    eroded = cv2.erode(bin_img, kernel)
    diff = cv2.subtract(bin_img, eroded)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bin_img, cmap='gray')
    axes[0].set_title(f'Oryginal ({kernel_name})')
    axes[0].axis('off')

    axes[1].imshow(eroded, cmap='gray')
    axes[1].set_title('Erozja')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='gray')
    axes[2].set_title('Roznica (Oryginal - Erozja)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Glowne menu w petli
while True:
    print("\nWybierz element strukturalny do erozji:")
    print("1. Krzyz 3x3")
    print("2. Kwadrat 3x3")
    print("3. Pionowy pasek 7x1")
    print("4. Sniezynka 7x7")
    print("0. Wyjście")

    choice = input("Podaj numer (0-4): ").strip()

    if choice in kernels:
        kernel_name, kernel = kernels[choice]
        show_morphology(kernel_name, kernel)
    elif choice == "0":
        print("Zamykanie programu...")
        break
    else:
        print("Niepoprawny wybor, sprobuj ponownie.")
