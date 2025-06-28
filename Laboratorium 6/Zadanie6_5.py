import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funkcja pomocnicza: przetwarzanie jednego obrazu i jednej struktury
def process_image(filename, name, kernel):
    img = cv2.imread(filename, 0)
    if img is None:
        print(f"Nie znaleziono pliku: {filename}")
        return

    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(bin_img, kernel)
    eroded = cv2.erode(bin_img, kernel)
    gradient = cv2.subtract(dilated, eroded)

    laplacian = cv2.add(dilated, eroded)
    laplacian = cv2.subtract(laplacian, 2 * bin_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{filename} — Struktura: {name}', fontsize=16)

    axes[0].imshow(gradient, cmap='gray')
    axes[0].set_title('Gradient morfologiczny')
    axes[0].axis('off')

    axes[1].imshow(laplacian, cmap='gray')
    axes[1].set_title('Laplasjan morfologiczny')
    axes[1].axis('off')

    axes[2].imshow(bin_img, cmap='gray')
    axes[2].set_title('Oryginal binarny')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# Elementy strukturalne
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

# Menu slownik
kernel_options = {
    "1": ("Krzyz 3x3", kernel_cross),
    "2": ("Kwadrat 3x3", kernel_square),
    "3": ("Pionowy pasek 7x1", kernel_vert7),
    "4": ("Sniezynka 7x7", kernel_snowflake)
}

# Menu wyboru struktury
while True:
    print("\nWybierz element strukturalny:")
    print("1. Krzyz 3x3")
    print("2. Kwadrat 3x3")
    print("3. Pionowy pasek 7x1")
    print("4. Sniezynka 7x7")
    print("0. Wyjscie")

    choice = input("Podaj numer: ").strip()

    if choice == "0":
        print("Zamykanie programu...")
        break
    elif choice in kernel_options:
        name, kernel = kernel_options[choice]
        print(f"\nPrzetwarzanie obrazow z wybraną struktura: {name}")
        process_image('test1.png', name, kernel)
        process_image('test2.png', name, kernel)
    else:
        print("Niepoprawny wybor. Sprobuj ponownie.")
