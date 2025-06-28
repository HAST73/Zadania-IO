import cv2
import numpy as np
from matplotlib import pyplot as plt

def Zadanie6_1():
    print("Wykonuję Zadanie 6_1")
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


def Zadanie6_2():
    print("Wykonuję Zadanie 6_2")
    # Wczytanie obrazu w odcieniach szarosci
    img = cv2.imread('test1.png', 0)
    if img is None:
        print("Nie znaleziono pliku 'test1.png'. Upewnij sie, ze plik jest w tym samym katalogu.")
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

    # Funkcja do wyswietlania obrazow
    def show_morph_ops(name, kernel):
        eroded = cv2.erode(bin_img, kernel)
        dilated = cv2.dilate(bin_img, kernel)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(bin_img, cmap='gray')
        axes[0].set_title(f'Oryginal ({name})')
        axes[0].axis('off')

        axes[1].imshow(eroded, cmap='gray')
        axes[1].set_title('Erozja')
        axes[1].axis('off')

        axes[2].imshow(dilated, cmap='gray')
        axes[2].set_title('Dylacja')
        axes[2].axis('off')

        plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
        plt.show()

    # Menu w pętli
    while True:
        print("\nWybierz element strukturalny do przetwarzania:")
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
            show_morph_ops(name, kernel)
        else:
            print("Niepoprawny wybor, sprobuj ponownie.")


def Zadanie6_3():
    print("Wykonuję Zadanie 6_3")
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


def Zadanie6_4():
    print("Wykonuję Zadanie 6_4")
    # Wczytanie obrazu w odcieniach szarosci
    img = cv2.imread('test3.png', 0)
    if img is None:
        print("Nie znaleziono pliku 'test3.png'. Upewnij sie, ze plik istnieje w katalogu.")
        exit()

    # Binarizacja obrazu (próg = 127)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Definicje elementów strukturalnych
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

    # Funkcja do operacji otwarcia i wyswietlenia wynikow
    def show_opening(name, kernel):
        opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        diff = cv2.absdiff(bin_img, opened)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(bin_img, cmap='gray')
        axes[0].set_title(f'Oryginal ({name})')
        axes[0].axis('off')

        axes[1].imshow(opened, cmap='gray')
        axes[1].set_title('Otwarcie')
        axes[1].axis('off')

        axes[2].imshow(diff, cmap='gray')
        axes[2].set_title('Roznica (Oryginal - Otwarcie)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    # Menu wyboru w pętli
    while True:
        print("\nWybierz element strukturalny do otwarcia:")
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
            show_opening(name, kernel)
        else:
            print("Niepoprawny wybor, sprobuj ponownie.")

def Zadanie6_5():
    print("Wykonuję Zadanie 6_5")

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


def main():
    while True:
        print("\nWybierz zadanie do uruchomienia:")
        print("1. Zadanie6_1")
        print("2. Zadanie6_2")
        print("3. Zadanie6_3")
        print("4. Zadanie6_4")
        print("5. Zadanie6_5")
        print("0. Wyjście")

        choice = input("Podaj numer: ")

        if choice == "1":
            Zadanie6_1()
        elif choice == "2":
            Zadanie6_2()
        elif choice == "3":
            Zadanie6_3()
        elif choice == "4":
            Zadanie6_4()
        elif choice == "5":
            Zadanie6_5()
        elif choice == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")


if __name__ == "__main__":
    main()

