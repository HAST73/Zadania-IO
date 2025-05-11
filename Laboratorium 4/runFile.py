import cv2
import numpy as np
from matplotlib import pyplot as plt

def Zadanie4_1():
    print("Wykonuję Zadanie 4_1")
    # Wczytanie dwoch roznych obrazow
    image1 = cv2.imread('himalayas.jpg')
    image2 = cv2.imread('himalayas-high-resolution.jpg')

    # Konwersja do skali szarosci
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Kopie obrazow jako int do obliczen
    output1 = image_gray1.astype(np.int32)
    output2 = image_gray2.astype(np.int32)

    # Funkcja kwantyzacji
    def find_closest_grey_color(pixel):
        return round(pixel / 255) * 255

    # Dithering na pierwszym obrazie
    for y in range(output1.shape[0] - 1):
        for x in range(1, output1.shape[1] - 1):
            oldpixel = output1[y, x]
            newpixel = find_closest_grey_color(oldpixel)
            output1[y, x] = newpixel
            quant_error = oldpixel - newpixel

            output1[y, x + 1] += quant_error * 7 / 16
            output1[y + 1, x - 1] += quant_error * 3 / 16
            output1[y + 1, x] += quant_error * 5 / 16
            output1[y + 1, x + 1] += quant_error * 1 / 16

    # Dithering na drugim obrazie
    for y in range(output2.shape[0] - 1):
        for x in range(1, output2.shape[1] - 1):
            oldpixel = output2[y, x]
            newpixel = find_closest_grey_color(oldpixel)
            output2[y, x] = newpixel
            quant_error = oldpixel - newpixel

            output2[y, x + 1] += quant_error * 7 / 16
            output2[y + 1, x - 1] += quant_error * 3 / 16
            output2[y + 1, x] += quant_error * 5 / 16
            output2[y + 1, x + 1] += quant_error * 1 / 16

    # Przyciecie i konwersja do uint8
    output1 = np.clip(output1, 0, 255).astype(np.uint8)
    output2 = np.clip(output2, 0, 255).astype(np.uint8)

    # Wyswietlenie obu wynikow obok siebie
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(output1, cmap='gray')
    plt.title('Dithering obrazu o obnizonej rozdzielczosci')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output2, cmap='gray')
    plt.title('Dithering obrazu o zwiekszonej rozdzielczosci')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Histogram
    histr = cv2.calcHist([output2], [0], None, [256], [0, 256])

    plt.figure(figsize=(12, 6))
    plt.plot(histr, label='himalayas-high-resolution.jpg')
    plt.xlim([0, 256])
    plt.xlabel('Wartosc skladowej koloru [0-255]')
    plt.ylabel('Liczba pikseli')
    plt.title('Histogram jasnosci po ditheringu')
    plt.legend()
    plt.grid(True)
    plt.show()


def Zadanie4_2():
    print("Wykonuję Zadanie 4_2")
    # Wczytanie obrazu i konwersja do RGB (dla matplotliba)
    image = cv2.imread('modern.jpg')
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Konfiguracja: liczba poziomow kolorow
    tones = 8  # np. 2 (cz-b), 4, 8, 16

    # Przygotowanie danych wyjsciowych
    output = np.copy(image_color).astype(int)
    reduced = np.copy(image_color).astype(int)

    # Funkcja kwantyzacji do palety
    def find_closest_palette_color(pix, k=2):
        return round((k - 1) * pix / 255) * 255 / (k - 1)

    # Redukcja kolorow (bez ditheringu) — przetwarzamy caly obraz
    for y in range(reduced.shape[0]):
        for x in range(reduced.shape[1]):
            for i in range(3):  # R, G, B
                reduced[y, x, i] = find_closest_palette_color(reduced[y, x, i], tones)

    # Dithering (Floyd–Steinberg)
    for y in range(output.shape[0] - 1):
        for x in range(1, output.shape[1] - 1):
            oldpixel = output[y, x].copy()
            newpixel = np.array([
                find_closest_palette_color(oldpixel[0], tones),
                find_closest_palette_color(oldpixel[1], tones),
                find_closest_palette_color(oldpixel[2], tones)
            ])
            output[y, x] = newpixel
            quant_error = oldpixel - newpixel
            for i in range(3):
                output[y, x + 1, i] += quant_error[i] * 7 / 16
                output[y + 1, x - 1, i] += quant_error[i] * 3 / 16
                output[y + 1, x, i] += quant_error[i] * 5 / 16
                output[y + 1, x + 1, i] += quant_error[i] * 1 / 16

    # Przyciecie wartosci i konwersja do uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    reduced = np.clip(reduced, 0, 255).astype(np.uint8)

    # Wyswietlenie obrazow
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(reduced, interpolation='nearest')
    ax[0].set_title(f'Redukcja kolorów ({tones} tonow)')
    ax[0].axis('off')

    ax[1].imshow(output, interpolation='nearest')
    ax[1].set_title(f'Dithering Floyda–Steinberga ({tones} tonow)')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Histogram RGB
    colors = ('r', 'g', 'b')
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    for i, col in enumerate(colors):
        hist_reduced = cv2.calcHist([reduced], [i], None, [256], [0, 256])
        hist_dithered = cv2.calcHist([output], [i], None, [256], [0, 256])

        ax[0].plot(hist_reduced, color=col)
        ax[1].plot(hist_dithered, color=col)

    ax[0].set_title('Histogram RGB – obraz po redukcji')
    ax[0].set_xlabel('Wartosc koloru [0-255]')
    ax[0].set_ylabel('Liczba pikseli')

    ax[1].set_title('Histogram RGB – po ditheringu')
    ax[1].set_xlabel('Wartosc koloru [0-255]')
    ax[1].set_ylabel('Liczba pikseli')

    plt.tight_layout()
    plt.show()


def Zadanie4_3():
    print("Wykonuję Zadanie 4.3")
    width = 80
    height = 60
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Funkcja do rysowania punktu
    def draw_point(image, x, y, color=(255, 255, 255)):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[image.shape[0] - 1 - y, x] = color

    # Algorytm Bresenhama do rysowania linii
    def draw_line(image, x1, y1, x2, y2, color=(255, 255, 255)):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = np.sign(x2 - x1)
        sy = np.sign(y2 - y1)
        x, y = x1, y1

        if dx > dy:
            err = dx // 2
            for _ in range(dx + 1):
                draw_point(image, x, y, color)
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
        else:
            err = dy // 2
            for _ in range(dy + 1):
                draw_point(image, x, y, color)
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy

    # Klasa wierzcholka
    class Vertex:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Funkcja do rysowania wypelnionego trojkata
    def draw_triangle(image, A, B, C, color=(255, 255, 255)):
        xmin = min(A.x, B.x, C.x)
        xmax = max(A.x, B.x, C.x)
        ymin = min(A.y, B.y, C.y)
        ymax = max(A.y, B.y, C.y)

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                # Obliczamy iloczyny wektorowe (pole trojkata ×2)
                ar1 = (x - A.x) * (B.y - A.y) - (y - A.y) * (B.x - A.x)
                ar2 = (x - B.x) * (C.y - B.y) - (y - B.y) * (C.x - B.x)
                ar3 = (x - C.x) * (A.y - C.y) - (y - C.y) * (A.x - C.x)

                if np.sign(ar1) == np.sign(ar2) == np.sign(ar3):
                    draw_point(image, x, y, color)

    # Przykladowe rysowanie
    # Jednokolorowa linia
    draw_line(image, 5, 5, 70, 40, color=(255, 255, 255))

    # Trojkat
    v0 = Vertex(20, 10)
    v1 = Vertex(60, 15)
    v2 = Vertex(40, 40)
    draw_triangle(image, v0, v1, v2, color=(0, 0, 139))

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Jednokolorowa linia i trojkat (z wypelnieniem)")
    plt.axis('off')
    plt.show()


def Zadanie4_4():
    print("Wykonuję Zadanie 4.4")
    width = 80
    height = 60
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Funkcja do rysowania punktu
    def draw_point(image, x, y, color=(255, 255, 255)):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[image.shape[0] - 1 - y, x] = np.clip(np.round(color), 0, 255)

    # Algorytm Bresenhama z interpolacja koloru
    def draw_line(image, x1, y1, x2, y2, col1, col2):
        col1 = np.asarray(col1, dtype=float)
        col2 = np.asarray(col2, dtype=float)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = np.sign(x2 - x1)
        sy = np.sign(y2 - y1)
        x, y = x1, y1
        steps = max(dx, dy)

        for i in range(steps + 1):
            t = i / steps if steps != 0 else 0
            color = (1 - t) * col1 + t * col2
            draw_point(image, x, y, color)

            if dx > dy:
                x += sx
                if (i * dy * 2) % dx >= dx:
                    y += sy
            else:
                y += sy
                if (i * dx * 2) % dy >= dy:
                    x += sx

    # Klasa wierzcholka z kolorem
    class Vertex:
        def __init__(self, x, y, color=(255, 255, 255)):
            self.x = x
            self.y = y
            self.color = np.asarray(color, dtype=float)

    # Funkcja do rysowania wypelnionego trojkata z interpolacja koloru
    def draw_triangle(image, A, B, C):
        xmin = min(A.x, B.x, C.x)
        xmax = max(A.x, B.x, C.x)
        ymin = min(A.y, B.y, C.y)
        ymax = max(A.y, B.y, C.y)

        # Calkowite pole trojkata
        def area(p1, p2, p3):
            return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)

        total_area = area(A, B, C)

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                P = Vertex(x, y)
                a0 = area(B, C, P)
                a1 = area(C, A, P)
                a2 = area(A, B, P)

                if np.sign(a0) == np.sign(a1) == np.sign(a2):
                    w0 = a0 / total_area
                    w1 = a1 / total_area
                    w2 = a2 / total_area
                    color = w0 * A.color + w1 * B.color + w2 * C.color
                    draw_point(image, x, y, color)

    # Przykladowe rysowanie
    # Linia z gradientem
    draw_line(image, 5, 5, 70, 40, (255, 0, 0), (0, 255, 0))

    # Trojkat z interpolacja koloru
    v0 = Vertex(20, 10, color=(255, 0, 0))
    v1 = Vertex(60, 15, color=(0, 255, 0))
    v2 = Vertex(40, 40, color=(0, 0, 255))
    draw_triangle(image, v0, v1, v2)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Interpolacja koloru: linia + trojkat")
    plt.axis('off')
    plt.show()

def Zadanie4_5():
    print("Wykonuję Zadanie 4.5")
    width = 80
    height = 60
    scale = 2  # SSAA skala
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Funkcja do rysowania punktu
    def draw_point(image, x, y, color=(255, 255, 255)):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[image.shape[0] - 1 - y, x] = np.clip(np.round(color), 0, 255)

    # Bresenham z interpolacja koloru
    def draw_line(image, x1, y1, x2, y2, col1, col2):
        col1 = np.asarray(col1, dtype=float)
        col2 = np.asarray(col2, dtype=float)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = np.sign(x2 - x1)
        sy = np.sign(y2 - y1)
        x, y = x1, y1
        steps = max(dx, dy)

        for i in range(steps + 1):
            t = i / steps if steps != 0 else 0
            color = (1 - t) * col1 + t * col2
            draw_point(image, x, y, color)

            if dx > dy:
                x += sx
                if (i * dy * 2) % dx >= dx:
                    y += sy
            else:
                y += sy
                if (i * dx * 2) % dy >= dy:
                    x += sx

    # Klasa wierzcholka
    class Vertex:
        def __init__(self, x, y, color=(255, 255, 255)):
            self.x = x
            self.y = y
            self.color = np.asarray(color, dtype=float)

    # Trojkat z interpolacja koloru
    def draw_triangle(image, A, B, C):
        xmin = int(min(A.x, B.x, C.x))
        xmax = int(max(A.x, B.x, C.x))
        ymin = int(min(A.y, B.y, C.y))
        ymax = int(max(A.y, B.y, C.y))

        def area(p1, p2, p3):
            return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)

        total_area = area(A, B, C)

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                P = Vertex(x, y)
                a0 = area(B, C, P)
                a1 = area(C, A, P)
                a2 = area(A, B, P)

                if np.sign(a0) == np.sign(a1) == np.sign(a2):
                    w0 = a0 / total_area
                    w1 = a1 / total_area
                    w2 = a2 / total_area
                    color = w0 * A.color + w1 * B.color + w2 * C.color
                    draw_point(image, x, y, color)

    # Downsampling SSAA 2x2
    def downsample(low_res, hi_res, scale):
        h, w = low_res.shape[:2]
        for y in range(h):
            for x in range(w):
                block = hi_res[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale]
                low_res[y, x] = np.clip(np.round(np.mean(block, axis=(0, 1))), 0, 255).astype(np.uint8)

    # Linia z antyaliasingiem (SSAA x2)
    def SSAA_line(image, x1, y1, x2, y2, col1, col2):
        h, w = image.shape[:2]
        hi_res = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
        draw_line(hi_res, x1 * scale, y1 * scale, x2 * scale, y2 * scale, col1, col2)
        downsample(image, hi_res, scale)

    # Trojkat z antyaliasingiem (SSAA x2)
    def SSAA_triangle(image, A, B, C):
        h, w = image.shape[:2]
        hi_res = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
        A2 = Vertex(A.x * scale, A.y * scale, A.color)
        B2 = Vertex(B.x * scale, B.y * scale, B.color)
        C2 = Vertex(C.x * scale, C.y * scale, C.color)
        draw_triangle(hi_res, A2, B2, C2)
        downsample(image, hi_res, scale)

    # Rysowanie linii z SSAA
    SSAA_line(image, 5, 5, 70, 40, (255, 0, 0), (0, 255, 0))

    # Rysowanie trojkata z interpolacja koloru i SSAA
    v0 = Vertex(20, 10, color=(255, 0, 0))
    v1 = Vertex(60, 15, color=(0, 255, 0))
    v2 = Vertex(40, 40, color=(0, 0, 255))
    SSAA_triangle(image, v0, v1, v2)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Interpolacja koloru z SSAA: linia + trojkat")
    plt.axis('off')
    plt.show()


def main():
    while True:
        print("\nWybierz zadanie do uruchomienia:")
        print("1. Zadanie4_1")
        print("2. Zadanie4_2")
        print("3. Zadanie4_3")
        print("4. Zadanie4_4")
        print("5. Zadanie4_5")
        print("0. Wyjście")

        choice = input("Podaj numer: ")

        if choice == "1":
            Zadanie4_1()
        elif choice == "2":
            Zadanie4_2()
        elif choice == "3":
            Zadanie4_3()
        elif choice == "4":
            Zadanie4_4()
        elif choice == "5":
            Zadanie4_5()
        elif choice == "0":
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawny wybór, spróbuj ponownie.")


if __name__ == "__main__":
    main()

