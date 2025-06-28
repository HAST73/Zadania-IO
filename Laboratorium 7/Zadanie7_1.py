import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# === PARAMETRY ===
image_path = 'zad8.png'    # ścieżka do pliku z obrazem
cell_size  = (16, 16)        # (wysokość, szerokość) komórki w pikselach
directions = 9             # liczba przedziałów orientacji (0–360°)
gamma      = 1.0           # korekcja gamma

# --- 1. Wczytanie i wstępna obróbka ---
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Nie znaleziono pliku '{image_path}'")
# konwersja do skali szarości i normalizacja wartości do zakresu [0,1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')/255.0
gray = np.power(gray, gamma)

# --- 2. Obliczenie gradientów Sobela, wartości i kąta 0–360° ---
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
mag   = np.hypot(gx, gy)  # moduł gradientu
angle = (np.degrees(np.arctan2(gy, gx)) + 360) % 360  # kąt w stopniach [0,360)

# --- 3. Budowa histogramów komórek ---
h, w   = gray.shape
ch, cw = cell_size
ty, tx = h // ch, w // cw
bin_w  = 360 / directions  # szerokość przedziału orientacji

cell_hists = np.zeros((ty, tx, directions), dtype=float)
for i in range(ty):
    for j in range(tx):
        m = mag[i*ch:(i+1)*ch, j*cw:(j+1)*cw].ravel()      # wartości gradientu
        a = angle[i*ch:(i+1)*ch, j*cw:(j+1)*cw].ravel()    # kąty orientacji
        hist, _ = np.histogram(a, bins=directions, range=(0,360), weights=m)
        cell_hists[i,j] = hist

total_cells = ty * tx  # liczba wszystkich komórek

# --- 4. Wczytanie n oraz sprawdzenie sąsiadów ---
while True:
    n = int(input(f"Wpisz numer komórki n (0..{total_cells-1}): "))
    if not (0 <= n < total_cells):
        print("❌ Poza zakresem")
        continue
    ok = True
    for off in (-1, 0, 1):
        idx = n + off
        if 0 <= idx < total_cells:
            yi, xi = divmod(idx, tx)
            if cell_hists[yi, xi].sum() == 0:
                print(f"❌ Komórka {idx} pusta")
                ok = False
    if ok:
        break

# obliczenie współrzędnych komórki o numerze n
yi, xi = divmod(n, tx)
y0, x0 = yi * ch, xi * cw

# --- 5. Obliczenie wizualizacji HOG na obrazie w skali szarości ---
fd, hog_image = hog(
    gray,
    orientations=directions,
    pixels_per_cell=cell_size,
    cells_per_block=(1, 1),
    visualize=True,
    feature_vector=False
)
# dopasowanie zakresu wartości do wyświetlania
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, np.max(hog_image)))

# --- 6. Rysowanie złożonej figury ---
fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.4, hspace=0.4)

# A: obraz w skali szarości z zaznaczoną komórką
af = fig.add_subplot(gs[0, 3])
af.imshow(gray, cmap='gray')
rect = patches.Rectangle((x0, y0), cw, ch, linewidth=2, edgecolor='green', facecolor='none')
af.add_patch(rect)
af.set_title(f"Obraz w skali szarości z komórką {n}")
af.axis('off')

# B: wizualizacja HOG
ax_hog = fig.add_subplot(gs[1, 3])
ax_hog.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax_hog.set_title('Wizualizacja HOG (z szarości)')
ax_hog.axis('off')

# Funkcje do wykresów radialnych i słupkowych
def draw_radial(ax, hist, idx):
    R = hist / hist.max() if hist.max() > 0 else hist
    angles_full = np.deg2rad(np.arange(0, 360, bin_w))
    for ang, r in zip(angles_full, R):
        ax.annotate('', xy=(r*np.cos(ang), r*np.sin(ang)), xytext=(0, 0),
                    arrowprops=dict(facecolor='blue', edgecolor='blue', width=2, headwidth=8))
    ax.set_aspect('equal')
    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
    # tylko główne kierunki
    major_angles = np.deg2rad([0, 90, 180, 270])
    ax.set_xticks(np.cos(major_angles))
    ax.set_yticks([])
    ax.set_title(f"komórka {idx}", fontsize=11)

def draw_bar(ax, hist, idx):
    ax.bar(np.arange(directions), hist, edgecolor='black')
    ax.set_xticks(np.arange(directions))
    ax.set_xticklabels([f"{int(i*bin_w)}°" for i in range(directions)], rotation=45, fontsize=8)
    ax.set_ylabel("suma |∇|", fontsize=9)
    ax.set_title(f"komórka {idx}", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Rysowanie dla komórek n-1, n, n+1
for col, off in enumerate((-1, 0, 1)):
    idx = n + off
    ax_r = fig.add_subplot(gs[0, col])
    if 0 <= idx < total_cells:
        yi2, xi2 = divmod(idx, tx)
        draw_radial(ax_r, cell_hists[yi2, xi2], idx)
    else:
        ax_r.text(0.5, 0.5, "poza zakres", ha='center', va='center')
        ax_r.axis('off')
    ax_b = fig.add_subplot(gs[1, col])
    if 0 <= idx < total_cells:
        yi2, xi2 = divmod(idx, tx)
        draw_bar(ax_b, cell_hists[yi2, xi2], idx)
    else:
        ax_b.axis('off')

fig.suptitle("HOG – histogramy komórek (n-1, n, n+1) oraz wizualizacja HOG z szarości", fontsize=16)
plt.show()
