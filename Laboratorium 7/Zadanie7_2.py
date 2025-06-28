import cv2
import numpy as np

# Wczytaj klasyfikator dla wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Wczytaj obraz nakładki (może zawierać kanał alfa)
overlay = cv2.imread('zad8_a.png', cv2.IMREAD_UNCHANGED)
if overlay is None:
    print("Nie udało się wczytać obrazu nakładki 'zad8_a.png'. Upewnij się, że plik znajduje się w tym samym folderze.")
    exit(1)

# Uruchom kamerę internetową
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Nie można otworzyć kamery.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Dopasuj rozmiar nakładki do wykrytej twarzy
        overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

        # Określ obszar w ramce
        y1, y2 = y, y + h
        x1, x2 = x, x + w

        # Jeśli nakładka ma kanał alfa, wykonaj mieszanie
        if overlay_resized.shape[2] == 4:
            alpha_s = overlay_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha_s * overlay_resized[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
        else:
            # Brak kanału alfa -> prosta podmiana
            frame[y1:y2, x1:x2] = overlay_resized

    cv2.imshow('Face Swap', frame)
    # Naciśnij 'q', aby zakończyć
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
