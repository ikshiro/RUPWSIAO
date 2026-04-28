import cv2
import numpy as np
import os

# ====== USTAWIENIA ======
IMAGE_PATH = "../zdjecia/kilka.jpg"
OUTPUT_DIR = "../pieces"
MIN_AREA = 2000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def resize_for_display(img, max_width=500):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img


# ====== WCZYTANIE ======
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise IOError("Nie można wczytać obrazu")

orig = img.copy()

# ====== KONWERSJA DO HSV ======
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ====== MASKA: wszystko co NIE jest białe ======
lower = np.array([0, 40, 40])
upper = np.array([179, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

# ====== MORFOLOGIA ======
kernel = np.ones((5,5), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# ====== KONTURY ======
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

count = 0

for c in contours:

    area = cv2.contourArea(c)

    if area < MIN_AREA:
        continue

    x, y, w, h = cv2.boundingRect(c)

    count += 1

    # ====== MASKA POJEDYNCZEGO PUZZLA ======
    piece_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(piece_mask, [c], -1, 255, -1)

    piece = cv2.bitwise_and(orig, orig, mask=piece_mask)
    crop = piece[y:y+h, x:x+w]

    out_path = os.path.join(OUTPUT_DIR, f"piece_{count:02d}.png")
    cv2.imwrite(out_path, crop)

    cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(orig, str(count), (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

print("Wykryto puzzli:", count)

# ====== PODGLĄD ======
cv2.imshow("Mask", resize_for_display(mask))
cv2.imshow("Detected pieces", resize_for_display(orig))
cv2.waitKey(0)
cv2.destroyAllWindows()