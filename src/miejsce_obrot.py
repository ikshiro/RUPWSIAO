import cv2
import numpy as np
import os

# =====================================
# STAŁE
# =====================================

REFERENCE_IMAGE = "../zdjecia/cala.jpg"
PUZZLE_IMAGE = "../zdjecia/kilka.jpg"

ROWS = 5
COLS = 6

EDGE_SIZE = 20
ANGLE_STEP = 10

MIN_AREA = 2000

OUTPUT_DIR = "../pieces"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# POMOCNICZE
# =====================================

def resize_for_display(img, max_width=1200):

    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    return img


# =====================================
# ROTACJA
# =====================================

def rotate_any(img, angle):

    h, w = img.shape[:2]

    center = (w//2, h//2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255,255,255)
    )

    return rotated


# =====================================
# HISTOGRAM KOLORU
# =====================================

def color_histogram(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0,1],
        None,
        [30,32],
        [0,180,0,256]
    )

    cv2.normalize(hist, hist)

    return hist


# =====================================
# WYCIĄGANIE KRAWĘDZI
# =====================================

def extract_edges(piece):

    h, w = piece.shape[:2]

    top = piece[0:EDGE_SIZE, :]
    bottom = piece[h-EDGE_SIZE:h, :]
    left = piece[:, 0:EDGE_SIZE]
    right = piece[:, w-EDGE_SIZE:w]

    return top, bottom, left, right


# =====================================
# HISTOGRAMY KRAWĘDZI
# =====================================

def edge_histograms(piece):

    edges = extract_edges(piece)

    hists = []

    for e in edges:
        hists.append(color_histogram(e))

    return hists


# =====================================
# PORÓWNANIE KRAWĘDZI
# =====================================

def edge_similarity(h1, h2):

    score = 0

    for i in range(4):

        s = cv2.compareHist(
            h1[i],
            h2[i],
            cv2.HISTCMP_BHATTACHARYYA
        )

        score += s

    return score


# =====================================
# BUDOWANIE REFERENCJI
# =====================================

def build_reference(ref_img):

    h, w = ref_img.shape[:2]

    ph = h // ROWS
    pw = w // COLS

    refs = []

    for r in range(ROWS):
        for c in range(COLS):

            piece = ref_img[r*ph:(r+1)*ph, c*pw:(c+1)*pw]

            hists = edge_histograms(piece)

            idx = r*COLS + c

            refs.append({
                "id": idx,
                "hists": hists
            })

    return refs


# =====================================
# DETEKCJA PUZZLI
# =====================================

def detect_pieces(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0,40,40])
    upper = np.array([179,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    pieces = []

    for c in contours:

        area = cv2.contourArea(c)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)

        piece_mask = np.zeros(mask.shape, np.uint8)

        cv2.drawContours(piece_mask, [c], -1, 255, -1)

        piece = cv2.bitwise_and(img, img, mask=piece_mask)

        crop = piece[y:y+h, x:x+w]

        pieces.append(crop)

    return pieces, mask


# =====================================
# SZUKANIE PUZZLA
# =====================================

def find_piece(piece, references):

    best_score = 999
    best_id = None
    best_rot = 0

    for rot in range(0, 360, ANGLE_STEP):

        rotated = rotate_any(piece, rot)

        hists = edge_histograms(rotated)

        for ref in references:

            score = edge_similarity(
                hists,
                ref["hists"]
            )

            if score < best_score:

                best_score = score
                best_id = ref["id"]
                best_rot = rot

    return best_id, best_rot


# =====================================
# PROGRAM GŁÓWNY
# =====================================

ref_img = cv2.imread(REFERENCE_IMAGE)

if ref_img is None:
    raise IOError("Nie można wczytać reference image")

references = build_reference(ref_img)

puzzle_img = cv2.imread(PUZZLE_IMAGE)

if puzzle_img is None:
    raise IOError("Nie można wczytać puzzle image")


pieces, mask = detect_pieces(puzzle_img)

print("Wykryto puzzli:", len(pieces))


for i, piece in enumerate(pieces):

    pid, rot = find_piece(piece, references)

    print(f"piece_{i:02d} -> puzzle {pid} rotation {rot}")

    cv2.imwrite(f"{OUTPUT_DIR}/piece_{i:02d}.png", piece)


cv2.imshow("Mask", resize_for_display(mask))
cv2.imshow("Puzzle image", resize_for_display(puzzle_img))

cv2.waitKey(0)
cv2.destroyAllWindows()