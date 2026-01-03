import cv2
import numpy as np
import os

# --- urutkan 4 titik supaya warp tidak terbalik ---
def titik(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

# --- klasifikasi memakai SUM OF ABSOLUTE DIFFERENCES (SAD) ---
def klasifikasi_kartu(warped, templates):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 150, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_name = "unknown"
    best_sad = float("inf")

    for nama, temp in templates.items():

        # 1. Ubah ke float
        img1 = thr.astype(np.float32)
        img2 = temp.astype(np.float32)

        # 2. SUM OF ABSOLUTE DIFFERENCES  ← BAGIAN PALING PENTING
        sad = np.sum(np.abs(img1 - img2))

        # 3. Pilih nilai terkecil
        if sad < best_sad:
            best_sad = sad
            best_name = nama

    return best_name


# --- load template (resize + threshold) ---
folder = "/Users/gungfebrian/Documents/Tugas/PCV/individual_cards_2"
w, h = 300, 420

templates = {}
ranks = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
suits = ["Spade","Heart","Diamond","Club"]

for s in suits:
    for r in ranks:
        path = os.path.join(folder, f"{r}_{s}.jpg")
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(g, 150, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        templates[f"{r} {s}"] = thr

# --- kamera ---
url = "http://192.168.111.199:4747/video"
cap = cv2.VideoCapture(0)
dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = titik(approx.reshape(4,2).astype(np.float32))
            M = cv2.getPerspectiveTransform(pts, dst)
            warp = cv2.warpPerspective(frame, M, (w, h))

            nama = klasifikasi_kartu(warp, templates)

            cv2.drawContours(frame, [approx], -1, (0,255,0), 2)
            x, y = approx[0][0]
            cv2.putText(frame, nama, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Kartu", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
