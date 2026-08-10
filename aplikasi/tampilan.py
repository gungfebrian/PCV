"""
Komposisi tampilan — MURNI, tanpa kamera dan tanpa model.

Dipisah dari `penyu_live.py` supaya bisa diuji tanpa layar dan tanpa webcam:
seluruh fungsi menerima frame + state, mengembalikan kanvas BGR. Tata letaknya
diregresi oleh `uji_tampilan.py`.

Tata letak v2 — engineer tool:

    +-----------+--------------------------------+-----------------+
    | SIDEBAR   | STRIP TAHAP PIPELINE (klik)    |                 |
    | 270 px    +--------------------------------+  PANEL KANAN    |
    | kontrol   |                                |  telemetry      |
    |           |  TAMPILAN UTAMA                |  putusan        |
    |           |  (+ bbox / keypoint overlay)   |  breakdown      |
    |           |                                |                 |
    +-----------+--------------------------------+-----------------+
    |           | STRIP TOP-5 + jumlah inlier                      |
    +-----------+--------------------------------------------------+
"""

import cv2
import numpy as np

LEBAR_SIDEBAR = 270
LEBAR_PANEL = 300
TINGGI_STRIP = 168
TINGGI_TAHAP = 104

# BGR. Kontras dinaikkan dari v1 — teks abu di atas abu terbaca buruk.
BG = (22, 20, 19)
PANEL = (38, 35, 33)
PANEL2 = (52, 48, 45)
TEKS = (245, 245, 245)
REDUP = (168, 165, 162)
HIJAU = (72, 190, 60)
MERAH = (60, 70, 235)
KUNING = (40, 200, 250)
BIRU = (235, 160, 60)
UNGU = (220, 120, 190)
AKTIF = (95, 82, 70)

F = cv2.FONT_HERSHEY_SIMPLEX


# Font Hershey OpenCV hanya ASCII. Karakter di luar itu digambar sebagai "?",
# dan em-dash muncul di banyak label — hasilnya "STAGE 2 ??? RE-RANK".
# Disaring di satu tempat supaya tidak perlu diingat di tiap pemanggilan.
_GANTI = {"—": "-", "–": "-", "→": "->", "≈": "~", "±": "+/-", "×": "x",
          "≥": ">=", "≤": "<=", "“": '"', "”": '"', "’": "'", "·": "-",
          "Δ": "d", "✓": "v", "✔": "v"}


def ascii_aman(s):
    s = str(s)
    for a, b in _GANTI.items():
        s = s.replace(a, b)
    return s.encode("ascii", "replace").decode("ascii")


def teks(img, s, xy, uk=0.42, w=TEKS, tebal=1):
    cv2.putText(img, ascii_aman(s), xy, F, uk, w, tebal, cv2.LINE_AA)


def judul(img, s, xy, uk=0.46, w=TEKS):
    """'Bold' di OpenCV = tebal 2. Dipakai konsisten untuk semua judul."""
    cv2.putText(img, ascii_aman(s), xy, F, uk, w, 2, cv2.LINE_AA)


def kotak(img, x, y, w, h, warna, isi=True):
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), warna,
                  -1 if isi else 1)


def muat_pas(im, kw, kh, bg=PANEL):
    """Skala menjaga rasio aspek, letakkan di tengah kanvas kw x kh.
    Mengembalikan (kanvas, skala, offset_x, offset_y) supaya overlay seperti
    bbox dan keypoint bisa dipetakan ke koordinat yang benar."""
    kanvas = np.full((kh, kw, 3), bg, np.uint8)
    if im is None or im.size == 0:
        return kanvas, 1.0, 0, 0
    h, w = im.shape[:2]
    s = min(kw / w, kh / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    kecil = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    if kecil.ndim == 2:
        kecil = cv2.cvtColor(kecil, cv2.COLOR_GRAY2BGR)
    y, x = (kh - nh) // 2, (kw - nw) // 2
    kanvas[y:y + nh, x:x + nw] = kecil
    return kanvas, s, x, y


# --------------------------------------------------------------- sidebar
def bangun_baris_sidebar(state):
    """Satu sumber kebenaran untuk yang digambar DAN yang bisa diklik.
    Kalau dipisah, area klik dan area terlihat pasti bergeser cepat atau
    lambat, dan tombol akan menekan hal yang salah tanpa error."""
    b = [("judul", "PREPROCESSING", None, False)]
    for k, label in state["kondisi_pilihan"]:
        b.append(("opsi", label, ("kondisi", k), state["kondisi"] == k))

    b.append(("judul", "STAGE 2 — MATCHER", None, False))
    for k, label in state.get("matcher_pilihan", []):
        b.append(("opsi", label, ("matcher", k), state.get("matcher") == k))

    b.append(("judul", "STAGE 2 — MODE", None, False))
    for k, label in state["rerank_pilihan"]:
        b.append(("opsi", label, ("rerank", k), state["rerank"] == k))

    b.append(("judul", "SISI QUERY", None, False))
    for k in ("left", "right"):
        b.append(("opsi", k, ("sisi", k), state["sisi"] == k))

    b.append(("judul", "OVERLAY", None, False))
    b.append(("opsi", "bounding box", ("toggle", "bbox"), state["bbox"]))
    b.append(("opsi", "keypoint SIFT", ("toggle", "keypoint"), state["keypoint"]))
    b.append(("opsi", "garis inlier", ("toggle", "match"), state["match"]))

    b.append(("judul", f"AMBANG  {state['ambang']:.2f}", None, False))
    b.append(("opsi", "[-] turun    [+] naik", ("ambang", None), False))

    b.append(("judul", "AKSI", None, False))
    b.append(("opsi", "[SPASI] jeda / lanjut", ("aksi", "jeda"), state["jeda"]))
    b.append(("opsi", "[S] simpan tangkapan", ("aksi", "simpan"), False))
    b.append(("opsi", "[Q] keluar", ("aksi", "keluar"), False))
    return b


def geometri_sidebar(baris, tinggi=None):
    y = 52
    keluar = []
    for i, (jenis, _, kunci, _) in enumerate(baris):
        if jenis == "judul":
            y += 28
            continue
        if kunci:
            keluar.append((i, y, y + 21))
        y += 23
    return keluar


def gambar_sidebar(kanvas, state):
    h = kanvas.shape[0]
    kotak(kanvas, 0, 0, LEBAR_SIDEBAR, h, PANEL)
    judul(kanvas, "RE-ID PENYU", (14, 26), 0.6)
    teks(kanvas, "engineer tool", (14, 42), 0.36, REDUP)

    baris = bangun_baris_sidebar(state)
    geo = {i: (a, b) for i, a, b in geometri_sidebar(baris)}
    y = 52
    for i, (jenis, label, kunci, aktif) in enumerate(baris):
        if jenis == "judul":
            cv2.line(kanvas, (12, y + 3), (LEBAR_SIDEBAR - 12, y + 3),
                     (70, 65, 60), 1)
            judul(kanvas, label, (14, y + 20), 0.36, KUNING)
            y += 28
            continue
        y0 = geo.get(i, (y, y + 21))[0]
        if aktif:
            kotak(kanvas, 8, y0, LEBAR_SIDEBAR - 16, 21, AKTIF)
            kotak(kanvas, 8, y0, 3, 21, KUNING)
        teks(kanvas, label, (18, y0 + 15), 0.41,
             TEKS if aktif else REDUP, 2 if aktif else 1)
        y += 23
    return kanvas


def klik_sidebar(state, x, y, tinggi=None):
    if x >= LEBAR_SIDEBAR:
        return None
    baris = bangun_baris_sidebar(state)
    for i, y0, y1 in geometri_sidebar(baris):
        if y0 <= y <= y1:
            return baris[i][2]
    return None


# ---------------------------------------------------- strip tahap pipeline
def gambar_tahap(kanvas, tahap, aktif, x0, y0, lebar):
    """DETECT -> ALIGN -> DESCRIBE -> MATCH, ditampilkan sebagai thumbnail.

    Kerangka yang sama dipakai MegaDescriptor dan sistem re-ID mana pun;
    menampilkannya berdampingan membuat jelas di tahap mana sesuatu rusak.
    """
    kotak(kanvas, x0, y0, lebar, TINGGI_TAHAP, PANEL)
    judul(kanvas, "TAHAP PIPELINE", (x0 + 10, y0 + 16), 0.36, KUNING)
    kw, kh = 96, TINGGI_TAHAP - 44
    for i, (nama, im) in enumerate(tahap):
        x = x0 + 10 + i * (kw + 8)
        if x + kw > x0 + lebar:
            break
        y = y0 + 22
        th, _, _, _ = muat_pas(im, kw, kh, PANEL2)
        kanvas[y:y + kh, x:x + kw] = th
        pilih = nama == aktif
        cv2.rectangle(kanvas, (x, y), (x + kw, y + kh),
                      KUNING if pilih else (75, 70, 66), 2 if pilih else 1)
        teks(kanvas, nama, (x, y + kh + 14), 0.35,
             TEKS if pilih else REDUP, 2 if pilih else 1)
    return kanvas


def geometri_tahap(tahap, x0, y0, lebar):
    kw, kh = 96, TINGGI_TAHAP - 44
    out = []
    for i, (nama, _) in enumerate(tahap):
        x = x0 + 10 + i * (kw + 8)
        if x + kw > x0 + lebar:
            break
        out.append((nama, x, y0 + 22, x + kw, y0 + 22 + kh + 16))
    return out


def klik_tahap(tahap, x, y, x0, y0, lebar):
    for nama, a, b, c, d in geometri_tahap(tahap, x0, y0, lebar):
        if a <= x <= c and b <= y <= d:
            return ("tahap", nama)
    return None


# ------------------------------------------------------------- overlay
def gambar_bbox(kanvas, bbox, skala, ox, oy, ragu=True):
    """Kotak deteksi. SENGAJA diberi label 'heuristik'.

    Deteksi kontur terbesar sudah terdokumentasi sering meleset ke riak pasir,
    bukan ke penyunya — dan itu sebabnya crop ROI dimatikan di pipeline lama.
    Kotaknya tetap digambar supaya kelihatan seberapa sering meleset, bukan
    karena dipercaya.
    """
    if bbox is None:
        return kanvas
    x, y, w, h = bbox
    p1 = (int(x * skala + ox), int(y * skala + oy))
    p2 = (int((x + w) * skala + ox), int((y + h) * skala + oy))
    warna = BIRU if ragu else HIJAU
    cv2.rectangle(kanvas, p1, p2, warna, 2)
    lab = "DETECT heuristik (tidak dipakai untuk identifikasi)"
    (tw, th), _ = cv2.getTextSize(lab, F, 0.34, 1)
    kotak(kanvas, p1[0], max(0, p1[1] - th - 8), tw + 10, th + 8, warna)
    teks(kanvas, lab, (p1[0] + 5, max(10, p1[1] - 5)), 0.34, (20, 20, 20), 1)
    return kanvas


def gambar_keypoint(kanvas, pts, skala, ox, oy, warna=KUNING, r=2):
    """Titik kuning — bukti bahwa model benar-benar menangkap tekstur, dan
    sekaligus bukti DI MANA ia menangkapnya. Kalau titiknya menumpuk di karang
    dan pasir, itu penjelasan langsung kenapa pasangan salah bisa menang."""
    if pts is None:
        return kanvas
    for px, py in pts:
        cv2.circle(kanvas, (int(px * skala + ox), int(py * skala + oy)), r,
                   warna, -1, cv2.LINE_AA)
    return kanvas


def panel_match(qim, gim, pasangan, lebar, tinggi):
    """Query | kandidat berdampingan, garis inlier di antaranya.

    Ini panel paling informatif di seluruh alat: kalau garis mendarat di
    karang alih-alih di sisik, penyebab kegagalannya terlihat langsung tanpa
    perlu membaca angka apa pun.
    """
    kanvas = np.full((tinggi, lebar, 3), PANEL, np.uint8)
    sep = lebar // 2
    a, sa, oxa, oya = muat_pas(qim, sep - 2, tinggi, PANEL2)
    b, sb, oxb, oyb = muat_pas(gim, lebar - sep - 2, tinggi, PANEL2)
    kanvas[:, :sep - 2] = a
    kanvas[:, sep + 2:] = b
    for (x1, y1), (x2, y2) in (pasangan or []):
        p1 = (int(x1 * sa + oxa), int(y1 * sa + oya))
        p2 = (int(x2 * sb + oxb) + sep + 2, int(y2 * sb + oyb))
        cv2.line(kanvas, p1, p2, HIJAU, 1, cv2.LINE_AA)
        cv2.circle(kanvas, p1, 2, KUNING, -1)
        cv2.circle(kanvas, p2, 2, KUNING, -1)
    cv2.line(kanvas, (sep, 0), (sep, tinggi), (80, 75, 70), 2)
    teks(kanvas, "QUERY", (8, 16), 0.36, REDUP)
    teks(kanvas, "KANDIDAT #1", (sep + 10, 16), 0.36, REDUP)
    return kanvas


# ---------------------------------------------------------- panel kanan
def gambar_panel(kanvas, x0, tinggi, t, hasil, ambang, ringkas):
    kotak(kanvas, x0, 0, LEBAR_PANEL, tinggi, PANEL)
    y = 26
    judul(kanvas, "PUTUSAN", (x0 + 12, y), 0.4, KUNING)
    y += 12

    if hasil is None:
        teks(kanvas, "belum ada", (x0 + 12, y + 20), 0.42, REDUP)
        y += 40
    else:
        kenal = hasil["skor"] >= ambang
        warna = HIJAU if kenal else MERAH
        kotak(kanvas, x0 + 12, y, LEBAR_PANEL - 24, 66, (26, 24, 23))
        cv2.rectangle(kanvas, (x0 + 12, y),
                      (x0 + LEBAR_PANEL - 12, y + 66), warna, 2)
        judul(kanvas, hasil["nama"] if kenal else "TIDAK DIKENAL",
              (x0 + 22, y + 28), 0.56, TEKS)
        teks(kanvas, f"cos {hasil['skor']:.4f}", (x0 + 22, y + 48), 0.38, REDUP)
        teks(kanvas, f"margin ke-2 {hasil['margin']:+.4f}",
             (x0 + 22, y + 60), 0.38,
             REDUP if abs(hasil["margin"]) > 0.02 else MERAH)
        y += 78
        if abs(hasil["margin"]) <= 0.02:
            teks(kanvas, "margin tipis — ini tebakan", (x0 + 14, y), 0.34, MERAH)
            y += 14

    y += 8
    judul(kanvas, "TELEMETRY", (x0 + 12, y), 0.4, KUNING)
    y += 10
    isi = [
        ("infer stage-1", f"{t['ms_infer']:.1f} ms"),
        ("praproses", f"{t['ms_pra']:.1f} ms"),
        ("stage-2 match", f"{t['ms_match']:.1f} ms" if t.get("ms_match") else "-"),
        ("fps", f"{t['fps']:.1f}"),
        ("input", f"{t['input_w']}x{t['input_h']}"),
        ("-> model", t["ukuran"]),
        ("keypoint", t.get("n_kp", "-")),
        ("inlier top-1", t.get("inlier", "-")),
        ("gallery", f"{t['n_gallery']} foto"),
    ]
    for k, v in isi:
        y += 15
        teks(kanvas, k, (x0 + 14, y), 0.36, REDUP)
        teks(kanvas, v, (x0 + 150, y), 0.36, TEKS, 1)

    y += 26
    judul(kanvas, "AKURASI TERUKUR", (x0 + 12, y), 0.4, KUNING)
    y += 8
    if not ringkas:
        y += 16
        teks(kanvas, "statistik.json belum ada", (x0 + 14, y), 0.34, REDUP)
        y += 13
        teks(kanvas, "angka usang lebih buruk", (x0 + 14, y), 0.34, REDUP)
        y += 12
        teks(kanvas, "daripada tidak ada angka", (x0 + 14, y), 0.34, REDUP)
    else:
        for k, v in ringkas:
            y += 15
            teks(kanvas, k, (x0 + 14, y), 0.36, REDUP)
            teks(kanvas, v, (x0 + 150, y), 0.36, TEKS)
    return kanvas


# ------------------------------------------------------- strip top-5
def gambar_strip(kanvas, y0, top5, ambang, mode_rerank):
    h_total = kanvas.shape[0] - y0
    lebar = kanvas.shape[1] - LEBAR_SIDEBAR
    kotak(kanvas, LEBAR_SIDEBAR, y0, lebar, h_total, PANEL)
    ket = {"off": "urutan stage-1 (cosine)",
           "murni": "diurutkan ulang: skor inlier",
           "rrf": "diurutkan ulang: RRF cosine + inlier"}[mode_rerank]
    judul(kanvas, "TOP-5 KANDIDAT", (LEBAR_SIDEBAR + 12, y0 + 18), 0.4, KUNING)
    teks(kanvas, ket, (LEBAR_SIDEBAR + 170, y0 + 18), 0.36, REDUP)

    if not top5:
        teks(kanvas, "galeri belum dimuat", (LEBAR_SIDEBAR + 12, y0 + 44),
             0.42, REDUP)
        return kanvas

    kw, kh = 104, h_total - 26 - 46
    for i, k in enumerate(top5[:5]):
        x = LEBAR_SIDEBAR + 12 + i * (kw + 12)
        y = y0 + 26
        th, _, _, _ = muat_pas(k.get("img"), kw, kh, PANEL2)
        kanvas[y:y + kh, x:x + kw] = th
        w = HIJAU if k["skor"] >= ambang else (78, 74, 70)
        cv2.rectangle(kanvas, (x, y), (x + kw, y + kh), w, 2)
        kotak(kanvas, x, y, 20, 16, w)
        teks(kanvas, f"{i + 1}", (x + 6, y + 12), 0.38, (20, 20, 20), 2)
        teks(kanvas, k["nama"][:15], (x, y + kh + 13), 0.36, TEKS, 1)
        teks(kanvas, f"cos {k['skor']:.3f}", (x, y + kh + 25), 0.34, REDUP)
        if k.get("inlier") is not None:
            teks(kanvas, f"inlier {int(k['inlier'])}", (x, y + kh + 37), 0.34,
                 KUNING if k["inlier"] > 0 else REDUP)
    return kanvas


# ------------------------------------------------------------- kanvas
def susun(frame, state, telemetry, hasil, top5, lebar=1500, tinggi=880,
          tahap=None, bbox=None, keypoint=None, pasangan=None,
          gambar_kandidat=None, ringkas=None):
    """Fungsi murni: semua masukan -> kanvas siap tampil."""
    kanvas = np.full((tinggi, lebar, 3), BG, np.uint8)
    gambar_sidebar(kanvas, state)

    x_panel = lebar - LEBAR_PANEL
    gambar_panel(kanvas, x_panel, tinggi, telemetry, hasil,
                 state["ambang"], ringkas)

    x0 = LEBAR_SIDEBAR
    lebar_tengah = x_panel - LEBAR_SIDEBAR
    tahap = tahap or []
    gambar_tahap(kanvas, tahap, state.get("tahap", "Asli"), x0, 0, lebar_tengah)

    y_strip = tinggi - TINGGI_STRIP
    y_utama = TINGGI_TAHAP
    h_utama = y_strip - y_utama

    if state["match"] and pasangan is not None:
        kanvas[y_utama:y_strip, x0:x_panel] = panel_match(
            frame, gambar_kandidat, pasangan, lebar_tengah, h_utama)
    else:
        vis, s, ox, oy = muat_pas(frame, lebar_tengah, h_utama, BG)
        if state["bbox"]:
            gambar_bbox(vis, bbox, s, ox, oy)
        if state["keypoint"]:
            gambar_keypoint(vis, keypoint, s, ox, oy)
        kanvas[y_utama:y_strip, x0:x_panel] = vis

    gambar_strip(kanvas, y_strip, top5, state["ambang"], state["rerank"])
    return kanvas
