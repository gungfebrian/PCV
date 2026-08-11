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
    b = [("judul", "DATASET", None, False)]
    for k, label in state.get("dataset_pilihan", []):
        b.append(("opsi2", label, ("dataset", k), state.get("dataset") == k))

    b.append(("judul", "SUMBER", None, False))
    for k, label in state.get("sumber_pilihan", []):
        b.append(("opsi2", label, ("sumber", k), state.get("sumber") == k))
    if state.get("sumber") == "dataset":
        b.append(("opsi", f"[<] [>] {state.get('info_query', '')}",
                  ("nav", "next"), False))

    # Dua kolom untuk daftar yang panjang. Dengan 11 kondisi preprocessing,
    # satu baris per kondisi membuat sidebar jauh melebihi layar sehingga
    # SEMUANYA harus di-scroll. Dua kolom memangkasnya jadi 6 baris dan
    # membuat scroll nyaris tidak diperlukan lagi.
    # Status crop kepala ditampilkan SEBAGAI BARIS SENDIRI, bukan sekadar
    # salah satu opsi preprocessing. Di Zakynthos crop kepala menaikkan
    # Rank-1 dari 12,50% ke 63,75% — selisih terbesar dari seluruh eksperimen.
    # Kalau mode ini mati tanpa disadari, angkanya runtuh tanpa pesan apa pun.
    # STAGE 1 dan STAGE 2 dipilih TERPISAH. Eksperimen membuktikan crop
    # kepala menolong keduanya sendiri-sendiri: di Zakynthos stage-1 saja
    # naik 8,75% -> 63,75%. Kalau keduanya dipaksa sama, konfigurasi
    # terbaik tidak bisa dicoba dari UI.
    # STAGE 0 punya seksi SENDIRI karena crop kepala terbukti menolong
    # KEDUA stage, bukan cuma stage-2: di Zakynthos stage-1 saja naik
    # 8,75% -> 63,75%. Menaruhnya di dalam STAGE 2 menyesatkan.
    kep = (state["kondisi"] in ("kepala", "kepala_gt")
           or state.get("stage1") in ("kepala", "kepala_gt"))
    b.append(("judul", "STAGE 0 - DETEKSI KEPALA", None, False))
    b.append(("opsi", ("[v] AKTIF - crop kepala" if kep
                       else "[ ] MATI - frame penuh"
                            + (f" ({state['catatan_kepala']})"
                               if state.get("catatan_kepala") else "")),
              ("toggle_kepala", None), kep))

    b.append(("judul", "STAGE 1 - CARI KANDIDAT", None, False))
    for k, label in state.get("stage1_pilihan", []):
        b.append(("opsi2", label, ("stage1", k), state.get("stage1") == k))

    b.append(("judul", "STAGE 2 - PERIKSA TELITI", None, False))
    for k, label in state["kondisi_pilihan"]:
        b.append(("opsi2", label, ("kondisi", k), state["kondisi"] == k))

    b.append(("judul", "STAGE 2 — MATCHER & MODE", None, False))
    for k, label in state.get("matcher_pilihan", []):
        b.append(("opsi2", label, ("matcher", k), state.get("matcher") == k))
    for k, label in state["rerank_pilihan"]:
        b.append(("opsi2", label, ("rerank", k), state["rerank"] == k))

    b.append(("judul", "SISI QUERY & OVERLAY", None, False))
    for k in ("left", "right"):
        b.append(("opsi2", k, ("sisi", k), state["sisi"] == k))
    b.append(("opsi2", "bbox", ("toggle", "bbox"), state["bbox"]))
    b.append(("opsi2", "keypoint", ("toggle", "keypoint"), state["keypoint"]))
    b.append(("opsi2", "garis inlier", ("toggle", "match"), state["match"]))
    b.append(("opsi2", f"ambang {state['ambang']:.2f}", ("ambang", None), False))

    b.append(("judul", "AKSI", None, False))
    b.append(("opsi2", "[SPASI] jeda", ("aksi", "jeda"), state["jeda"]))
    b.append(("opsi2", "[S] simpan", ("aksi", "simpan"), False))
    b.append(("opsi", "[Q] keluar", ("aksi", "keluar"), False))
    return b


# Geometri kolom. Satu tempat, dipakai menggambar DAN hit-test.
_X0, _X1 = 8, LEBAR_SIDEBAR - 8
_XTENGAH = (_X0 + _X1) // 2
KOLOM = {"opsi": (_X0, _X1),
         "opsi2a": (_X0, _XTENGAH - 2),
         "opsi2b": (_XTENGAH + 2, _X1)}
TINGGI_BARIS = 23
TINGGI_JUDUL = 28
Y_AWAL = 52
TINGGI_KEPALA = 44


def _tata_letak(baris, geser=0):
    """(indeks, x0, y0, x1, y1) tiap baris yang bisa diklik, termasuk geseran.

    SATU fungsi untuk menggambar DAN hit-test. Kalau keduanya menghitung
    sendiri-sendiri, tombol akan menekan hal yang salah begitu sidebar
    di-scroll atau tata letaknya berubah — dan tidak ada error yang muncul.

    Baris "opsi2" mengisi setengah lebar dan berpasangan: yang pertama di
    kolom kiri, berikutnya di kolom kanan, baru turun satu baris.
    """
    y = Y_AWAL - geser
    keluar = []
    kolom_kanan = False
    for i, (jenis, _, kunci, _) in enumerate(baris):
        if jenis == "judul":
            if kolom_kanan:                 # tutup pasangan yang menggantung
                y += TINGGI_BARIS
                kolom_kanan = False
            y += TINGGI_JUDUL
            continue
        if jenis == "opsi2":
            x0, x1 = KOLOM["opsi2b" if kolom_kanan else "opsi2a"]
            if kunci:
                keluar.append((i, x0, y, x1, y + 21))
            if kolom_kanan:
                y += TINGGI_BARIS
            kolom_kanan = not kolom_kanan
            continue
        if kolom_kanan:                     # "opsi" lebar penuh memaksa turun
            y += TINGGI_BARIS
            kolom_kanan = False
        x0, x1 = KOLOM["opsi"]
        if kunci:
            keluar.append((i, x0, y, x1, y + 21))
        y += TINGGI_BARIS
    return keluar, y + (TINGGI_BARIS if kolom_kanan else 0)


def tinggi_sidebar(baris):
    """Tinggi total isi sidebar, untuk membatasi scroll."""
    return _tata_letak(baris, 0)[1] + 12


def geometri_sidebar(baris, tinggi=None, geser=0):
    return _tata_letak(baris, geser)[0]


def batas_geser(baris, tinggi_kanvas):
    return max(0, tinggi_sidebar(baris) - tinggi_kanvas + 8)


def tombol_geser(tinggi):
    """Dua tombol panah yang MENEMPEL di layar, tidak ikut tergeser.

    Ini ada karena roda mouse tidak bisa diandalkan: backend Cocoa OpenCV di
    macOS tidak pernah mengirim EVENT_MOUSEWHEEL sama sekali, jadi sidebar
    terasa "tidak bisa di-scroll" walau logikanya benar. Tombol yang bisa
    diklik selalu bekerja di semua backend.
    """
    s = 22
    x = LEBAR_SIDEBAR - s - 6
    return {"naik": (x, TINGGI_KEPALA - s - 4, s, s),
            "turun": (x, tinggi - s - 6, s, s)}


def potong_label(label, lebar):
    """Potong label agar muat di kolom selebar `lebar` px.

    Hershey SIMPLEX pada skala 0,39 kira-kira 6,4 px per karakter. Dipisah
    jadi fungsi sendiri supaya tes bisa memeriksa hasil AKHIR yang terbaca
    pengguna, bukan label sebelum dipotong: "Resize seragam 512x512" dan
    "Resize seragam 368x368" sama-sama menjadi "Resize seragam ." di kolom
    sempit — tombolnya tetap berfungsi, tapi tidak ada cara membedakannya.
    """
    maks = max(4, int((lebar - 18) / 6.4))
    return label if len(label) <= maks else label[:maks - 1] + "."


def _judul_y(baris, geser):
    """Posisi tiap judul, dihitung dengan aturan tata letak yang sama."""
    y = Y_AWAL - geser
    keluar = {}
    kolom_kanan = False
    for i, (jenis, _, _, _) in enumerate(baris):
        if jenis == "judul":
            if kolom_kanan:
                y += TINGGI_BARIS
                kolom_kanan = False
            keluar[i] = y
            y += TINGGI_JUDUL
        elif jenis == "opsi2":
            if kolom_kanan:
                y += TINGGI_BARIS
            kolom_kanan = not kolom_kanan
        else:
            if kolom_kanan:
                y += TINGGI_BARIS
                kolom_kanan = False
            y += TINGGI_BARIS
    return keluar


def gambar_sidebar(kanvas, state):
    h = kanvas.shape[0]
    kotak(kanvas, 0, 0, LEBAR_SIDEBAR, h, PANEL)

    baris = bangun_baris_sidebar(state)
    maks = batas_geser(baris, h)
    geser = int(np.clip(state.get("geser", 0), 0, maks))
    state["geser"] = geser
    geo = {i: (x0, y0, x1) for i, x0, y0, x1, _ in
           geometri_sidebar(baris, h, geser)}
    y_judul = _judul_y(baris, geser)

    for i, (jenis, label, kunci, aktif) in enumerate(baris):
        if jenis == "judul":
            y = y_judul[i]
            if TINGGI_KEPALA <= y <= h - 10:
                cv2.line(kanvas, (12, y + 3), (LEBAR_SIDEBAR - 12, y + 3),
                         (70, 65, 60), 1)
                judul(kanvas, label, (14, y + 20), 0.36, KUNING)
            continue
        if i not in geo:                          # baris info tanpa aksi
            continue
        x0, y0, x1 = geo[i]
        if y0 < TINGGI_KEPALA or y0 > h - 26:     # di luar layar
            continue
        lebar = x1 - x0
        if aktif:
            kotak(kanvas, x0, y0, lebar, 21, AKTIF)
            kotak(kanvas, x0, y0, 3, 21, KUNING)
        teks(kanvas, potong_label(label, lebar), (x0 + 9, y0 + 15), 0.39,
             TEKS if aktif else REDUP, 2 if aktif else 1)

    # kepala menimpa isi yang tergeser, supaya judul selalu terbaca
    kotak(kanvas, 0, 0, LEBAR_SIDEBAR, TINGGI_KEPALA, PANEL)
    judul(kanvas, "RE-ID PENYU", (14, 26), 0.6)
    teks(kanvas, "engineer tool", (14, 40), 0.34, REDUP)

    if maks > 0:
        tb = tombol_geser(h)
        for nama, (bx, by, bw, bh) in tb.items():
            bisa = geser > 0 if nama == "naik" else geser < maks
            kotak(kanvas, bx, by, bw, bh, PANEL2 if bisa else PANEL)
            kotak(kanvas, bx, by, bw, bh, (80, 76, 70), isi=False)
            cx, cy = bx + bw // 2, by + bh // 2
            arah = -1 if nama == "naik" else 1
            titik = np.array([[cx - 5, cy + 3 * arah], [cx + 5, cy + 3 * arah],
                              [cx, cy - 4 * arah]], np.int32)
            cv2.fillConvexPoly(kanvas, titik, KUNING if bisa else (90, 86, 80))
        # batang posisi
        tinggi_bar = max(30, int(h * h / (h + maks)))
        y_bar = TINGGI_KEPALA + int((h - TINGGI_KEPALA - tinggi_bar)
                                    * geser / maks)
        kotak(kanvas, LEBAR_SIDEBAR - 4, TINGGI_KEPALA, 2,
              h - TINGGI_KEPALA, (60, 56, 52))
        kotak(kanvas, LEBAR_SIDEBAR - 4, y_bar, 2, tinggi_bar, KUNING)
        teks(kanvas, "geser: klik panah / n p", (12, h - 8), 0.32, REDUP)
    return kanvas


def klik_sidebar(state, x, y, tinggi=None):
    """Kembalikan aksi baris yang diklik, atau ('geser', +/-1) untuk panah.

    Hit-test memakai x DAN y. Sejak sidebar punya dua kolom, memeriksa y saja
    akan mengembalikan tombol kolom kiri untuk klik di kolom kanan.
    """
    if x >= LEBAR_SIDEBAR:
        return None
    baris = bangun_baris_sidebar(state)
    if tinggi:
        maks = batas_geser(baris, tinggi)
        if maks > 0:
            for nama, (bx, by, bw, bh) in tombol_geser(tinggi).items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    return ("geser", -1 if nama == "naik" else 1)
    if y < TINGGI_KEPALA:                 # area judul, bukan tombol
        return None
    for i, x0, y0, x1, y1 in geometri_sidebar(baris, tinggi,
                                              state.get("geser", 0)):
        if x0 <= x <= x1 and y0 <= y <= y1:
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

    PENTING: `qim` dan `gim` HARUS gambar yang benar-benar dilihat matcher,
    yaitu SETELAH preprocessing. Koordinat di `pasangan` hidup di ruang itu.
    Sebelumnya panel ini diberi frame asli, sehingga garisnya diskalakan
    dengan faktor gambar asli padahal titiknya dalam ruang 512x512 — garis
    berakhir jauh di luar panel dan tidak ada error apa pun yang muncul.
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
        galat = t.get("galat")
        if galat:
            # Pesan galat harus menyebut PERINTAH yang memperbaikinya, bukan
            # cuma "belum ada" — itu yang membedakan alat yang bisa dipakai
            # sendiri dari alat yang selalu perlu ditanyakan.
            kotak(kanvas, x0 + 12, y, LEBAR_PANEL - 24, 3, MERAH)
            y += 14
            kata, baris_ = galat.split(), ""
            for w in kata:
                if len(baris_) + len(w) > 34:
                    teks(kanvas, baris_, (x0 + 14, y), 0.33, MERAH)
                    y += 12
                    baris_ = ""
                baris_ += w + " "
            teks(kanvas, baris_, (x0 + 14, y), 0.33, MERAH)
            y += 22
        else:
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

    y += 24
    terbaik = t.get("terbaik")
    if terbaik:
        judul(kanvas, "OPSI TERBAIK", (x0 + 12, y), 0.4, HIJAU)
        y += 10
        kotak(kanvas, x0 + 12, y, LEBAR_PANEL - 24, 62, (24, 34, 24))
        cv2.rectangle(kanvas, (x0 + 12, y),
                      (x0 + LEBAR_PANEL - 12, y + 62), HIJAU, 1)
        judul(kanvas, f"Rank-1 {terbaik['rank1']:.2f}%", (x0 + 20, y + 20),
              0.5, HIJAU)
        # `.get` bukan `[...]`: berkas hasil lama tidak selalu punya rank5/mAP,
        # dan panel yang melempar KeyError akan mematikan seluruh aplikasi
        # hanya karena satu angka pelengkap tidak ada.
        def _a(k):
            v = terbaik.get(k)
            return f"{v:.2f}%" if isinstance(v, (int, float)) else "-"
        teks(kanvas, f"Rank-5 {_a('rank5')}  mAP {_a('mAP')}",
             (x0 + 20, y + 36), 0.36, TEKS)
        teks(kanvas, terbaik["label"][:38], (x0 + 20, y + 52), 0.33, REDUP)
        y += 74
        if terbaik.get("aktif"):
            teks(kanvas, "^ konfigurasi ini sedang dipakai", (x0 + 14, y),
                 0.33, HIJAU)
        else:
            teks(kanvas, "^ BUKAN yang sedang dipakai", (x0 + 14, y), 0.33, BIRU)
        y += 18

    y += 10
    papan = t.get("papan") or []
    if papan:
        judul(kanvas, "PAPAN SKOR KONDISI", (x0 + 12, y), 0.4, KUNING)
        y += 6
        teks(kanvas, f"Rank-1, {t.get('papan_ket', '')}", (x0 + 12, y + 10),
             0.31, REDUP)
        y += 20
        # Batang relatif terhadap yang terbaik supaya urutannya langsung
        # terlihat tanpa harus membandingkan angka satu per satu.
        atas = max(p["rank1"] for p in papan)
        bawah = min(p["rank1"] for p in papan)
        for i, p in enumerate(papan[:9]):
            terbaik = p["rank1"] >= atas - 1e-9
            terburuk = p["rank1"] <= bawah + 1e-9
            w = KUNING if p.get("aktif") else (
                HIJAU if terbaik else (MERAH if terburuk else TEKS))
            lebar = int((LEBAR_PANEL - 130) * p["rank1"] / max(atas, 1e-9))
            kotak(kanvas, x0 + 96, y - 8, max(lebar, 2), 10,
                  (60, 90, 60) if terbaik else PANEL2)
            teks(kanvas, potong_label(p["label"], 96), (x0 + 12, y), 0.33, w,
                 2 if (terbaik or p.get("aktif")) else 1)
            teks(kanvas, f"{p['rank1']:.2f}%", (x0 + LEBAR_PANEL - 58, y),
                 0.35, w, 2 if terbaik else 1)
            if terbaik:
                teks(kanvas, "TERBAIK", (x0 + 96, y + 10), 0.28, HIJAU, 1)
                y += 10
            elif terburuk and len(papan) > 1:
                teks(kanvas, "TERBURUK", (x0 + 96, y + 10), 0.28, MERAH, 1)
                y += 10
            y += 17
        if t.get("papan_catatan"):
            y += 2
            for baris in t["papan_catatan"]:
                teks(kanvas, baris, (x0 + 12, y), 0.29, REDUP)
                y += 11
        return kanvas

    judul(kanvas, "AKURASI TERUKUR", (x0 + 12, y), 0.4, KUNING)
    y += 8
    if not ringkas:
        y += 16
        for baris in ("belum ada hasil terukur", "untuk dataset ini.",
                      "Jalankan rerank.py dulu -", "angka usang lebih buruk",
                      "daripada tidak ada angka."):
            teks(kanvas, baris, (x0 + 14, y), 0.33, REDUP)
            y += 13
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
          gambar_kandidat=None, ringkas=None, gambar_query=None):
    """Fungsi murni: semua masukan -> kanvas siap tampil.

    `gambar_query` adalah versi query SETELAH preprocessing, dipakai khusus
    untuk panel match. Terpisah dari `frame` karena koordinat `pasangan`
    hidup di ruang gambar praproses, bukan gambar asli.
    """
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
            gambar_query if gambar_query is not None else frame,
            gambar_kandidat, pasangan, lebar_tengah, h_utama)
    else:
        vis, s, ox, oy = muat_pas(frame, lebar_tengah, h_utama, BG)
        if state["bbox"]:
            gambar_bbox(vis, bbox, s, ox, oy)
        if state["keypoint"]:
            gambar_keypoint(vis, keypoint, s, ox, oy)
        kanvas[y_utama:y_strip, x0:x_panel] = vis

    gambar_strip(kanvas, y_strip, top5, state["ambang"], state["rerank"])
    return kanvas
