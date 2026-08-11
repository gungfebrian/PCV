"""
Deteksi kepala penyu dengan YOLO, lalu potong dan seragamkan ke 512x512.

    # 1. latih detektor (butuh SeaTurtleID2022 yang punya mask kepala)
    python3 yolo_kepala.py --siapkan
    python3 yolo_kepala.py --latih

    # 2. potong seluruh dataset yang dipakai eksperimen
    MODEL=L python3 yolo_kepala.py --potong

    # 3. jalankan sebagai kondisi biasa
    MODEL=L python3 rerank.py --matcher xfeat --kondisi kepala --k 50

HARUS DIJALANKAN DI MAC. Bobot YOLO diunduh dari github.com/ultralytics/assets
yang diblokir dari sandbox eksperimen (ConnectionError, sudah diverifikasi).
Sisa kodenya jalan di mana saja.

Kenapa ini layak dicoba
-----------------------
Pada k = 84 langit-langitnya 100% tapi hasilnya 75,60%. Artinya 24,4% query
gagal karena XFeat menaruh foto yang SALAH di atas yang benar. Analisis
kegagalan sebelumnya menemukan pasangan yang benar hampir selalu dapat inlier;
yang salah justru dapat LEBIH BANYAK.

Tersangka utamanya: latar. Pasir berpola, riak air, karang, dan tangan
peneliti semuanya bertekstur kaya dan menghasilkan korespondensi yang
konsisten secara geometris antara dua foto yang sama sekali berbeda penyunya.
Membuang latar berarti membuang sumber inlier palsu itu.

Kenapa BUKAN crop tengah
------------------------
Kondisi `crop` (tengah 70%) sudah diuji dan TIDAK menolong. Itu bukan bukti
bahwa memotong tidak berguna — hanya bukti bahwa memotong secara buta tidak
berguna. Kepala penyu jarang berada tepat di tengah frame.

Sumber label kepala
-------------------
`SeaTurtleIDHeads/annotations.json` TIDAK punya bounding box: gambarnya sudah
berupa potongan kepala, dan anotasinya hanya identitas + posisi. Sudah
diperiksa, bukan diasumsikan.

Yang punya label kepala sebenarnya adalah **SeaTurtleID2022** (dataset induk),
yang memuat mask segmentasi untuk head / flippers / carapace:

    kaggle datasets download wildlifedatasets/seaturtleid2022

Mask kepala diubah jadi kotak pembatas di sini. Kalau dataset itu belum ada,
skrip berhenti dengan pesan yang jelas, bukan melatih pada data karangan.
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

import protokol as P

AKAR = os.path.dirname(P.BASE) if os.path.basename(P.BASE) == "eksperimen" \
    else P.BASE
DATASET = os.path.join(AKAR, "dataset_penyu")
SUMBER = os.path.join(DATASET, "SeaTurtleID2022")
KERJA = os.path.join(P.BASE, "yolo_kepala")
BOBOT = os.path.join(KERJA, "kepala.pt")
UKURAN_POTONG = 512
MARGIN = 0.18            # perlebar kotak; kepala yang terpotong lebih buruk
                         # daripada sedikit latar yang ikut


# --------------------------------------------- siapkan dari Zakynthos
def siapkan_zakynthos():
    """Data latih YOLO dari `Zakynthos/bbox.csv` — 160 kotak anotasi manusia.

    JALUR TERMUDAH, dan tidak perlu mengunduh apa pun. Zakynthos sudah ada di
    disk dan sudah punya kotak kepala.

    Split-nya BUKAN acak: latih pada foto GALERI (tahun pertama), uji pada
    foto QUERY (tahun kedua). Itu split yang sama dengan protokol §3, jadi
    detektornya tidak pernah melihat foto yang nanti dipakai mengukur.

    Kalau dibagi acak, foto individu yang sama dari sesi yang sama bisa
    tersebar ke latih dan uji sekaligus. Detektornya akan terlihat hebat dan
    angkanya tidak berarti apa-apa.

    160 gambar itu SEDIKIT untuk melatih detektor. Tapi kelasnya cuma satu,
    objeknya khas, dan tujuannya bukan detektor produksi — tujuannya menjawab
    satu pertanyaan: seberapa dekat detektor bisa mendekati plafon 63,75%
    yang dicapai kotak sempurna.
    """
    import csv
    os.environ.setdefault("DATASET", "zakynthos")
    if P.DATASET != "zakynthos":
        raise SystemExit("jalankan dengan DATASET=zakynthos")

    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    peran = {r["path"]: "train" for r in gal}
    peran.update({r["path"]: "val" for r in qry})

    for bagian in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(KERJA, "data", bagian, sub), exist_ok=True)

    n = {"train": 0, "val": 0}
    tanpa_kotak = []
    for path, bagian in peran.items():
        kotak = P.kotak_kepala_gt(path)
        if kotak is None:
            tanpa_kotak.append(os.path.basename(path))
            continue
        im = cv2.imread(path)
        if im is None:
            continue
        H, W = im.shape[:2]
        x, y, w, h = kotak
        nama = os.path.basename(path)
        dst = os.path.join(KERJA, "data", bagian, "images", nama)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(path), dst)
        with open(os.path.join(KERJA, "data", bagian, "labels",
                               os.path.splitext(nama)[0] + ".txt"), "w") as f:
            f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                    f"{w / W:.6f} {h / H:.6f}\n")
        n[bagian] += 1

    yaml = os.path.join(KERJA, "kepala.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {os.path.join(KERJA, 'data')}\n"
                "train: train/images\nval: val/images\n"
                "names:\n  0: kepala\n")
    print(f"siap: {n['train']} latih (galeri, tahun-1), "
          f"{n['val']} validasi (query, tahun-2)")
    if tanpa_kotak:
        print(f"  {len(tanpa_kotak)} foto tanpa kotak, dilewati")
    print(f"  -> {yaml}")
    print("\nBerikutnya:  python3 yolo_kepala.py --latih")


# ------------------------------------------------------------- siapkan
def siapkan():
    """COCO mask kepala -> kotak pembatas -> format YOLO."""
    ann = os.path.join(SUMBER, "annotations.json")
    if not os.path.exists(ann):
        raise SystemExit(
            f"{ann} tidak ada.\n\n"
            "SeaTurtleIDHeads TIDAK bisa dipakai untuk ini: gambarnya sudah\n"
            "berupa potongan kepala dan anotasinya tanpa bounding box (sudah\n"
            "diperiksa). Yang punya mask kepala adalah dataset induknya:\n\n"
            "    kaggle datasets download wildlifedatasets/seaturtleid2022 \\\n"
            f"      -p {DATASET} --unzip\n")

    d = json.load(open(ann))
    kat = {c["id"]: c["name"] for c in d["categories"]}
    id_kepala = [i for i, n in kat.items() if "head" in str(n).lower()]
    if not id_kepala:
        raise SystemExit(f"tidak ada kategori 'head'. Yang ada: {list(kat.values())}")

    gambar = {str(g["id"]): g for g in d["images"]}
    per_gambar = {}
    for a in d["annotations"]:
        if int(a.get("category_id", -1)) not in id_kepala:
            continue
        g = gambar.get(str(a["image_id"]))
        if not g:
            continue
        kotak = a.get("bbox")
        if not kotak and a.get("segmentation"):
            # mask -> kotak: ambil min/max dari poligonnya
            seg = a["segmentation"]
            pts = np.array(seg[0] if isinstance(seg[0], list) else seg,
                           np.float32).reshape(-1, 2)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            kotak = [x0, y0, x1 - x0, y1 - y0]
        if not kotak:
            continue
        per_gambar.setdefault(str(a["image_id"]), []).append((g, kotak))

    if not per_gambar:
        raise SystemExit("tidak ada anotasi kepala yang bisa dipakai")

    # split deterministik berbasis hash nama berkas — tanpa seed, tanpa acak,
    # konsisten dengan semangat protokol §3
    for bagian in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(KERJA, "data", bagian, sub), exist_ok=True)

    n = {"train": 0, "val": 0}
    for iid, daftar in per_gambar.items():
        g = daftar[0][0]
        nama = os.path.basename(g["file_name"])
        bagian = "val" if hash(nama) % 10 == 0 else "train"
        src = os.path.join(SUMBER, g["file_name"])
        if not os.path.exists(src):
            continue
        dst = os.path.join(KERJA, "data", bagian, "images", nama)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        W, H = g["width"], g["height"]
        baris = []
        for _, (x, y, w, h) in daftar:
            baris.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}")
        with open(os.path.join(KERJA, "data", bagian, "labels",
                               os.path.splitext(nama)[0] + ".txt"), "w") as f:
            f.write("\n".join(baris))
        n[bagian] += 1

    yaml = os.path.join(KERJA, "kepala.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {os.path.join(KERJA, 'data')}\n"
                "train: train/images\nval: val/images\n"
                "names:\n  0: kepala\n")
    print(f"siap: {n['train']} latih, {n['val']} validasi -> {yaml}")


# --------------------------------------------------------------- latih
def latih(epoch=40, ukuran=640):
    from ultralytics import YOLO
    yaml = os.path.join(KERJA, "kepala.yaml")
    if not os.path.exists(yaml):
        raise SystemExit("jalankan --siapkan dulu")
    m = YOLO("yolo11n.pt")          # nano: cukup untuk satu kelas, cepat di MPS
    m.train(data=yaml, epochs=epoch, imgsz=ukuran, batch=16,
            project=KERJA, name="latih", exist_ok=True)
    hasil = os.path.join(KERJA, "latih", "weights", "best.pt")
    shutil.copy(hasil, BOBOT)
    print("bobot terbaik ->", BOBOT)


# -------------------------------------------------------------- potong
def kotak_kepala(model, bgr, ambang=0.25):
    """Kotak kepala dengan kepercayaan tertinggi, atau None."""
    r = model.predict(bgr, verbose=False, conf=ambang)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0.0
    i = int(np.argmax(r.boxes.conf.cpu().numpy()))
    x0, y0, x1, y1 = r.boxes.xyxy.cpu().numpy()[i]
    return (float(x0), float(y0), float(x1), float(y1)), \
        float(r.boxes.conf.cpu().numpy()[i])


def potong_satu(bgr, kotak, ukuran=UKURAN_POTONG, margin=MARGIN):
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = kotak
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return cv2.resize(bgr[y0:y1, x0:x1], (ukuran, ukuran),
                      interpolation=cv2.INTER_AREA)


def potong_dataset():
    from ultralytics import YOLO
    if not os.path.exists(BOBOT):
        raise SystemExit(f"{BOBOT} tidak ada — jalankan --latih dulu")
    model = YOLO(BOBOT)

    kat = P.baca_katalog()
    keluar = os.path.join(DATASET, f"{P.DATASET}_kepala")
    os.makedirs(keluar, exist_ok=True)
    # SETIAP foto dapat entri, termasuk yang detektornya gagal. Yang gagal
    # memakai frame penuh yang diperkecil, dan ditandai `gagal: true`.
    #
    # Kenapa bukan dibuang: kalau dibuang, foto tersulit hilang dari evaluasi
    # dan akurasinya naik semu. Kenapa bukan berhenti dengan error: itu
    # membuat seluruh run gagal hanya karena satu foto.
    #
    # Cadangan frame penuh adalah yang akan dilakukan sistem sungguhan juga —
    # asal fraksinya DILAPORKAN, bukan disembunyikan.
    peta, gagal, rusak = {}, 0, 0
    for i, r in enumerate(kat):
        bgr = cv2.imread(r["path"])
        if bgr is None:
            rusak += 1
            continue
        kotak, conf = kotak_kepala(model, bgr)
        potong = potong_satu(bgr, kotak) if kotak else None
        jatuh = potong is None
        if jatuh:
            gagal += 1
            conf = 0.0
            potong = cv2.resize(bgr, (UKURAN_POTONG, UKURAN_POTONG),
                                interpolation=cv2.INTER_AREA)
        nama = f"{i:05d}.png"
        cv2.imwrite(os.path.join(keluar, nama), potong)
        peta[r["path"]] = {"berkas": nama, "kotak": kotak, "conf": conf,
                           "gagal": jatuh}
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(kat)}")

    with open(os.path.join(keluar, "peta.json"), "w") as f:
        json.dump(peta, f, indent=1)
    n = max(len(kat), 1)
    print(f"selesai: {len(peta)} foto -> {keluar}")
    print(f"  terdeteksi     : {len(peta) - gagal} ({100 * (len(peta) - gagal) / n:.1f}%)")
    print(f"  cadangan penuh : {gagal} ({100 * gagal / n:.1f}%)  <- ikut dievaluasi")
    if rusak:
        print(f"  berkas rusak   : {rusak}")
    if gagal:
        print(f"\nCATATAN WAJIB saat melaporkan angka: {100 * gagal / n:.1f}% "
              f"foto memakai frame penuh karena kepalanya tidak terdeteksi.\n"
              f"Angka re-ID di bawah ini adalah campuran crop dan frame penuh, "
              f"bukan crop murni.")


def ukur_deteksi(ambang=0.25):
    """Seberapa dekat YOLO dengan kotak anotasi manusia, pada foto QUERY.

    Dua angka yang berbeda dan keduanya penting:

      recall    berapa persen foto yang kepalanya KETEMU sama sekali
      IoU       kalau ketemu, seberapa pas kotaknya

    Foto yang tidak terdeteksi TIDAK boleh dibuang dari evaluasi re-ID.
    Kalau dibuang, akurasinya naik semu karena kasus tersulit hilang.
    """
    from ultralytics import YOLO
    if not os.path.exists(BOBOT):
        raise SystemExit(f"{BOBOT} tidak ada - jalankan --latih dulu")
    model = YOLO(BOBOT)
    kat = P.baca_katalog()
    _, qry = P.bangun_split(kat)

    ious, ketemu = [], 0
    for r in qry:
        gt = P.kotak_kepala_gt(r["path"])
        if gt is None:
            continue
        bgr = cv2.imread(r["path"])
        kotak, conf = kotak_kepala(model, bgr, ambang)
        if kotak is None:
            ious.append(0.0)
            continue
        ketemu += 1
        gx0, gy0 = gt[0], gt[1]
        gx1, gy1 = gt[0] + gt[2], gt[1] + gt[3]
        px0, py0, px1, py1 = kotak
        ix = max(0, min(gx1, px1) - max(gx0, px0))
        iy = max(0, min(gy1, py1) - max(gy0, py0))
        inter = ix * iy
        uni = gt[2] * gt[3] + (px1 - px0) * (py1 - py0) - inter
        ious.append(inter / max(uni, 1e-9))

    ious = np.array(ious)
    n = len(ious)
    print(f"n = {n} foto query")
    print(f"  recall deteksi   : {100 * ketemu / n:.2f}%  "
          f"({n - ketemu} foto tidak terdeteksi sama sekali)")
    print(f"  IoU rata-rata    : {ious.mean():.3f}")
    print(f"  IoU >= 0,5       : {100 * (ious >= 0.5).mean():.2f}%")
    print(f"  IoU >= 0,7       : {100 * (ious >= 0.7).mean():.2f}%")
    print("\nPlafon re-ID dengan kotak SEMPURNA: Rank-1 63,75%")
    print("Berikutnya, ukur re-ID dengan kotak YOLO:")
    print("  MODEL=L DATASET=zakynthos python3 yolo_kepala.py --potong")
    print("  MODEL=L DATASET=zakynthos python3 jalankan.py kepala")
    print("  MODEL=L DATASET=zakynthos STAGE1=kepala python3 rerank.py \\")
    print("      --matcher xfeat --kondisi kepala --k 10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siapkan-zakynthos", action="store_true",
                    help="JALUR TERMUDAH - pakai bbox.csv yang sudah ada")
    ap.add_argument("--siapkan", action="store_true",
                    help="dari SeaTurtleID2022 (perlu diunduh, detektor lebih kuat)")
    ap.add_argument("--ukur", action="store_true",
                    help="ukur kualitas deteksi: IoU vs kotak anotasi")
    ap.add_argument("--latih", action="store_true")
    ap.add_argument("--potong", action="store_true")
    ap.add_argument("--epoch", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(KERJA, exist_ok=True)
    if a.siapkan_zakynthos:
        siapkan_zakynthos()
    elif a.ukur:
        ukur_deteksi()
    elif a.siapkan:
        siapkan()
    elif a.latih:
        latih(a.epoch)
    elif a.potong:
        potong_dataset()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
