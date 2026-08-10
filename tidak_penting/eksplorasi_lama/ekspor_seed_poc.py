"""
Ekspor SeaTurtleIDHeads ke format seed turtle-identification-poc.

POC menunggu berkas bernama `t278_bYZNTwocWk.jpg` (lihat
be/app/seed_images.py: `parse_turtle_code`), yaitu `<kode>_<id>.jpg`.
SeaTurtleIDHeads menyimpannya sebagai `images/t278/bYZNTwocWk.JPG` — jadi
datanya sudah cocok, hanya perlu diratakan namanya.

Berkas di-symlink, bukan disalin: 7.500 foto = 400 MB, dan symlink membuat
ekspor ulang menjadi instan tanpa menggandakan ruang. Pakai --salin kalau
folder tujuan akan dipindah ke mesin lain.

Jalankan:
    .venv/bin/python ekspor_seed_poc.py                  # semua individu
    .venv/bin/python ekspor_seed_poc.py --maks-foto 8    # batasi per individu
    .venv/bin/python ekspor_seed_poc.py --salin          # salin, bukan symlink

Lalu di POC:
    python be/scripts/preseed_images.py <folder_keluaran>
"""

import argparse
import os
import re
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMBER = os.path.join(BASE_DIR, "dataset_penyu", "SeaTurtleIDHeads", "images")
TUJUAN = os.path.join(BASE_DIR, "seed_poc")

# Pola yang diterima POC: t + angka. Individu di luar pola ini dilewati,
# karena preseed_images.py akan menolaknya juga.
POLA_KODE = re.compile(r"^t\d+$", re.IGNORECASE)


def ekspor(sumber=SUMBER, tujuan=TUJUAN, maks_foto=None, salin=False):
    if not os.path.isdir(sumber):
        print(f"Dataset tidak ada di {sumber}")
        print("Unduh dulu: .venv/bin/python unduh_dataset_penyu.py")
        return 0

    os.makedirs(tujuan, exist_ok=True)
    total = individu = dilewati = 0

    for kode in sorted(os.listdir(sumber)):
        d = os.path.join(sumber, kode)
        if not os.path.isdir(d):
            continue
        if not POLA_KODE.fullmatch(kode):
            dilewati += 1
            continue

        fotos = sorted(n for n in os.listdir(d)
                       if n.lower().endswith((".jpg", ".jpeg", ".png")))
        if maks_foto:
            fotos = fotos[:maks_foto]
        if not fotos:
            continue

        individu += 1
        for n in fotos:
            asal = os.path.join(d, n)
            # POC memakai suffix huruf kecil untuk mencocokkan SEED_IMAGE_SUFFIXES.
            batang, ext = os.path.splitext(n)
            keluar = os.path.join(tujuan, f"{kode.lower()}_{batang}{ext.lower()}")
            if os.path.lexists(keluar):
                continue
            if salin:
                shutil.copy2(asal, keluar)
            else:
                os.symlink(asal, keluar)
            total += 1

    print(f"Individu diekspor : {individu}")
    print(f"Foto              : {total} ({'salinan' if salin else 'symlink'})")
    if dilewati:
        print(f"Dilewati (nama tidak cocok pola t<angka>): {dilewati}")
    print(f"Folder            : {tujuan}")
    print(f"\nLangkah berikutnya di POC:")
    print(f"  python be/scripts/preseed_images.py {tujuan}")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maks-foto", type=int, default=None,
                    help="batasi jumlah foto per individu")
    ap.add_argument("--salin", action="store_true",
                    help="salin berkas, bukan symlink")
    ap.add_argument("--tujuan", default=TUJUAN)
    a = ap.parse_args()
    ekspor(tujuan=a.tujuan, maks_foto=a.maks_foto, salin=a.salin)
