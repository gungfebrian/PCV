"""
Unduh dataset penyu dari Kaggle lewat wildlife-datasets.

KENAPA INI PENTING
------------------
Seluruh pengukuran sejauh ini menunjuk satu hambatan: pola sisik penyu tidak
pernah terpisah dari latar. 81.3% keypoint jatuh di pasir dan air, dan jarak
antara pasangan individu SAMA (0.3101) hampir tak terpisahkan dari pasangan
BEDA (0.3588).

SeaTurtleIDHeads menyelesaikan itu di hulu: isinya foto KEPALA penyu yang
sudah terpotong. Kalau hipotesis kami benar, akurasinya harus melonjak jauh
di atas 26.5% tanpa mengubah apa pun selain datanya. Kalau ternyata tidak
melonjak, hipotesisnya salah dan itu sama berharganya untuk diketahui.

    SeaTurtleIDHeads   kepala sudah dipotong  <- pakai ini untuk menguji
    SeaTurtleID2022    foto utuh + anotasi segmentasi & bbox kepala

LANGKAH SEKALI SAJA (kredensial Kaggle)
---------------------------------------
1. Buka https://www.kaggle.com/settings/account
2. Bagian "API" -> "Create New Token" -> mengunduh kaggle.json
3. Jalankan:

       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

4. Setujui syarat datasetnya sekali di halaman ini (wajib, kalau tidak
   unduhannya ditolak dengan error 403):

       https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads

Lalu jalankan:

    .venv/bin/python unduh_dataset_penyu.py

Catatan: modul wildlife_datasets ada di venv turtle Anda, bukan di .venv PCV.
Skrip ini memakai penerjemah itu bila tersedia.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TUJUAN = os.path.join(BASE_DIR, "dataset_penyu")
VENV_TURTLE = "/Users/gung/.venvs/turtle/bin/python"

PILIHAN = {
    "heads": ("SeaTurtleIDHeads",
              "kepala penyu sudah terpotong — untuk menguji hipotesis crop"),
    "full": ("SeaTurtleID2022",
             "foto utuh + anotasi segmentasi dan bbox kepala"),
}


def punya_kredensial():
    return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def penerjemah():
    """wildlife_datasets ada di venv turtle, bukan di .venv PCV."""
    if os.path.exists(VENV_TURTLE):
        cek = subprocess.run(
            [VENV_TURTLE, "-c", "import wildlife_datasets"],
            capture_output=True)
        if cek.returncode == 0:
            return VENV_TURTLE
    return sys.executable


def unduh(kunci="heads"):
    nama, guna = PILIHAN[kunci]
    folder = os.path.join(TUJUAN, nama)
    os.makedirs(folder, exist_ok=True)

    py = penerjemah()
    print(f"Dataset  : {nama}  ({guna})")
    print(f"Tujuan   : {folder}")
    print(f"Python   : {py}\n")

    kode = (
        "from wildlife_datasets import datasets\n"
        f"datasets.{nama}.get_data({folder!r})\n"
        f"d = datasets.{nama}({folder!r})\n"
        "print('SELESAI')\n"
        "print('gambar   :', len(d.df))\n"
        "print('individu :', d.df['identity'].nunique())\n"
        "print('kolom    :', list(d.df.columns))\n"
    )
    hasil = subprocess.run([py, "-c", kode], capture_output=False)
    return hasil.returncode == 0


if __name__ == "__main__":
    kunci = sys.argv[1] if len(sys.argv) > 1 else "heads"
    if kunci not in PILIHAN:
        print(f"Pilihan: {', '.join(PILIHAN)}")
        raise SystemExit(1)

    if not punya_kredensial():
        print("Kredensial Kaggle belum ada di ~/.kaggle/kaggle.json.\n")
        print("Langkah sekali saja:")
        print("  1. https://www.kaggle.com/settings/account -> API -> Create New Token")
        print("  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Setujui syarat dataset di:")
        print("     https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads")
        raise SystemExit(1)

    raise SystemExit(0 if unduh(kunci) else 1)
