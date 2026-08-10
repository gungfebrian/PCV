"""
Unduh deck kartu resolusi tinggi jadi Templatekartu_hd/.

Kenapa perlu: Templatekartu/ berisi FOTO kartu (ada bayangan, miring sedikit,
pencahayaan tidak rata, dan 2 kartu hilang). Deck ini gambar vektor yang
di-render — bersih, konsisten, lengkap 52. Template yang konsisten bikin skor
SAD jauh lebih bisa dipercaya.

Sumber: github.com/hayeah/playing-cards-assets (deck PNG bebas pakai).

Jalankan:
    .venv/bin/python unduh_template.py
"""

import os
import urllib.error
import urllib.request

import cv2
import numpy as np

BASE = "https://raw.githubusercontent.com/hayeah/playing-cards-assets/master/png"
TUJUAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Templatekartu_hd")

RANKS = {"A": "ace", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
         "8": "8", "9": "9", "10": "10", "J": "jack", "Q": "queen", "K": "king"}
SUITS = {"Spade": "spades", "Heart": "hearts", "Diamond": "diamonds", "Club": "clubs"}


def unduh():
    os.makedirs(TUJUAN, exist_ok=True)
    ok = gagal = lewat = 0

    for suit, s_web in SUITS.items():
        for rank, r_web in RANKS.items():
            # Nama file disesuaikan dengan pola yang dipakai pipeline: "K_Heart.jpg"
            keluar = os.path.join(TUJUAN, f"{rank}_{suit}.jpg")
            if os.path.exists(keluar):
                lewat += 1
                continue

            url = f"{BASE}/{r_web}_of_{s_web}.png"
            try:
                with urllib.request.urlopen(url, timeout=25) as r:
                    data = r.read()
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  gagal {rank}_{suit}: {e}")
                gagal += 1
                continue

            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  gagal decode {rank}_{suit}")
                gagal += 1
                continue

            # PNG punya kanal alpha; ratakan ke latar putih supaya threshold
            # tidak menganggap area transparan sebagai tinta hitam.
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                putih = np.full(img[:, :, :3].shape, 255, np.float32)
                img = (img[:, :, :3] * alpha + putih * (1 - alpha)).astype(np.uint8)
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(keluar, cv2.resize(img, (300, 420),
                                           interpolation=cv2.INTER_AREA))
            ok += 1
            print(f"  ok {rank}_{suit}")

    print(f"\nSelesai: {ok} diunduh, {lewat} sudah ada, {gagal} gagal")
    print(f"Folder : {TUJUAN}")
    total = len([f for f in os.listdir(TUJUAN) if f.endswith('.jpg')])
    print(f"Total  : {total}/52 template")
    return total


if __name__ == "__main__":
    print(f"Mengunduh 52 kartu ke {TUJUAN} ...")
    unduh()
