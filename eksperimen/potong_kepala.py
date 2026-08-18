"""
Buat potongan kepala dengan detektor YOLO pada ambang keyakinan tertentu.

    MODEL=MIEWID DATASET=zakynthos python3 potong_kepala.py --konf 0.25
    MODEL=MIEWID DATASET=zakynthos python3 potong_kepala.py --konf 0.05

Kenapa ada
----------
Potongan lama di `dataset_penyu/{ds}_kepala_lintas/` dibuat di worktree dengan
ambang 0,25, dan 15 dari 160 foto Zakynthos (9%) GAGAL terdeteksi — sementara
Amvrakikos nol gagal. Selisih akurasi detektor-vs-anotasi juga cuma muncul di
Zakynthos (12,5 poin, p=0,013), jadi kegagalan itu tersangka utamanya.

Pemeriksaan pada 15 foto tersebut: 12 terdeteksi di ambang 0,02, dan 9 di
antaranya berkotak bagus (IoU 0,69-0,90). Jadi detektornya SUDAH melihat
kepalanya; skornya saja di bawah ambang.

Skrip ini membuat potongan pada ambang yang bisa diatur supaya hipotesis itu
bisa diuji, bukan diperdebatkan.

Geometri dikunci ke P.kepala_gt()
---------------------------------
margin 18%, resize INTER_AREA ke 512x512. WAJIB sama, karena kalau potongan
detektor dan potongan anotasi dibuat dengan geometri berbeda, selisih
akurasinya mencampur dua sebab dan tidak bisa dibaca.

Itu juga alasan menjalankan --konf 0.25 di sini meskipun potongan lama sudah
ada: supaya perbandingan 0,25 vs 0,05 hanya berbeda pada ambangnya.
"""

import argparse
import json
import os

import cv2

import protokol as P

BOBOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "yolo_kepala", "kepala_lintas.pt")


def potong(bgr, kotak, ukuran=512, margin=0.18):
    """Salinan geometri P.kepala_gt(). Jangan diubah sendirian."""
    h, w = bgr.shape[:2]
    x, y, bw, bh = kotak
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return cv2.resize(bgr[y0:y1, x0:x1], (ukuran, ukuran),
                      interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--konf", type=float, required=True)
    a = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(BOBOT)

    tag = f"kepala_c{int(round(a.konf * 100)):02d}"
    keluar = os.path.join(os.path.dirname(P.BASE), "dataset_penyu",
                          f"{P.DATASET}_{tag}")
    os.makedirs(keluar, exist_ok=True)

    kat = P.baca_katalog()
    peta, gagal = {}, 0
    for n, r in enumerate(kat):
        bgr = cv2.imread(r["path"])
        if bgr is None:
            gagal += 1
            continue
        res = det.predict(bgr, conf=a.konf, verbose=False)[0]
        pot = None
        skor = 0.0
        if res.boxes is not None and len(res.boxes):
            i = int(res.boxes.conf.argmax())
            x0, y0, x1, y1 = (float(v) for v in res.boxes.xyxy[i])
            skor = float(res.boxes.conf[i])
            pot = potong(bgr, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        rusak = pot is None
        if rusak:
            # Fallback: frame penuh diperas ke 512. Ini MENIRU perilaku
            # potongan lama (sudah diverifikasi identik), supaya perbandingan
            # antar ambang hanya berbeda pada ambangnya.
            #
            # Perlu disadari: fallback ini mencampur gambar TANPA crop ke
            # dalam kondisi yang seharusnya ter-crop. Itu bukan kecelakaan
            # melainkan pilihan produksi — kalau detektor gagal, sistem tetap
            # harus menjawab. Tapi angkanya harus dibaca sebagai "detektor
            # plus fallback", bukan "detektor".
            gagal += 1
            pot = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA)
        nama = f"{n:05d}.png"
        cv2.imwrite(os.path.join(keluar, nama), pot)
        # Kunci = nama berkas, bukan path absolut. Peta lama memakai path
        # absolut worktree dan itu tidak cocok di clone lain.
        peta[os.path.basename(r["path"])] = {
            "berkas": nama, "conf": skor, "gagal": rusak}

    with open(os.path.join(keluar, "peta.json"), "w") as f:
        json.dump(peta, f, indent=1)
    print(f"{P.DATASET} konf={a.konf}: {len(peta)}/{len(kat)} terpotong, "
          f"{gagal} gagal  ->  {keluar}")


if __name__ == "__main__":
    main()
