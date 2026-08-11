"""
Tes kesetaraan: apakah APLIKASI memberi jawaban yang sama dengan EKSPERIMEN?

    DATASET=zakynthos MODEL=L python3 uji_setara.py
    DATASET=zakynthos MODEL=L python3 uji_setara.py --n 40

Kenapa berkas ini ada
---------------------
`uji_tampilan.py` menguji tata letak. `eksperimen/uji.py` menguji protokol.
Tidak satu pun dari keduanya bisa menangkap kelas bug yang paling berbahaya di
proyek ini: **aplikasi dan eksperimen pelan-pelan berbeda tanpa ada yang sadar.**

Dua bug seperti itu sudah benar-benar terjadi:

1. Aplikasi menerapkan preprocessing ke KEDUA stage, padahal eksperimen hanya
   ke stage-2. Konfigurasi terbaik jadi tidak bisa direproduksi sama sekali.

2. Aplikasi membuat matcher tanpa kondisi (`buat_matcher(nama)`), sehingga
   fitur GALERI diambil dari gambar raw sementara fitur QUERY dari gambar
   yang sudah di-crop. Hasilnya Rank-1 30% padahal eksperimen 67,50% —
   separuh akurasi hilang, tanpa satu pun pesan galat.

Keduanya lolos semua tes yang ada waktu itu. Yang menangkapnya cuma
membandingkan angka aplikasi dengan angka eksperimen secara langsung.

Cara kerjanya
-------------
Jalankan N query pertama lewat jalur APLIKASI (`Mesin.kenali`), lalu lewat
jalur EKSPERIMEN (matriks skor yang tersimpan), dan bandingkan **keputusan per
query**, bukan cuma rata-ratanya. Dua sistem bisa kebetulan punya rata-rata
sama tapi salah di query yang berbeda.
"""

import argparse
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "eksperimen"))

import protokol as P            # noqa: E402
import rerank as R              # noqa: E402


def jalur_eksperimen(gal, qry, kondisi, k, n):
    """Keputusan rank-1 dari matriks skor yang sudah tersimpan."""
    tag = "" if R.KONDISI_STAGE1 == "raw" else f"_s1-{R.KONDISI_STAGE1}"
    p = os.path.join(R.HASIL, f"rerank_xfeat-{kondisi}{tag}_k{k}.npy")
    if not os.path.exists(p):
        raise SystemExit(
            f"berkas hasil eksperimen tidak ada:\n  {p}\n"
            f"Jalankan dulu:\n"
            f"  DATASET={P.DATASET} MODEL={P.MODEL} STAGE1={R.KONDISI_STAGE1} "
            f"python3 rerank.py --matcher xfeat --kondisi {kondisi} --k {k}")
    M = np.load(p)
    _, top, k = R.kandidat_stage1(gal, qry, k)
    id_g = np.array([r["identity"] for r in gal])
    keluar = []
    for i in range(n):
        # urutan yang sama persis dengan `rerank.evaluasi_rerank` mode 'murni'
        o = np.lexsort((np.arange(k), -M[i]))
        keluar.append(id_g[top[i][o[0]]])
    return keluar


def jalur_aplikasi(mesin, qry, kondisi, stage1, k, n):
    keluar = []
    for r in qry[:n]:
        h, _, t, _, _, _ = mesin.kenali(
            cv2.imread(r["path"]), kondisi, r["side"], "murni", k,
            "xfeat", stage1, r["path"])
        keluar.append(None if h is None else h["nama"])
    return keluar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kondisi", default="kepala_gt")
    ap.add_argument("--stage1", default=None,
                    help="default: ikut env STAGE1 yang dipakai eksperimen")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    stage1 = a.stage1 or R.KONDISI_STAGE1

    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    n = min(a.n, len(qry))
    print(f"dataset {P.DATASET} / model {P.MODEL} / stage1 {stage1} / "
          f"stage2 {a.kondisi} / k {a.k} / n {n}")

    eks = jalur_eksperimen(gal, qry, a.kondisi, a.k, n)

    import penyu_live as A
    mesin = A.Mesin()
    app = jalur_aplikasi(mesin, qry, a.kondisi, stage1, a.k, n)

    # Aplikasi memangkas prefiks spesies untuk TAMPILAN ("Green/Baguette" ->
    # "Baguette"). Itu pilihan tampilan, bukan perbedaan keputusan. Kedua sisi
    # dinormalkan ke segmen terakhir supaya yang dibandingkan identitasnya,
    # bukan cara menuliskannya.
    def norm(x):
        return None if x is None else str(x).split("/")[-1]

    eks = [norm(x) for x in eks]
    app = [norm(x) for x in app]
    id_q = [norm(r["identity"]) for r in qry[:n]]

    beda = [(i, e, b) for i, (e, b) in enumerate(zip(eks, app)) if e != b]
    benar_e = sum(e == q for e, q in zip(eks, id_q))
    benar_a = sum(b == q for b, q in zip(app, id_q))

    print(f"\n  eksperimen : {benar_e}/{n} = {100 * benar_e / n:.1f}% rank-1")
    print(f"  aplikasi   : {benar_a}/{n} = {100 * benar_a / n:.1f}% rank-1")
    print(f"  keputusan berbeda: {len(beda)}/{n}")

    if beda:
        print("\n  QUERY YANG BERBEDA (10 pertama):")
        for i, e, b in beda[:10]:
            print(f"    #{i:3} benar={id_q[i]:12} eksperimen={e:12} aplikasi={b}")
        print("\nGAGAL — aplikasi dan eksperimen tidak setara.")
        print("Tersangka yang sudah pernah terjadi:")
        print("  - matcher dibuat tanpa kondisi (galeri raw vs query crop)")
        print("  - preprocessing diterapkan ke stage yang salah")
        print("  - embedding stage-1 tidak cocok dengan kondisi query")
        return 1

    print("\nLOLOS — setiap keputusan identik.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
