"""
Periksa angka yang kelihatan TERLALU BAGUS. Jalankan sebelum mempercayainya.

    DATASET=seaturtleheads MODEL=L python3 periksa_curiga.py

Kenapa berkas ini ada
---------------------
Angka tinggi jauh lebih berbahaya daripada angka rendah. Angka rendah membuat
orang mencari kesalahan; angka tinggi membuat orang berhenti mencari.

Spesifikasi §8 menyebut lima tersangka kalau baseline aneh. Berkas ini
memeriksa semuanya sekaligus, plus beberapa yang ditemukan di jalan.

Aturan praktis dari proyek ini: kalau sebuah perubahan memberi lompatan lebih
dari ~20 poin dan kamu tidak bisa menjelaskan mekanismenya dalam satu kalimat,
periksa dulu. Sudah dua kali angka bagus ternyata bug.
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np

import protokol as P
from evaluasi import muat

HASIL = os.path.join(P.BASE, "hasil", f"{P.DATASET}_{P.MODEL}_{P.TRANSFORM}")

LULUS, GAGAL, PERIKSA = "LULUS", "GAGAL", "PERIKSA"
_hasil = []


def lapor(nama, status, pesan):
    _hasil.append((status, nama))
    warna = {"LULUS": "\033[32m", "GAGAL": "\033[31m",
             "PERIKSA": "\033[33m"}[status]
    print(f"  {warna}{status:8}\033[0m {nama}")
    if pesan:
        for baris in str(pesan).split("\n"):
            print(f"           {baris}")


def main(kondisi="raw"):
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    print(f"dataset {P.DATASET} / model {P.MODEL} / kondisi {kondisi}")
    print(f"galeri {len(gal)}  query {len(qry)}\n")

    # ---------------------------------------------------------- 1. tumpang tindih
    pg, pq = {r["path"] for r in gal}, {r["path"] for r in qry}
    sama = pg & pq
    lapor("berkas yang sama muncul di galeri DAN query",
          GAGAL if sama else LULUS,
          f"{len(sama)} berkas bocor, contoh: {list(sama)[:3]}" if sama
          else "tidak ada tumpang tindih")

    # ---------------------------------------------------------- 2. arah tahun
    tahun_g = defaultdict(set)
    for r in gal:
        tahun_g[(r["identity"], r["side"])].add(r["year"])
    salah = [r for r in qry
             if any(y >= r["year"] for y in tahun_g.get((r["identity"],
                                                         r["side"]), {9999}))]
    lapor("query lebih baru dari galeri", GAGAL if salah else LULUS,
          f"{len(salah)} query tahunnya TIDAK lebih baru" if salah
          else "semua query dari tahun setelah galerinya")

    # -------------------------------------------- 3. foto dari HARI yang sama
    # Ini tersangka terkuat untuk angka yang terlalu bagus. Kalau galeri dan
    # query berisi foto dari sesi pemotretan yang sama, sistem mencocokkan
    # dua gambar yang nyaris identik - dan angkanya tidak berarti apa-apa.
    tgl = {r["path"]: r.get("date") or r.get("year") for r in kat}
    hari_g = defaultdict(set)
    for r in gal:
        hari_g[r["identity"]].add(tgl.get(r["path"]))
    tabrakan = sum(1 for r in qry if tgl.get(r["path"]) in hari_g[r["identity"]])
    lapor("query dan galeri dari tanggal yang sama",
          GAGAL if tabrakan else LULUS,
          f"{tabrakan}/{len(qry)} query punya foto galeri dari TANGGAL SAMA"
          if tabrakan else "tidak ada tanggal yang bertabrakan")

    # ---------------------------------------------------------- 4. embedding
    try:
        Eg, Eq = muat(kondisi, gal, qry)
    except Exception as e:
        lapor("embedding bisa dimuat", GAGAL, str(e))
        return rangkum()

    ng = np.linalg.norm(Eg, axis=1)
    nq = np.linalg.norm(Eq, axis=1)
    ok_norm = np.allclose(ng, 1, atol=1e-4) and np.allclose(nq, 1, atol=1e-4)
    lapor("embedding ternormalisasi L2", LULUS if ok_norm else GAGAL,
          f"norma galeri {ng.min():.4f}-{ng.max():.4f}, "
          f"query {nq.min():.4f}-{nq.max():.4f}")

    nol_g = int((ng < 1e-6).sum())
    nol_q = int((nq < 1e-6).sum())
    lapor("tidak ada embedding kosong",
          GAGAL if (nol_g or nol_q) else LULUS,
          f"{nol_g} galeri + {nol_q} query bernilai nol "
          f"(berarti embed gagal diam-diam)" if (nol_g or nol_q)
          else "semua terisi")

    # baris identik = gambar yang sama diproses dua kali
    _, idx = np.unique(np.round(Eg, 5), axis=0, return_index=True)
    dup = len(Eg) - len(idx)
    lapor("tidak ada embedding galeri yang identik",
          PERIKSA if dup else LULUS,
          f"{dup} baris galeri identik dengan baris lain" if dup
          else "semua unik")

    # ------------------------------------------------- 5. struktur galeri
    per = Counter((r["identity"], r["side"]) for r in gal)
    nilai = sorted(per.values())
    med = nilai[len(nilai) // 2]
    lapor("foto galeri per (individu, sisi)",
          PERIKSA if med >= 5 else LULUS,
          f"min {nilai[0]}  median {med}  maks {nilai[-1]}\n"
          f"median tinggi membuat tugasnya JAUH lebih mudah - "
          f"bukan bug, tapi harus disebut saat membandingkan dataset")

    yatim = sum(1 for r in qry
                if (r["identity"], r["side"]) not in per)
    lapor("setiap query punya pasangan di galeri sisi sama",
          GAGAL if yatim else LULUS,
          f"{yatim} query tanpa pasangan" if yatim else "lengkap")

    # ------------------------------------------------------- 6. kunci sisi
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    s_g = np.array([r["side"] for r in gal])
    s_q = np.array([r["side"] for r in qry])

    S = Eq @ Eg.T
    S_kunci = S.copy()
    S_kunci[s_q[:, None] != s_g[None, :]] = -np.inf
    h_kunci = P.metrik_dari_matriks(S_kunci, id_q, id_g)
    h_bebas = P.metrik_dari_matriks(S.copy(), id_q, id_g)
    r1k = 100 * h_kunci["rank1"].mean()
    r1b = 100 * h_bebas["rank1"].mean()
    lapor("kunci sisi benar-benar berpengaruh",
          PERIKSA if abs(r1k - r1b) < 0.5 else LULUS,
          f"dengan kunci {r1k:.2f}%  tanpa kunci {r1b:.2f}%\n"
          f"kalau nyaris sama, kunci sisinya mungkin tidak terpasang")

    # ------------------------------------------- 7. seberapa mudah tugasnya
    kandidat = int(np.isfinite(S_kunci[0]).sum())
    acak = 100.0 / max(len({(i, s) for i, s in zip(id_g, s_g)}), 1)
    lapor("Rank-1 dibandingkan tebakan acak", LULUS,
          f"Rank-1 {r1k:.2f}%  vs tebak acak {acak:.2f}%  "
          f"({kandidat} kandidat per query)")

    # ------------------------------- 8. seberapa dekat jawaban benar vs salah
    urut = np.argsort(-S_kunci, axis=1)
    benar = id_g[urut] == id_q[:, None]
    skor1 = np.take_along_axis(S_kunci, urut[:, :1], 1).ravel()
    skor2 = np.take_along_axis(S_kunci, urut[:, 1:2], 1).ravel()
    margin = float(np.median(skor1 - skor2))
    lapor("margin antara peringkat 1 dan 2",
          PERIKSA if margin > 0.25 else LULUS,
          f"median {margin:.4f}\n"
          f"margin sangat lebar sering berarti query dan galeri "
          f"terlalu mirip - periksa apakah fotonya near-duplicate")

    print()
    return rangkum()


def rangkum():
    n_gagal = sum(1 for s, _ in _hasil if s == GAGAL)
    n_periksa = sum(1 for s, _ in _hasil if s == PERIKSA)
    if n_gagal:
        print(f"{n_gagal} GAGAL - angkanya TIDAK bisa dipercaya. "
              f"Perbaiki dulu sebelum melaporkan apa pun.")
        return 1
    if n_periksa:
        print(f"{n_periksa} perlu diperiksa manual. Tidak otomatis salah, "
              f"tapi harus bisa dijelaskan sebelum angkanya dikutip.")
        return 0
    print("Semua lulus. Angkanya boleh dipercaya - "
          "dengan tetap menyebut n dan p.")
    return 0


if __name__ == "__main__":
    kondisi = sys.argv[1] if len(sys.argv) > 1 else "raw"
    raise SystemExit(main(kondisi))
