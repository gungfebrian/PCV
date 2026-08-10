"""
KONSENSUS DUA SISI — dua foto (kiri + kanan) harus sepakat pada satu nama.

Gagasan: satu foto memberi satu peringkat, dan peringkat teratas dari satu
foto bisa saja kebetulan. Kalau foto sisi KIRI dan foto sisi KANAN dari penyu
yang sama sama-sama menunjuk nama yang sama, kebetulan itu jauh lebih kecil
kemungkinannya. Dua bukti independen yang sepakat jauh lebih kuat dari satu.

Kenapa ini masuk akal untuk penyu, bukan sekadar trik:
    - Pola sisik kiri dan kanan adalah pola yang BERBEDA (Notion bagian 22:
      mencocokkan dalam satu sisi naik +17-20 poin dibanding sisi tercampur).
    - Karena itu embedding kiri dan kanan membawa informasi yang saling bebas.
      Dua sumber bukti yang bebas boleh dikalikan peluangnya.

Kalau kedua sisi TIDAK sepakat, itu sinyal berharga, bukan kegagalan: berarti
sistem sedang menebak, dan jawabannya "perlu diperiksa manusia" — jauh lebih
jujur daripada memaksakan satu nama.

Galeri harus menyimpan sisi secara terpisah:
    galeri = {"t023": {"left": [vek, ...], "right": [vek, ...]}, ...}
"""

import numpy as np

# Bobot bukti: peringkat-1 bernilai penuh, peringkat berikutnya meluruh.
# Dipakai saat kedua sisi tidak sepakat pada peringkat-1, supaya kandidat yang
# konsisten berada di peringkat atas KEDUA sisi tetap bisa menang.
def _skor_peringkat(peringkat, n=5):
    """Ubah peringkat (nama, jarak) jadi {nama: skor}, skor tinggi = baik."""
    return {nama: 1.0 / (i + 1) for i, (nama, _) in enumerate(peringkat[:n])}


def cocokkan_dua_sisi(peringkat_kiri, peringkat_kanan, prob_fn=None):
    """Gabungkan dua peringkat menjadi satu putusan.

    peringkat_*: list (nama, jarak) terurut dari paling mirip.
    prob_fn: fungsi jarak -> P(sama), mis. turtle_mode.prob_sama.

    Return dict dengan status:
        'sepakat'      kedua sisi menunjuk nama yang sama di peringkat-1
        'sebagian'     nama teratas berbeda, tapi satu nama kuat di keduanya
        'bertentangan' kedua sisi menunjuk nama berbeda tanpa titik temu
        'satu sisi'    hanya satu peringkat tersedia
    """
    ada_kiri = bool(peringkat_kiri)
    ada_kanan = bool(peringkat_kanan)

    if not ada_kiri and not ada_kanan:
        return {"nama": None, "status": "kosong", "keyakinan": 0.0,
                "rincian": "tidak ada foto"}

    if ada_kiri != ada_kanan:
        satu = peringkat_kiri if ada_kiri else peringkat_kanan
        sisi = "kiri" if ada_kiri else "kanan"
        nama, jarak = satu[0]
        p = prob_fn(jarak) if prob_fn else 0.5
        return {"nama": nama, "status": "satu sisi", "keyakinan": p,
                "rincian": f"hanya sisi {sisi}: {nama} ({p:.0%})",
                "jarak_kiri": jarak if ada_kiri else None,
                "jarak_kanan": jarak if ada_kanan else None}

    n_ki, j_ki = peringkat_kiri[0]
    n_ka, j_ka = peringkat_kanan[0]
    p_ki = prob_fn(j_ki) if prob_fn else 0.5
    p_ka = prob_fn(j_ka) if prob_fn else 0.5

    if n_ki == n_ka:
        # Dua bukti bebas yang sepakat. Digabung dengan aturan Bayes untuk
        # dua pengamatan bebas: peluang gabungan menguat ke arah yang sama.
        # p_gab = p1*p2 / (p1*p2 + (1-p1)*(1-p2))
        atas = p_ki * p_ka
        bawah = atas + (1 - p_ki) * (1 - p_ka)
        gab = atas / bawah if bawah > 0 else 0.5
        return {"nama": n_ki, "status": "sepakat", "keyakinan": gab,
                "rincian": f"kiri {p_ki:.0%} + kanan {p_ka:.0%} -> {gab:.0%}",
                "jarak_kiri": j_ki, "jarak_kanan": j_ka}

    # Tidak sepakat di peringkat-1: cari kandidat terbaik di kedua peringkat.
    sk = _skor_peringkat(peringkat_kiri)
    sa = _skor_peringkat(peringkat_kanan)
    bersama = set(sk) & set(sa)
    if bersama:
        terbaik = max(bersama, key=lambda k: sk[k] + sa[k])
        # Keyakinan sengaja ditahan: kedua sisi tidak sepakat di peringkat-1,
        # jadi ini kandidat, bukan kesimpulan.
        jk = dict(peringkat_kiri)[terbaik]
        ja = dict(peringkat_kanan)[terbaik]
        p = ((prob_fn(jk) + prob_fn(ja)) / 2) if prob_fn else 0.5
        return {"nama": terbaik, "status": "sebagian", "keyakinan": p * 0.7,
                "rincian": f"peringkat-1 beda ({n_ki} vs {n_ka}); "
                           f"titik temu: {terbaik}",
                "jarak_kiri": jk, "jarak_kanan": ja}

    return {"nama": None, "status": "bertentangan", "keyakinan": 0.0,
            "rincian": f"kiri bilang {n_ki} ({p_ki:.0%}), "
                       f"kanan bilang {n_ka} ({p_ka:.0%}) — perlu diperiksa",
            "jarak_kiri": j_ki, "jarak_kanan": j_ka}


def peringkat_dari_galeri(vektor, galeri_sisi):
    """Peringkat (nama, jarak) terhadap satu sisi galeri."""
    if not galeri_sisi:
        return []
    out = []
    for nama, vecs in galeri_sisi.items():
        if not vecs:
            continue
        best = max(float(np.dot(vektor, v)) for v in vecs)
        out.append((nama, max(0.0, min(1.0, (1.0 - best) / 2.0))))
    out.sort(key=lambda x: x[1])
    return out
