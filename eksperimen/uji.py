"""
Tes invarian protokol §3. Jalankan sebelum percaya angka mana pun.

    python3 uji.py            # semua
    python3 uji.py -v         # + detail tiap tes

Kenapa ini ada: validasi manual-vs-wildlife-tools hanya membuktikan dua
implementasi sepakat. Kalau keduanya salah dengan cara yang sama, ia lolos.
Tes di bawah menguji hal yang berbeda — invarian yang, kalau rusak, akan
membuat seluruh perbandingan batal TANPA memunculkan error:

  - split berubah antar pemanggilan (tidak deterministik)
  - query kiri diam-diam dicocokkan ke gallery kanan
  - gallery ternyata lebih baru dari query
  - embedding tidak ternormalisasi
  - metrik salah hitung (diuji dengan kasus buatan yang jawabannya diketahui)
  - preprocessing tidak deterministik atau merusak dtype
  - bobot model termuat sebagian tanpa error

Sengaja tanpa pytest supaya bisa jalan di `.venv` apa adanya.
"""

import sys
import time
import traceback

import numpy as np

import protokol as P

VERBOSE = "-v" in sys.argv
_daftar = []


def uji(nama):
    def bungkus(fn):
        _daftar.append((nama, fn))
        return fn
    return bungkus


def sama(a, b, pesan=""):
    assert a == b, f"{pesan} — dapat {a!r}, harusnya {b!r}"


# ------------------------------------------------------------ katalog
@uji("katalog: semua berkas ada di disk")
def _():
    kat = P.baca_katalog()          # baca_katalog sendiri yang melempar
    assert len(kat) > 0, "katalog kosong"
    return f"{len(kat)} foto"


@uji("katalog: ReunionTurtles sesuai angka di spesifikasi §2")
def _():
    if P.DATASET != "reunion":
        return "dilewati (DATASET != reunion)"
    kat = P.baca_katalog()
    sama(len(kat), 336, "jumlah foto")
    sama(len({r["identity"] for r in kat}), 84, "jumlah individu")
    spesies = {}
    for r in kat:
        spesies.setdefault(r["species"], set()).add(r["identity"])
    sama(len(spesies["Green"]), 50, "individu hijau")
    sama(len(spesies["Hawksbill"]), 34, "individu sisik")
    sama(sum(r["position"] == "left" for r in kat), 168, "foto kiri")
    sama(sum(r["position"] == "right" for r in kat), 168, "foto kanan")
    return "84 individu, 50 hijau + 34 sisik, 168 L + 168 R"


# -------------------------------------------------------------- split
@uji("split: deterministik — dua pemanggilan menghasilkan hal yang sama persis")
def _():
    kat = P.baca_katalog()
    g1, q1 = P.bangun_split(kat)
    g2, q2 = P.bangun_split(list(reversed(kat)))   # urutan input diacak
    sama([r["path"] for r in g1], [r["path"] for r in g2], "gallery")
    sama([r["path"] for r in q1], [r["path"] for r in q2], "query")
    return f"{len(g1)} gallery / {len(q1)} query, stabil terhadap urutan input"


@uji("split: gallery selalu LEBIH LAMA dari query untuk pasangan yang sama")
def _():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    tahun_g = {}
    for r in gal:
        k = (r["identity"], r["side"])
        tahun_g[k] = max(tahun_g.get(k, -9999), r["year"])
    salah = [r for r in qry if r["year"] <= tahun_g.get((r["identity"], r["side"]), -9999)]
    sama(len(salah), 0, "query yang tahunnya tidak lebih baru dari gallery")
    return "nol pembalikan tahun"


@uji("split: tiap query punya pasangan di gallery pada SISI yang sama")
def _():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    kunci = {(r["identity"], r["side"]) for r in gal}
    yatim = [r for r in qry if (r["identity"], r["side"]) not in kunci]
    sama(len(yatim), 0, "query yatim")
    return "nol query yatim"


@uji("split: gallery dan query tidak pernah berbagi berkas yang sama")
def _():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    irisan = {r["path"] for r in gal} & {r["path"] for r in qry}
    sama(len(irisan), 0, "berkas yang muncul di gallery DAN query")
    return "nol tumpang tindih"


@uji("split: hanya sisi kiri/kanan yang masuk, tidak ada 'top'/'front'")
def _():
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    sisi = {r["side"] for r in gal} | {r["side"] for r in qry}
    sama(sisi, set(P.SISI), "himpunan sisi")
    return f"sisi = {sorted(sisi)}"


# ------------------------------------------------------------- metrik
@uji("metrik: kasus buatan dengan jawaban yang sudah diketahui")
def _():
    """Tiga query, tiga gallery, satu sisi. Similarity disusun tangan.

        q0 -> g0 benar di rank-1
        q1 -> g1 benar di rank-2   (rank5 kena, rank1 tidak)
        q2 -> tidak ada yang benar
    """
    id_g = np.array(["A", "B", "C"])
    id_q = np.array(["A", "B", "D"])
    s = np.array(["left"] * 3)
    # dibuat lewat embedding supaya jalur kodenya identik dengan yang asli
    Eg = np.eye(3, dtype=np.float32)
    Eq = np.array([[0.9, 0.1, 0.0],      # paling dekat g0 = A -> benar rank-1
                   [0.9, 0.8, 0.0],      # g0 dulu, lalu g1 = B -> benar rank-2
                   [0.5, 0.4, 0.3]],     # D tidak ada di gallery
                  dtype=np.float32)
    Eq /= np.linalg.norm(Eq, axis=1, keepdims=True)
    h = P.evaluasi_manual(Eq, Eg, id_q, id_g, s, s)
    sama(h["rank1"].tolist(), [True, False, False], "rank1")
    sama(h["rank5"].tolist(), [True, True, False], "rank5")
    # AP: q0 -> 1/1 = 1.0 ; q1 -> 1/2 = 0.5 ; q2 -> 0
    assert np.allclose(h["ap"], [1.0, 0.5, 0.0]), f"AP salah: {h['ap']}"
    return "rank1/rank5/AP cocok dengan hitungan tangan"


@uji("metrik: AP benar saat SATU query punya BANYAK jawaban benar")
def _():
    """Kasus di atas hanya punya satu jawaban benar per query, sehingga
    AP = 1/rank dan rumus mAP yang salah pun lolos. Di ReunionTurtles memang
    hanya ada satu foto gallery per (individu, sisi) — tapi di
    SeaTurtleIDHeads bisa banyak, dan di situlah rumusnya menentukan.

    Susunan: gallery = [A, X, A, X]. Jawaban benar ada di peringkat 1 dan 3.
        presisi@1 = 1/1 = 1.000
        presisi@3 = 2/3 = 0.667
        AP        = (1.000 + 0.667) / 2 = 0.8333
    Rumus keliru "1 / rank_pertama" akan memberi 1.0 — jadi tes ini
    membedakannya.
    """
    id_g = np.array(["A", "X", "A", "X"])
    id_q = np.array(["A"])
    s_g = np.array(["left"] * 4)
    s_q = np.array(["left"])
    Eg = np.eye(4, dtype=np.float32)
    # similarity menurun: g0 > g1 > g2 > g3  -> benar di peringkat 1 dan 3
    Eq = np.array([[0.9, 0.8, 0.7, 0.6]], np.float32)
    Eq /= np.linalg.norm(Eq)
    h = P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g)
    harap = (1.0 / 1 + 2.0 / 3) / 2
    assert abs(float(h["ap"][0]) - harap) < 1e-9, \
        f"AP = {h['ap'][0]:.6f}, harusnya {harap:.6f} — rumus mAP salah"
    return f"AP multi-relevan = {harap:.4f} (bukan 1.0)"


@uji("metrik: kunci sisi benar-benar mencegah pencocokan silang")
def _():
    """Jawaban benar SENGAJA ditaruh di sisi berlawanan dengan similarity
    tertinggi. Kalau kunci sisi bocor, rank-1 akan jadi True."""
    id_g = np.array(["A", "B"])
    id_q = np.array(["A"])
    s_g = np.array(["right", "left"])       # A ada di KANAN
    s_q = np.array(["left"])                # query KIRI
    Eg = np.eye(2, dtype=np.float32)
    Eq = np.array([[1.0, 0.0]], np.float32)  # paling mirip A (sisi kanan)
    h = P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g)
    sama(h["rank1"].tolist(), [False], "rank1 harus False — A ada di sisi lain")
    sama(float(h["ap"][0]), 0.0, "AP")
    return "pasangan beda-sisi tidak pernah masuk peringkat"


@uji("metrik: kandidat beda-sisi tidak ikut mengisi top-5")
def _():
    """Kalau -inf hanya dipakai untuk mengurutkan tapi tidak disaring, entri
    beda-sisi akan menempati slot top-5 dan menggeser jawaban benar keluar."""
    id_g = np.array(["X"] * 5 + ["A"])
    id_q = np.array(["A"])
    s_g = np.array(["right"] * 5 + ["left"])
    s_q = np.array(["left"])
    Eg = np.zeros((6, 2), np.float32)
    Eg[:5, 0] = 1.0          # lima distraktor sisi kanan, similarity tinggi
    Eg[5, 1] = 1.0           # jawaban benar, sisi kiri
    Eq = np.array([[0.99, 0.14]], np.float32)
    Eq /= np.linalg.norm(Eq)
    h = P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g)
    sama(h["rank1"].tolist(), [True], "jawaban benar harus rank-1")
    return "distraktor beda-sisi tidak menempati slot peringkat"


@uji("metrik: foto individu yang sama di SISI SEBERANG tidak dihitung benar")
def _():
    """Jebakan nyata yang sempat lolos di `rerank.py`: tiap individu punya foto
    galeri di kedua sisi. Kalau mask `sah` hilang, foto sisi seberang terhitung
    sebagai jawaban benar KEDUA di peringkat jauh — rank-1 tetap terlihat benar
    sementara mAP anjlok diam-diam (37.40 -> 19.52 saat itu terjadi).

    Susunan: A ada di kiri (peringkat 1) dan di kanan (peringkat terakhir).
    Query kiri. AP harus 1.0, bukan rata-rata yang ikut menghitung entri kanan.
    """
    id_g = np.array(["A", "X", "A"])
    id_q = np.array(["A"])
    s_g = np.array(["left", "left", "right"])   # A muncul di kedua sisi
    s_q = np.array(["left"])
    Eg = np.eye(3, dtype=np.float32)
    Eq = np.array([[0.9, 0.4, 0.8]], np.float32)   # entri kanan skornya tinggi
    Eq /= np.linalg.norm(Eq)
    h = P.evaluasi_manual(Eq, Eg, id_q, id_g, s_q, s_g)
    sama(h["rank1"].tolist(), [True], "rank1")
    assert abs(float(h["ap"][0]) - 1.0) < 1e-9, \
        f"AP = {h['ap'][0]:.4f}, harusnya 1.0 — entri sisi seberang bocor ke mAP"
    return "entri sisi seberang tidak menambah 'jawaban benar'"


@uji("metrik: stage-1 lewat rerank.py identik dengan lewat evaluasi.py")
def _():
    """Dua modul menghitung angka yang sama; kalau berbeda, salah satunya
    memakai jalur metrik sendiri dan protokol §3 sudah bocor."""
    import rerank as R
    from evaluasi import evaluasi
    kat = P.baca_katalog()
    gal, qry = P.bangun_split(kat)
    try:
        a = evaluasi("raw", gal, qry)
    except (FileNotFoundError, RuntimeError) as e:
        return f"dilewati — embedding belum ada ({e})"
    S, _, _ = R.kandidat_stage1(gal, qry, 84)
    id_g = np.array([r["identity"] for r in gal])
    id_q = np.array([r["identity"] for r in qry])
    b = P.metrik_dari_matriks(S, id_q, id_g)
    for m in ("rank1", "rank5", "ap"):
        assert np.allclose(a[m], b[m]), f"{m} berbeda antar modul"
    return f"rank1 {a['rank1'].mean()*100:.2f}%  mAP {a['ap'].mean()*100:.2f}% di kedua jalur"


@uji("metrik: ringkas() konsisten dengan vektor mentahnya")
def _():
    h = {"rank1": np.array([1, 0, 1, 1], bool),
         "rank5": np.array([1, 1, 1, 1], bool),
         "ap": np.array([1.0, 0.0, 0.5, 1.0])}
    r = P.ringkas(h)
    sama(round(r["rank1"], 6), 75.0, "rank1")
    sama(round(r["mAP"], 6), 62.5, "mAP")
    sama(r["n"], 4, "n")
    return "persen dihitung benar"


# ----------------------------------------------------- preprocessing
@uji("preprocessing: deterministik, dtype dan bentuk terjaga")
def _():
    rng = np.random.default_rng(0)
    im = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    catatan = []
    for nama, fn in P.KONDISI.items():
        a, b = fn(im.copy()), fn(im.copy())
        assert a.dtype == np.uint8, f"{nama}: dtype jadi {a.dtype}"
        assert a.ndim == 3 and a.shape[2] == 3, f"{nama}: bentuk {a.shape}"
        assert np.array_equal(a, b), f"{nama}: tidak deterministik"
        catatan.append(f"{nama}{a.shape[:2]}")
    return " ".join(catatan)


@uji("preprocessing: raw benar-benar tidak mengubah apa pun")
def _():
    rng = np.random.default_rng(1)
    im = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    assert np.array_equal(P.KONDISI["raw"](im), im), "raw mengubah piksel"
    return "identitas"


@uji("preprocessing: grayscale membuat ketiga kanal identik")
def _():
    rng = np.random.default_rng(2)
    im = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    g = P.KONDISI["gray"](im)
    assert np.array_equal(g[:, :, 0], g[:, :, 1]) and \
           np.array_equal(g[:, :, 1], g[:, :, 2]), "kanal tidak sama"
    return "R = G = B"


@uji("preprocessing: crop benar-benar memperkecil, crop_wb = crop lalu wb")
def _():
    rng = np.random.default_rng(3)
    im = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)
    c = P.KONDISI["crop"](im)
    sama(c.shape[:2], (70, 140), "ukuran setelah crop 70%")
    assert np.array_equal(P.KONDISI["crop_wb"](im),
                          P.KONDISI["wb"](P.KONDISI["crop"](im))), \
        "crop_wb bukan komposisi crop lalu wb"
    return "crop 70% dan komposisinya benar"


# ---------------------------------------------------------- transform
@uji("transform: keluaran selalu (3, UKURAN, UKURAN) apa pun rasio aspeknya")
def _():
    rng = np.random.default_rng(4)
    for h, w in [(50, 300), (300, 50), (224, 224), (17, 19)]:
        im = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for nama, tf in (("squash", P.transform_squash), ("cfg", P.transform_cfg)):
            x = tf(im)
            sama(x.shape, (3, P.UKURAN, P.UKURAN), f"{nama} pada {h}x{w}")
    return f"({P.UKURAN}, {P.UKURAN}) untuk semua rasio, dua transform"


@uji("transform: normalisasi memakai mean/std dari config, bukan angka ketikan")
def _():
    x = P.transform_squash(np.full((30, 30, 3), 255, np.uint8))
    harap = (1.0 - P.MEAN) / P.STD
    assert np.allclose(x[:, 0, 0], harap, atol=1e-5), \
        f"putih -> {x[:, 0, 0]}, harusnya {harap}"
    return f"mean={P.MEAN.round(3).tolist()} std={P.STD.round(3).tolist()}"


# -------------------------------------------------------------- model
@uji("model: bobot termuat penuh, embedding ternormalisasi")
def _():
    model, cfg = P.muat_model()      # muat_model melempar kalau ada key hilang
    kat = P.baca_katalog()
    paths = [r["path"] for r in kat[:4]]
    t0 = time.time()
    E = P.embed(paths, "raw", model, batch=4)
    dt = time.time() - t0
    sama(E.shape, (4, P.DIM), "bentuk embedding")
    n = np.linalg.norm(E, axis=1)
    assert np.allclose(n, 1.0, atol=1e-5), f"norma bukan 1: {n}"
    return f"{cfg['architecture']} dim={P.DIM} norma=1.0 ({dt / 4:.2f} s/foto)"


@uji("model: embedding stabil — dua pemanggilan memberi hasil identik")
def _():
    model, _ = P.muat_model()
    kat = P.baca_katalog()
    paths = [r["path"] for r in kat[:2]]
    a = P.embed(paths, "raw", model, batch=2)
    b = P.embed(paths, "raw", model, batch=1)   # batch berbeda, hasil sama
    assert np.allclose(a, b, atol=1e-5), \
        f"selisih maks {np.abs(a - b).max():.2e} — hasil bergantung batch"
    return "tidak bergantung ukuran batch"


# ------------------------------------------------------------ statistik
@uji("statistik: McNemar simetris dan menangani kasus tanpa perbedaan")
def _():
    from statistik import mcnemar
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    r = mcnemar(a, b)
    sama(r["n01_baseline_salah_kondisi_benar"], 1, "n01")
    sama(r["n10_baseline_benar_kondisi_salah"], 1, "n10")
    sama(round(r["p_value"], 6), 1.0, "p saat n01 == n10")
    sama(mcnemar(a, a)["p_value"], 1.0, "p saat tidak ada perbedaan sama sekali")
    return "n01/n10 benar, p=1.0 saat seimbang"


@uji("statistik: bootstrap CI melingkupi selisih nyata dan berpasangan")
def _():
    from statistik import bootstrap_delta
    rng = np.random.default_rng(7)
    x = rng.normal(50, 10, 400)
    y = x + 5.0                    # selisih konstan: CI harus rapat di +5
    r = bootstrap_delta(x, y, B=2000)
    assert abs(r["delta"] - 5.0) < 1e-9, r["delta"]
    assert r["ci95"][0] > 4.9 and r["ci95"][1] < 5.1, \
        f"CI {r['ci95']} terlalu lebar — resampling mungkin tidak berpasangan"
    assert r["signifikan"], "harusnya signifikan"
    r0 = bootstrap_delta(x, x.copy(), B=2000)
    assert not r0["signifikan"], "selisih nol tidak boleh signifikan"
    return f"delta konstan -> CI {[round(v, 3) for v in r['ci95']]}"


@uji("matcher: semua kelas memenuhi kontrak yang sama")
def t_kontrak_matcher():
    """Menangkap kelas bug yang mematikan aplikasi saat RoMa dipilih.

    Aplikasi dulu menebak jenis matcher lewat `hasattr(mm, "X")` lalu jatuh
    ke `mm.det.detectAndCompute`. RoMa tidak punya `.det`, jadi memilihnya di
    UI langsung melempar AttributeError. Tes ini bersifat STATIS — memeriksa
    kelasnya, bukan instansnya — supaya tetap berjalan di lingkungan yang
    bobot RoMa dan ALIKED-nya tidak bisa dipasang sama sekali.
    """
    import rerank as R
    wajib = ("ekstrak", "ekstrak_array", "korespondensi", "skor")
    atribut = ("KOORD_ASLI", "PUNYA_KEYPOINT")
    kelas = [R.Klasik, R.XFeat, R.ALIKED, R.VisMatch, R.RoMa]
    for K in kelas:
        for m in wajib:
            assert callable(getattr(K, m, None)), f"{K.__name__} tidak punya .{m}()"
        for a in atribut:
            assert isinstance(getattr(K, a, None), bool), \
                f"{K.__name__} tidak menyatakan {a}"
    return f"{len(kelas)} kelas x {len(wajib) + len(atribut)} anggota kontrak"


@uji("matcher: ekstrak dari array == ekstrak dari berkas")
def t_ekstrak_array():
    """Jalur kamera dan jalur eksperimen harus memberi fitur yang sama.

    Kalau berbeda, angka di layar tidak lagi bisa dibandingkan dengan angka
    di laporan — dan tidak ada error apa pun yang muncul.
    """
    import cv2
    import rerank as R
    kat = P.baca_katalog()
    gal, _ = P.bangun_split(kat)
    path = gal[0]["path"]
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    diuji = []
    for nama in ("sift", "xfeat"):
        try:
            m = R.buat_matcher(nama)
        except Exception:
            continue                      # bobot tidak ada di mesin ini
        a, b = m.ekstrak(path), m.ekstrak_array(rgb)
        assert a is not None and b is not None, f"{nama}: fitur kosong"
        assert len(a[0]) == len(b[0]), \
            f"{nama}: jumlah keypoint beda ({len(a[0])} vs {len(b[0])})"
        assert np.allclose(a[0], b[0], atol=1e-3), f"{nama}: koordinat beda"
        # skor self-match harus tinggi lewat kontrak yang baru
        s = m.skor(b, b)
        assert s >= 100, f"{nama}: self-match cuma {s} inlier"
        diuji.append(f"{nama} {len(a[0])}kp self={s:.0f}")
    assert diuji, "tidak ada matcher yang bisa diuji di mesin ini"
    return " | ".join(diuji)


# ----------------------------------------------------------------- main
def main():
    lolos = gagal = 0
    for nama, fn in _daftar:
        try:
            catatan = fn()
            lolos += 1
            print(f"  \033[32mOK\033[0m   {nama}")
            if VERBOSE and catatan:
                print(f"       {catatan}")
        except Exception as e:
            gagal += 1
            print(f"  \033[31mGAGAL\033[0m {nama}")
            print(f"       {type(e).__name__}: {e}")
            if VERBOSE:
                traceback.print_exc()

    print(f"\n{lolos} lolos, {gagal} gagal "
          f"[{P.DATASET} / {P.MODEL} / {P.TRANSFORM}]")
    return 1 if gagal else 0


if __name__ == "__main__":
    raise SystemExit(main())
