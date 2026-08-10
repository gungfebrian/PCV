"""
FACE ID PENYU — daftarkan seekor penyu, lalu kenali dia lagi nanti.

Cara kerjanya persis Face ID di ponsel, cuma objeknya penyu:

    ENROLL   simpan beberapa vektor "sidik jari" untuk satu individu
    VERIFY   foto baru -> vektor -> cari individu terdekat di galeri
    AMBANG   kalau jarak terdekat masih terlalu jauh -> "TIDAK DIKENAL"

Bagian AMBANG itu yang paling sering dilupakan. Tanpa ambang, sistem akan
selalu menjawab dengan nama seseorang, bahkan untuk penyu yang belum pernah
didaftarkan sama sekali. Sistem re-ID yang jujur harus bisa bilang "saya
tidak tahu".

Data tersimpan di penyu_terdaftar.npz — vektor saja, bukan foto.
"""

import os
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "penyu_terdaftar.npz")

# Jarak cosine maksimum yang masih dianggap individu yang sama.
#
# 0.42 adalah hasil kalibrasi_ambang() pada 20 individu TurtleID2022 memakai
# deskriptor_baseline. TAPI akurasi seimbang di ambang itu cuma 51.5% —
# sedangkan tebak acak 50%. Artinya: dengan deskriptor baseline TIDAK ADA
# ambang yang benar-benar memisahkan penyu sama dari penyu beda.
#
# Jadi angka ini hanya placeholder supaya alurnya bisa diuji. Face ID penyu
# baru betul-betul berfungsi setelah deskriptornya diganti MegaDescriptor,
# dan sesudah itu WAJIB dikalibrasi ulang — ambang satu deskriptor tidak
# pernah berlaku untuk deskriptor lain.
AMBANG_DEFAULT = 0.42


class Galeri:
    """Kumpulan individu terdaftar beserta vektor-vektornya."""

    def __init__(self, path=DB):
        self.path = path
        self.data = {}        # nama -> list[np.ndarray]
        self.dibuat = {}      # nama -> waktu pendaftaran
        self.muat()

    # ------------------------------------------------------------ simpanan
    def muat(self):
        """Baca galeri. Sengaja TANPA allow_pickle: kalau file ini dimuat
        dengan pickle, file yang dirusak bisa menjalankan kode sembarang saat
        dibuka. Semua disimpan sebagai array biasa, jadi tidak perlu pickle."""
        if not os.path.exists(self.path):
            return
        try:
            z = np.load(self.path)                       # allow_pickle=False (default)
            nama = z["nama"].astype(str)
            vektor = z["vektor"].astype(np.float32)
            self.data = {}
            for n, v in zip(nama, vektor):
                self.data.setdefault(str(n), []).append(v)
            self.dibuat = dict(zip(z["cap_nama"].astype(str),
                                   z["cap_waktu"].astype(str)))
        except (OSError, KeyError, ValueError) as e:
            # File rusak jangan bikin aplikasi mati; mulai dari galeri kosong.
            print(f"[faceid] gagal membaca {self.path}: {e}")
            self.data, self.dibuat = {}, {}

    def simpan(self):
        nama, vektor = [], []
        for n, vs in self.data.items():
            for v in vs:
                nama.append(n)
                vektor.append(v)
        np.savez(
            self.path,
            nama=np.array(nama, dtype=np.str_),
            vektor=(np.stack(vektor) if vektor
                    else np.zeros((0, 1), dtype=np.float32)),
            cap_nama=np.array(list(self.dibuat.keys()), dtype=np.str_),
            cap_waktu=np.array(list(self.dibuat.values()), dtype=np.str_))

    # -------------------------------------------------------------- enroll
    def daftarkan(self, nama, vektor):
        """Tambahkan satu contoh untuk individu. Boleh dipanggil berkali-kali;
        makin banyak sudut pandang yang disimpan, makin tahan pengenalannya."""
        nama = nama.strip()
        if not nama:
            return False, "Nama tidak boleh kosong."
        vektor = np.asarray(vektor, dtype=np.float32)
        # Menolak mencampur dimensi: satu galeri harus satu deskriptor.
        if self.data:
            dim_lama = next(iter(self.data.values()))[0].shape
            if vektor.shape != dim_lama:
                return False, (f"Galeri ini dibuat dengan deskriptor lain "
                               f"({dim_lama[0]}-dim, sekarang {vektor.shape[0]}-dim). "
                               f"Kosongkan galeri dulu sebelum mendaftar ulang.")
        if nama not in self.data:
            self.data[nama] = []
            self.dibuat[nama] = time.strftime("%Y-%m-%d %H:%M")
        self.data[nama].append(np.asarray(vektor, dtype=np.float32))
        self.simpan()
        return True, f"'{nama}' tersimpan ({len(self.data[nama])} contoh)."

    def hapus(self, nama):
        if nama in self.data:
            del self.data[nama]
            self.dibuat.pop(nama, None)
            self.simpan()
            return True
        return False

    def kosongkan(self):
        self.data, self.dibuat = {}, {}
        self.simpan()

    # -------------------------------------------------------------- verify
    def kenali(self, vektor, ambang=AMBANG_DEFAULT, min_margin=0.0):
        """Cari individu terdekat.

        Return dict berisi nama, jarak, status, dan peringkat. Status:
          'dikenal'       jarak <= ambang
          'tidak dikenal' ada yang terdaftar tapi semuanya terlalu jauh
          'kosong'        belum ada yang didaftarkan
        """
        if not self.data:
            return {"nama": None, "jarak": 1.0, "status": "kosong",
                    "peringkat": [], "ambang": ambang}

        v = np.asarray(vektor, dtype=np.float32)

        # Galeri bisa berisi vektor dari deskriptor lain (mis. didaftarkan
        # memakai MegaDescriptor lalu deskriptor diganti ke baseline). Dimensinya
        # beda, jadi cosine-nya mustahil dihitung. Ini dilaporkan, bukan
        # dibiarkan meledak di tengah loop video.
        cocok = {n: [w for w in vs if w.shape == v.shape]
                 for n, vs in self.data.items()}
        cocok = {n: vs for n, vs in cocok.items() if vs}
        if not cocok:
            dim_lama = next(iter(self.data.values()))[0].shape[0]
            return {"nama": None, "jarak": 1.0, "status": "beda deskriptor",
                    "peringkat": [], "ambang": ambang,
                    "dim_galeri": dim_lama, "dim_query": v.shape[0]}

        peringkat = []
        for nama, vecs in cocok.items():
            # Ambil contoh terbaik individu ini, bukan rata-ratanya: penyu yang
            # sama dari sudut berbeda bisa sangat berbeda, dan merata-ratakan
            # justru mengaburkan cirinya.
            best = max(float(np.dot(v, w)) for w in vecs)
            peringkat.append((nama, max(0.0, min(1.0, (1.0 - best) / 2.0))))
        peringkat.sort(key=lambda x: x[1])

        nama, jarak = peringkat[0]

        # Uji margin: kandidat teratas harus unggul JELAS dari kandidat kedua.
        # Tanpa ini, dua orang yang sama-sama berjarak ~0.60 akan membuat
        # sistem memilih salah satu secara sewenang-wenang padahal ia
        # sebenarnya ragu. Menolak lebih baik daripada menebak.
        margin = (peringkat[1][1] - jarak) if len(peringkat) > 1 else 1.0
        cukup = jarak <= ambang and margin >= min_margin

        if jarak > ambang:
            status = "tidak dikenal"
        elif not cukup:
            status = "ragu"
        else:
            status = "dikenal"

        return {"nama": nama if cukup else None,
                "jarak": jarak,
                "margin": margin,
                "min_margin": min_margin,
                "kandidat": nama,
                "status": status,
                "peringkat": peringkat[:8],
                "ambang": ambang}

    # -------------------------------------------------------------- ringkas
    @property
    def jumlah_individu(self):
        return len(self.data)

    @property
    def jumlah_contoh(self):
        return sum(len(v) for v in self.data.values())

    def ringkasan(self):
        if not self.data:
            return "Belum ada penyu terdaftar."
        baris = [f"{n:<14} {len(v)} contoh   {self.dibuat.get(n,'')}"
                 for n, v in sorted(self.data.items())]
        return "\n".join(baris)


def kalibrasi_ambang(galeri_uji, deskriptor_fn=None):
    """Cari ambang yang memisahkan pasangan sama vs beda paling baik.

    Dipakai untuk memilih AMBANG_DEFAULT secara berbasis data, bukan tebakan.
    galeri_uji: dict nama -> list vektor.

    Return (ambang_terbaik, akurasi) dari pemindaian sederhana.
    """
    sama, beda = [], []
    nama_list = list(galeri_uji)
    for i, n in enumerate(nama_list):
        vs = galeri_uji[n]
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                sama.append((1.0 - float(np.dot(vs[a], vs[b]))) / 2.0)
        for m in nama_list[i + 1:]:
            for va in vs:
                for vb in galeri_uji[m]:
                    beda.append((1.0 - float(np.dot(va, vb))) / 2.0)
    if not sama or not beda:
        return AMBANG_DEFAULT, 0.0

    sama, beda = np.array(sama), np.array(beda)
    kandidat = np.linspace(0.0, 1.0, 201)
    # Akurasi seimbang: rata-rata dari benar-terima dan benar-tolak, supaya
    # tidak bias walaupun pasangan 'beda' jauh lebih banyak.
    skor = [((sama <= t).mean() + (beda > t).mean()) / 2 for t in kandidat]
    i = int(np.argmax(skor))
    return float(kandidat[i]), float(skor[i])
