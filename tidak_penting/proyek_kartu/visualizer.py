"""
Visualizer pipeline CV — REALTIME. Pencet tombol tahap, lihat videonya jadi apa.

Kamera jalan terus; tombol Gray / Blur / Edges / Kontur / dst. menentukan
tahap mana yang ditampilkan besar. Slider parameter berubah langsung terlihat
efeknya di video.

Jalankan:
    .venv/bin/python visualizer.py
"""

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    # Penyebab paling sering: dijalankan dengan Python yang salah. Paket-paket
    # ini ada di .venv/, bukan di Python global.
    import sys
    print(f"\n  GAGAL IMPORT: {e}\n")
    print(f"  Python yang dipakai: {sys.executable}\n")
    print("  Paket PCV ada di .venv/, bukan di Python global.")
    print("  Jalankan dengan perintah ini:\n")
    print("      .venv/bin/python visualizer.py\n")
    raise SystemExit(1)

import face_mode
import faceid_penyu
import pipeline
import turtle_mode

BG = "#1e1e2e"
PANEL = "#282838"
BTN = "#3a3a4e"
FG = "#e8e8f0"
MUTED = "#9a9ab0"
ACCENT = "#7aa2f7"
OK = "#9ece6a"
WARN = "#e0af68"
BAD = "#f7768e"

# Label pendek untuk tombol, dipetakan dari Stage.key. Nama panjang tidak muat.
LABEL = {
    "asli": "Frame Asli", "resize": "Perkecil", "gray": "Gray", "blur": "Blur",
    "edges": "Edges", "contours": "Kontur", "quads": "Segi-4", "titik": "Pojok",
    "warp": "Warp", "wgray": "Card Gray", "desc": "Sidik Jari",
    "banding": "Template", "selisih": "Selisih SAD", "gagal": "Gagal",
    # tahap khusus mode penyu
    "clahe": "CLAHE", "roi": "Wilayah Objek", "patch": "Patch Baku",
    "vektor": "Vektor",
    "titik_pola": "Titik Pola", "sisik": "Pola Sisik",
    # tahap khusus mode muka
    "align": "Wajah Lurus",
}
PER_BARIS = 7


class Klik(tk.Label):
    """Tombol buatan sendiri.

    tk.Button di macOS mengabaikan opsi bg/fg — tombolnya selalu putih bawaan
    sistem, jadi teks terang di atasnya tidak terbaca. tk.Label menghormati
    warna, jadi dipakai sebagai tombol dengan binding klik.
    """

    def __init__(self, parent, text, command, warna=BTN, **kw):
        super().__init__(parent, text=text, bg=warna, fg=FG, padx=12, pady=6,
                         cursor="pointinghand", font=("Helvetica", 11), **kw)
        self.command = command
        self.dasar = warna          # warna istirahat, boleh beda per tombol
        self._aktif = False
        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", self._masuk)
        self.bind("<Leave>", self._keluar)

    def set_aktif(self, aktif):
        self._aktif = aktif
        self.config(bg=ACCENT if aktif else self.dasar,
                    fg="#11111b" if aktif else FG)

    def _terang(self):
        """Versi lebih terang dari warna istirahat, untuk efek hover."""
        r, g, b = (int(self.dasar[i:i + 2], 16) for i in (1, 3, 5))
        return f"#{min(255, r+22):02x}{min(255, g+22):02x}{min(255, b+22):02x}"

    def _masuk(self, _):
        if not self._aktif:
            self.config(bg=self._terang())

    def _keluar(self, _):
        if not self._aktif:
            self.config(bg=self.dasar)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Visualizer Pipeline CV — Realtime")
        root.configure(bg=BG)
        root.geometry("1340x880")

        self.params = pipeline.Params()
        self.templates, hilang = pipeline.muat_template(params=self.params)

        # Mode penyu dimuat malas: membaca 1700+ foto butuh waktu, dan
        # pengguna mungkin tidak pernah membukanya.
        self.mode = "kartu"
        self.deskriptor = "baseline"
        self.banding_hasil = None
        self.banding_gambar = None
        self.banding_label = None
        self._seed_banding = 0
        self.banding_peringkat = None
        self.dataset_penyu = "utuh"
        # Galeri terpisah: embedding penyu dan embedding wajah beda ruang
        # vektor sama sekali, tidak boleh dicampur dalam satu galeri.
        self.faceid = faceid_penyu.Galeri()
        self.faceid_muka = faceid_penyu.Galeri(
            path=os.path.join(pipeline.BASE_DIR, "muka_terdaftar.npz"))
        self.galeri_penyu = None
        self.contoh_penyu = []
        self.idx_contoh = 0

        # Sumber kamera: ("mac", indeks) atau ("ip", url).
        self.url_ip = os.environ.get("CAM_URL", "http://10.64.53.105:4747/video")
        self.sumber = ("mac", 0)

        self.cap = None
        self.live = False
        self.stages, self.hasil = [], None
        self.pilih_key = "asli"      # simpan key, bukan indeks: jumlah tahap berubah
        self.tombol = {}
        self._photo = None
        self._after_id = None
        self._t_last = time.perf_counter()
        self._fps = 0.0

        self.frame = self._gambar_awal()
        self._bangun_ui()
        self.btn_mode["kartu"].set_aktif(True)
        # Mode awal adalah kartu, dan set_mode() keluar lebih awal kalau
        # modenya tidak berubah — jadi keadaan awal diatur di sini.
        self.descbar.pack_forget()

        if hilang:
            self._status(f"Template kurang {len(hilang)}: {', '.join(hilang)}", WARN)
        self.proses()
        self.mulai_live()            # langsung coba nyalakan kamera

    # --------------------------------------------------------------- sumber
    def _gambar_awal(self):
        for folder in ("Templatekartu", "individual_cards_2"):
            d = os.path.join(pipeline.BASE_DIR, folder)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(d, f))
                    if img is not None:
                        return self._beri_latar(img)
        return np.full((600, 800, 3), 40, np.uint8)

    def _beri_latar(self, kartu, pad=90):
        """File template sudah terpotong pas di tepi kartu, jadi tidak punya
        kontur luar untuk dideteksi. Tempel di atas latar polos supaya ada."""
        kartu = cv2.resize(kartu, (pipeline.CARD_W, pipeline.CARD_H))
        h, w = kartu.shape[:2]
        kanvas = np.full((h + pad * 2, w + pad * 2, 3), 60, np.uint8)
        kanvas[pad:pad + h, pad:pad + w] = kartu
        return kanvas

    # ---------------------------------------------------------- sumber kamera
    def _nama_sumber(self):
        jenis, nilai = self.sumber
        return f"Webcam Mac #{nilai}" if jenis == "mac" else f"Kamera IP {nilai}"

    def set_sumber(self, jenis, nilai):
        """Ganti sumber kamera. Kalau sedang live, langsung disambung ulang."""
        if jenis == "ip" and not nilai:
            sedang = self.live
            self.henti_live()
            nilai = simpledialog.askstring(
                "Kamera IP / HTTP",
                "URL stream (mis. DroidCam / IP Webcam):",
                initialvalue=self.url_ip, parent=self.root)
            if not nilai:
                self._status("Dibatalkan.", MUTED)
                if sedang:
                    self.mulai_live()
                return
            self.url_ip = nilai

        self.sumber = (jenis, nilai)
        for k, b in self.btn_kamera.items():
            b.set_aktif(k == (jenis if jenis == "ip" else f"mac{nilai}"))

        self.henti_live()
        self.mulai_live()
        if not self.live:
            return
        self._status(f"Sumber kamera: {self._nama_sumber()}", OK)

    @staticmethod
    def _ip_hidup(url, batas=0.6):
        """Cek cepat apakah host:port kamera IP menerima koneksi."""
        import socket
        from urllib.parse import urlparse
        u = urlparse(url)
        if not u.hostname:
            return False
        try:
            with socket.create_connection((u.hostname, u.port or 80), batas):
                return True
        except OSError:
            return False

    def mulai_live(self):
        if self.live:
            return
        self._status("Membuka kamera...", MUTED)
        self.root.update_idletasks()


        jenis, nilai = self.sumber
        cap = None

        if jenis == "ip":
            # Cek soket berbatas waktu dulu: VideoCapture pada URL tak
            # terjangkau menggantung ~30 detik dan UI ikut membeku.
            if not self._ip_hidup(nilai):
                self._status(f"Kamera IP tidak menjawab di {nilai}. Pastikan "
                             "HP dan Mac satu WiFi, dan aplikasi kameranya "
                             "sedang jalan.", BAD)
                self.btn_live.config(text="▶  Live Kamera")
                return
            cap = cv2.VideoCapture(nilai)
        else:
            cap = cv2.VideoCapture(nilai)

        if cap is None or not cap.isOpened():
            if cap:
                cap.release()
            self._status(
                f"Tidak bisa membuka {self._nama_sumber()}. "
                + ("Coba indeks kamera lain, atau beri izin di System Settings "
                   "> Privacy & Security > Camera untuk terminal Anda."
                   if jenis == "mac" else "Periksa URL-nya."), BAD)
            self.btn_live.config(text="▶  Live Kamera")
            return
        # Kamera macOS mengirim beberapa frame hitam saat baru dibuka; kalau
        # langsung dipakai, tahap-tahap awal terlihat kosong dan membingungkan.
        for _ in range(10):
            cap.read()
        self.cap = cap
        self.live = True
        self.btn_live.config(text="⏸  Jeda")
        self._status("Live. Pencet tahap mana pun untuk lihat versi realtime-nya.", OK)
        self._tick()

    def henti_live(self):
        self.live = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_live.config(text="▶  Live Kamera")
        self._status("Dijeda. Frame terakhir dibekukan — slider tetap bisa dipakai.",
                     MUTED)

    def toggle_live(self):
        self.henti_live() if self.live else self.mulai_live()

    def _tick(self):
        """Loop video. Dijadwalkan lewat after() supaya UI tetap responsif."""
        if not self.live or self.cap is None:
            return
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            now = time.perf_counter()
            dt = now - self._t_last
            self._t_last = now
            if dt > 0:                      # rata-rata bergerak, biar tidak loncat
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self.proses()
        self._after_id = self.root.after(15, self._tick)

    def buka_file(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp"), ("Semua", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._status("Gagal membaca file itu.", BAD)
            return
        self.henti_live()
        self.frame = self._beri_latar(img) if max(img.shape[:2]) < 500 else img
        self._status(f"Dimuat: {os.path.basename(path)}", OK)
        self.proses()

    # ------------------------------------------------------- dataset penyu
    DATASET_PENYU = {
        "utuh": (turtle_mode.GALERI_DEFAULT, None),
        "kepala": (os.path.join(pipeline.BASE_DIR,
                                "dataset_penyu", "SeaTurtleIDHeads", "images"),
                   60),        # 400 individu; 60 cukup untuk demo & tetap cepat
    }

    def set_dataset(self, kunci):
        """Ganti sumber galeri penyu. Statistik kalibrasi ikut diganti —
        ambang & probabilitas dari dataset lain tidak berlaku (bagian 19)."""
        folder, maks = self.DATASET_PENYU[kunci]
        if not os.path.isdir(folder):
            self._status(f"Dataset belum ada di {folder}. Jalankan "
                         ".venv/bin/python unduh_dataset_penyu.py dulu.", BAD)
            return
        self.dataset_penyu = kunci
        turtle_mode.set_stats(kunci)
        for k, b in self.btn_ds.items():
            b.set_aktif(k == kunci)
        self.galeri_penyu = None          # paksa muat ulang
        if self.mode == "penyu":
            self.mode = None              # lolos dari guard "mode tidak berubah"
            self.set_mode("penyu")
        else:
            self._status(f"Dataset penyu: {kunci}. Berlaku saat mode Penyu dibuka.", OK)

    def kelola_tersimpan(self):
        """Jendela kecil untuk melihat & menghapus individu tersimpan."""
        g = self._galeri_aktif()
        win = tk.Toplevel(self.root)
        win.title("Kelola Tersimpan")
        win.configure(bg=PANEL)
        win.geometry("360x420")

        subjek = "muka" if self.mode == "muka" else "penyu"
        tk.Label(win, text=f"TERSIMPAN ({subjek})", bg=PANEL, fg=MUTED,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=14,
                                                      pady=(12, 6))
        lb = tk.Listbox(win, bg="#15151f", fg=FG, font=("Courier", 12),
                        relief="flat", selectbackground=ACCENT,
                        highlightthickness=0, activestyle="none")
        lb.pack(fill="both", expand=True, padx=14)

        def segarkan():
            lb.delete(0, tk.END)
            for nama in sorted(g.data):
                lb.insert(tk.END,
                          f"{nama:<16} {len(g.data[nama])} contoh  "
                          f"{g.dibuat.get(nama, '')}")
            if not g.data:
                lb.insert(tk.END, "  (kosong)")

        def hapus():
            pilih = lb.curselection()
            if not pilih or not g.data:
                return
            nama = sorted(g.data)[pilih[0]]
            if messagebox.askyesno("Hapus", f"Hapus '{nama}'?", parent=win):
                g.hapus(nama)
                segarkan()
                self.proses()

        def kosongkan():
            if g.data and messagebox.askyesno(
                    "Kosongkan", f"Hapus SEMUA {len(g.data)} individu?",
                    parent=win):
                g.kosongkan()
                segarkan()
                self.proses()

        baris = tk.Frame(win, bg=PANEL)
        baris.pack(fill="x", padx=14, pady=10)
        Klik(baris, "Hapus", hapus, warna="#4a2d2d").pack(side="left")
        Klik(baris, "Kosongkan Semua", kosongkan,
             warna="#3a2020").pack(side="left", padx=8)
        Klik(baris, "Tutup", win.destroy).pack(side="right")
        segarkan()

    # ------------------------------------------------------------- banding
    def ambil_pasangan(self, sama=True):
        """Ambil sepasang foto dari dataset dan bandingkan."""
        import banding
        pas = banding.pasangan_dataset(sama=sama, seed=self._seed_banding)
        self._seed_banding += 1
        if pas is None:
            self._status("Dataset penyu tidak ditemukan.", BAD)
            return
        pa, pb, la, lb, kebenaran = pas
        self._banding_dari(pa, pb, la, lb, kebenaran)

    def banding_kamera(self):
        """Ambil satu frame kamera, bandingkan ke seluruh penyu tersimpan."""
        cap = cv2.VideoCapture(self.sumber[1] if self.sumber[0] == "mac"
                               else self.sumber[1])
        if not cap.isOpened():
            cap.release()
            self._status("Kamera tidak bisa dibuka. Pakai 'Foto vs Yang "
                         "Tersimpan' untuk memilih file.", BAD)
            return
        for _ in range(10):
            ok, f = cap.read()
        cap.release()
        if not ok:
            self._status("Kamera tidak memberi frame.", BAD)
            return
        self._banding_ke_tersimpan(f, "dari kamera")

    def banding_file(self):
        path = filedialog.askopenfilename(
            title="Foto penyu yang mau dicek",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp"), ("Semua", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self._status("Gagal membaca file itu.", BAD)
            return
        self._banding_ke_tersimpan(img, os.path.basename(path))

    def _banding_ke_tersimpan(self, img, label):
        """Bandingkan satu foto ke SEMUA penyu yang sudah didaftarkan.

        Ini yang membuat mode banding berguna: bukan menguji sistem dengan
        pasangan acak, tapi menjawab "penyu di depan saya ini yang mana dari
        yang sudah saya simpan?" — dengan bukti yang bisa diperiksa.
        """
        import banding

        g = self.faceid
        if not g.data:
            self._status("Belum ada penyu tersimpan. Buka mode 🐢 Penyu, "
                         "lalu pencet '➕ Daftarkan Penyu Ini' dulu.", WARN)
            return
        if not turtle_mode._patch_acuan:
            self._status("Foto acuan belum dimuat. Buka mode 🐢 Penyu sekali "
                         "supaya galerinya terbaca, lalu kembali ke sini.", WARN)
            return

        self._status(f"Membandingkan {label} ke {len(g.data)} penyu tersimpan...",
                     MUTED)
        self.root.update_idletasks()

        varian = self.deskriptor if self.deskriptor in ("T", "L") else "T"
        hasil = []
        for nama in g.data:
            acuan = turtle_mode._patch_acuan.get(nama)
            if acuan is None:
                continue
            try:
                h = banding.bandingkan(img, acuan, varian=varian)
            except Exception:
                continue
            hasil.append((nama, h))

        if not hasil:
            self._status("Tidak ada penyu tersimpan yang punya foto acuan. "
                         "Daftarkan dari mode Penyu supaya acuannya ikut tersimpan.",
                         WARN)
            return

        hasil.sort(key=lambda x: x[1]["jarak"])
        nama, h = hasil[0]
        self.banding_hasil = h
        self.banding_gambar = banding.gambar(h, label, f"tersimpan: {nama}", None)
        self.banding_peringkat = [
            (n, x["jarak"], x["sisik"]["jumlah"] if x["sisik"] else 0)
            for n, x in hasil[:6]]
        self.proses()

        n_sisik = h["sisik"]["jumlah"] if h["sisik"] else 0
        if h["putusan"] == "INDIVIDU SAMA":
            self._status(f"{label} paling cocok dengan '{nama}' — jarak "
                         f"{h['jarak']:.4f}, {n_sisik} pasang pola sisik.", OK)
        elif h["putusan"] == "RAGU":
            self._status(f"{label}: kandidat terdekat '{nama}' ({h['jarak']:.4f}) "
                         f"tapi belum meyakinkan. Perlu foto lebih jelas.", WARN)
        else:
            self._status(f"{label} TIDAK cocok dengan satu pun dari "
                         f"{len(hasil)} penyu tersimpan. Terdekat '{nama}' "
                         f"({h['jarak']:.4f}). Kemungkinan individu baru.", ACCENT)

    def pilih_dua_file(self):
        pa = filedialog.askopenfilename(title="Foto PERTAMA")
        if not pa:
            return
        pb = filedialog.askopenfilename(title="Foto KEDUA")
        if not pb:
            return
        self._banding_dari(pa, pb, os.path.basename(pa), os.path.basename(pb), None)

    def _banding_dari(self, pa, pb, la, lb, kebenaran):
        import banding
        a, b = cv2.imread(pa), cv2.imread(pb)
        if a is None or b is None:
            self._status("Salah satu foto gagal dibaca.", BAD)
            return
        self._status("Membandingkan...", MUTED)
        self.root.update_idletasks()
        try:
            h = banding.bandingkan(a, b, varian=self.deskriptor
                                   if self.deskriptor in ("T", "L") else "T")
        except Exception as e:
            self._status(f"Gagal membandingkan: {e}", BAD)
            return
        self.banding_hasil = h
        self.banding_peringkat = None   # mode uji: tidak ada peringkat galeri
        self.banding_gambar = banding.gambar(h, la, lb, kebenaran)
        self.banding_label = (la, lb, kebenaran)
        self.proses()

        n = h["sisik"]["jumlah"] if h["sisik"] else 0
        if kebenaran is None:
            self._status(f"{h['putusan']} — jarak {h['jarak']:.4f}, "
                         f"{n} pasang pola sisik.", MUTED)
        else:
            benar = kebenaran == (h["putusan"] == "INDIVIDU SAMA")
            self._status(
                f"{la} vs {lb} — seharusnya "
                f"{'SAMA' if kebenaran else 'BEDA'}, sistem bilang "
                f"{h['putusan']} ({'BENAR' if benar else 'SALAH'}). "
                f"jarak {h['jarak']:.4f}, {n} pasang pola sisik.",
                OK if benar else BAD)

    # -------------------------------------------------------------- face id
    def _galeri_aktif(self):
        return self.faceid_muka if self.mode == "muka" else self.faceid

    def daftarkan(self):
        """Simpan subjek pada frame sekarang ke galeri Face ID mode aktif."""
        if self.mode not in ("penyu", "muka"):
            self._status("Pindah ke mode Penyu atau Muka Manusia dulu.", WARN)
            return
        if not self.hasil or self.hasil.get("vektor") is None:
            self._status(
                "Tidak ada wajah terdeteksi untuk didaftarkan."
                if self.mode == "muka" else "Belum ada vektor untuk didaftarkan.",
                WARN)
            return

        subjek = "Muka" if self.mode == "muka" else "Penyu"
        sedang_live = self.live
        self.henti_live()          # dialog modal akan membekukan loop video
        nama = simpledialog.askstring(
            f"Daftarkan {subjek}", f"Nama / ID {subjek.lower()} ini:",
            parent=self.root)
        if nama:
            if self.mode == "muka" and sedang_live:
                # Rekam beberapa frame, bukan satu. Satu contoh hanya mewakili
                # satu pose; beberapa frame menangkap variasi kecil sehingga
                # pengenalannya jauh lebih tahan gerakan dan pencahayaan.
                n = self._rekam_contoh(nama, jumlah=5)
                self._status(f"'{nama}' tersimpan dari {n} frame.", OK if n else WARN)
            else:
                ok, pesan = self._galeri_aktif().daftarkan(nama, self.hasil["vektor"])
                self._status(pesan, OK if ok else WARN)
            self.proses()
        else:
            self._status("Pendaftaran dibatalkan.", MUTED)
        if sedang_live:
            self.mulai_live()

    def _rekam_contoh(self, nama, jumlah=5):
        """Ambil beberapa frame berturut-turut untuk satu orang."""
        cap = cv2.VideoCapture(self.sumber[1])
        if not cap.isOpened():
            cap.release()
            ok, _ = self._galeri_aktif().daftarkan(nama, self.hasil["vektor"])
            return 1 if ok else 0
        tersimpan = 0
        for _ in range(jumlah * 4):        # coba lebih banyak, ambil yang lolos
            ret, f = cap.read()
            if not ret:
                break
            _, h = face_mode.jalankan(f, self._galeri_aktif())
            if h and h.get("vektor") is not None:
                if self._galeri_aktif().daftarkan(nama, h["vektor"])[0]:
                    tersimpan += 1
            if tersimpan >= jumlah:
                break
        cap.release()
        return tersimpan

    def _teks_faceid(self):
        f = self.hasil.get("faceid") if self.hasil else None
        g = self._galeri_aktif()
        subjek = "muka" if self.mode == "muka" else "penyu"
        judul = (f"TERDAFTAR: {g.jumlah_individu} {subjek}, "
                 f"{g.jumlah_contoh} contoh")
        if not f:
            return judul, "—", MUTED
        # Mode muka: tampilkan SEMUA wajah, bukan hanya subjek utama.
        if self.mode == "muka" and self.hasil:
            semua = self.hasil.get("semua_wajah") or []
            total = self.hasil.get("jumlah_wajah", 0)
            if not semua:
                return (judul, f"{total} wajah terdeteksi,\ntidak ada yang cukup "
                        f"jelas.\n\nMinimal {face_mode.MIN_UKURAN}px, "
                        f"skor {face_mode.MIN_SKOR}.", MUTED)
            baris = []
            for i, w in enumerate(semua, 1):
                tanda = "◆" if w["utama"] else " "
                if w["status"] == "dikenal":
                    baris.append(f"{tanda}{i}. {w['nama']}\n     {w['jarak']:.3f} ✓")
                elif w["status"] == "ragu":
                    baris.append(f"{tanda}{i}. RAGU\n"
                                 f"     {w['kandidat']}? {w['jarak']:.3f}")
                elif w["status"] == "tidak dikenal":
                    baris.append(f"{tanda}{i}. BELUM TERDAFTAR\n"
                                 f"     dekat {w['kandidat']} {w['jarak']:.3f}")
                else:
                    baris.append(f"{tanda}{i}. belum ada galeri")
            dikenal = sum(1 for w in semua if w["status"] == "dikenal")
            warna = OK if dikenal else WARN
            return (judul,
                    f"{len(semua)}/{total} wajah dinilai, {dikenal} dikenal\n"
                    f"◆ = target tombol ➕\n\n" + "\n".join(baris), warna)

        if f["status"] == "kosong":
            return judul, "Belum ada yang didaftarkan.\nPencet tombol ➕ di atas.", MUTED
        if f["status"] == "beda deskriptor":
            return (judul,
                    f"GALERI DARI DESKRIPTOR LAIN\n"
                    f"galeri {f['dim_galeri']}-dim,\nsekarang {f['dim_query']}-dim.\n\n"
                    f"Hapus {os.path.basename(self._galeri_aktif().path)}\n"
                    f"lalu daftar ulang.", BAD)
        # Persen (P individu sama) di depan, jarak mentah kecil di belakang —
        # riset kalibrasi: skor mentah tidak boleh jadi tampilan utama.
        baris = [f"{n:<10} {turtle_mode.prob_sama(d):>4.0%}  ({d:.3f})"
                 for n, d in f["peringkat"][:5]]
        pr = f.get("prob", turtle_mode.prob_sama(f["jarak"]))
        s_ak = turtle_mode.STATS[turtle_mode.STATS_AKTIF]["akurasi"]
        catatan = (f"\n\npersen = P(individu sama)\n"
                   f"keandalan kalibrasi: {s_ak:.0f}%")
        if f["status"] == "dikenal":
            return (judul, f"DIKENAL: {f['nama']}\n{pr:.0%} kemungkinan sama\n\n"
                    + "\n".join(baris) + catatan, OK)
        return (judul, f"TIDAK DIKENAL\nterdekat {f['kandidat']} ({pr:.0%})\n\n"
                + "\n".join(baris) + catatan, WARN)

    # ---------------------------------------------------------- deskriptor
    def set_deskriptor(self, key):
        """Ganti deskriptor mode penyu. Galeri harus dibangun ulang: vektor
        dari deskriptor berbeda sama sekali tidak sebanding."""
        if key != "baseline":
            import megadescriptor
            if key == "arcface" and not megadescriptor.arcface_tersedia():
                self._status("Bobot ArcFace belum ada / masih dilatih. "
                             "Cek: tail -f latih_arcface.log", WARN)
                return
            if not megadescriptor.tersedia():
                self._status("torch/timm belum terpasang. "
                             "Jalankan: .venv/bin/pip install torch timm", BAD)
                return
            self._status(f"Memuat MegaDescriptor-{key} "
                         "(unduhan pertama bisa beberapa menit)...", MUTED)
            self.root.update_idletasks()
            try:
                megadescriptor.muat(key)
            except Exception as e:
                self._status(f"Gagal memuat model: {e}", BAD)
                return
            turtle_mode.set_deskriptor(
                lambda bgr, v=key: megadescriptor.deskriptor(bgr, varian=v))
        else:
            turtle_mode.set_deskriptor(turtle_mode.deskriptor_baseline)

        self.deskriptor = key
        if key == "arcface" and "arcface" in turtle_mode.STATS:
            turtle_mode.set_stats("arcface")
        for k, b in self.btn_desc.items():
            b.set_aktif(k == key)
        self.galeri_penyu = None          # paksa bangun ulang dengan vektor baru
        self.faceid_perlu_reset = True
        if self.mode == "penyu":
            self.set_mode("penyu")
        else:
            self._status(f"Deskriptor penyu: {key}. Berlaku saat mode Penyu dibuka.", OK)

    # ----------------------------------------------------------------- mode
    def set_mode(self, mode):
        if mode == self.mode and self.stages:
            return
        self.mode = mode
        for k, b in self.btn_mode.items():
            b.set_aktif(k == mode)

        if mode == "penyu":
            self.henti_live()        # dataset penyu itu foto, bukan kamera
            if self.galeri_penyu is None:
                self._status("Memuat galeri penyu...", MUTED)
                self.root.update_idletasks()
                folder, maks = self.DATASET_PENYU[self.dataset_penyu]
                self.galeri_penyu, self.contoh_penyu, pesan = turtle_mode.muat_galeri(
                    folder=folder, params=self.params, maks_individu=maks)
                self._status(pesan, OK if self.galeri_penyu else BAD)
            if self.contoh_penyu:
                self.idx_contoh = 0
                self._muat_contoh()
            else:
                self._status(
                    f"Tidak ada foto penyu di {turtle_mode.GALERI_DEFAULT}. "
                    "Pakai 'Buka Gambar' untuk memilih foto sendiri.", BAD)
        elif mode == "banding":
            self.henti_live()      # perbandingan itu dua foto diam, bukan video
            self._status("Pilih pasangan: 'Pasangan SAMA' atau 'Pasangan BEDA' "
                         "untuk mengambil dari dataset, atau 'Pilih 2 File' "
                         "untuk foto sendiri.", MUTED)
            if self.banding_hasil is None:
                self.ambil_pasangan(sama=True)

        elif mode == "muka":
            if not face_mode.tersedia():
                self._status("Model wajah belum ada. Jalankan: "
                             ".venv/bin/python unduh_model_wajah.py", BAD)
            else:
                # Wajib ganti sumber gambar. Kalau tidak, deteksi wajah jalan di
                # foto penyu yang tertinggal dari mode sebelumnya — hasilnya 0
                # wajah dan pengguna mengira fiturnya rusak.
                self.mulai_live()
                if not self.live:
                    self.frame = self._layar_pesan(
                        "MODE MUKA BUTUH KAMERA",
                        "Kamera tidak bisa dibuka.",
                        "Pencet 'Buka Gambar' untuk memilih foto berisi wajah,",
                        "atau beri izin kamera lalu pencet 'Live Kamera'.")
                    self._status("Kamera tidak aktif — mode muka butuh wajah, "
                                 "bukan foto penyu. Buka gambar berisi wajah.", BAD)
                else:
                    self._status("Mode muka. Hadapkan wajah ke kamera, lalu "
                                 "pencet '➕ Daftarkan Muka Ini'.", OK)
        else:
            self.frame = self._gambar_awal()
            self._status("Mode kartu. Nyalakan Live Kamera atau buka gambar.", MUTED)

        self.btn_daftar.config(
            text={"muka": "➕  Daftarkan Muka Ini",
                  "penyu": "➕  Daftarkan Penyu Ini"}.get(mode, "➕  Daftarkan"))
        # Pemilih deskriptor hanya relevan untuk penyu; di mode lain cuma
        # bikin bingung karena angkanya tidak berlaku.
        if mode == "penyu":
            self.descbar.pack(fill="x", padx=12, pady=(4, 2), after=self.kambar)
            self.dsbar.pack(fill="x", padx=12, pady=(4, 2), after=self.descbar)
        else:
            self.descbar.pack_forget()
            self.dsbar.pack_forget()
        if mode == "banding":
            self.bandbar.pack(fill="x", padx=12, pady=(4, 2), after=self.kambar)
        else:
            self.bandbar.pack_forget()

        self.pilih_key = "asli"
        self.proses()

    @staticmethod
    def _layar_pesan(judul, *baris):
        """Gambar pesan sebagai frame, supaya pengguna melihatnya di kanvas
        besar dan bukan cuma di baris status yang mudah terlewat."""
        img = np.full((520, 900, 3), 26, np.uint8)
        cv2.putText(img, judul, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (110, 170, 250), 2)
        for i, t in enumerate(baris):
            cv2.putText(img, t, (40, 190 + i * 46), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (220, 220, 230), 1)
        return img

    def _muat_contoh(self):
        p = self.contoh_penyu[self.idx_contoh % len(self.contoh_penyu)]
        img = cv2.imread(p)
        if img is not None:
            self.frame = img
            # Nama individu = komponen path pertama SETELAH folder dataset.
            # Jangan pakai dirname dua kali: struktur kedua dataset berbeda
            # (utuh: id/view/file, kepala: id/file).
            akar = self.DATASET_PENYU[self.dataset_penyu][0]
            try:
                ind = os.path.relpath(p, akar).split(os.sep)[0]
            except ValueError:
                ind = os.path.basename(os.path.dirname(p))
            self._status(f"Contoh {self.idx_contoh+1}/{len(self.contoh_penyu)} "
                         f"— individu sebenarnya: {ind}  ({os.path.basename(p)})", OK)

    def contoh_berikut(self):
        """Ganti ke gambar contoh berikutnya dari dataset mode yang aktif."""
        if self.mode == "penyu":
            if not self.contoh_penyu:
                self._status("Belum ada contoh penyu yang dimuat.", WARN)
                return
            self.henti_live()
            self.idx_contoh += 1
            self._muat_contoh()
        else:
            d = os.path.join(pipeline.BASE_DIR, "Templatekartu")
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".png"))) if os.path.isdir(d) else []
            if not files:
                return
            self.henti_live()
            self.idx_contoh += 1
            f = files[self.idx_contoh % len(files)]
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                self.frame = self._beri_latar(img)
                self._status(f"Contoh kartu: {f}", OK)
        self.proses()

    def simpan(self):
        if not self.stages:
            return
        st = self._stage_aktif()
        nama = f"tahap_{st.key}.png"
        cv2.imwrite(os.path.join(pipeline.BASE_DIR, nama), st.image)
        self._status(f"Tersimpan: {nama}", OK)

    # ------------------------------------------------------------------- ui
    def _bangun_ui(self):
        # --- pemilih mode / dataset
        modebar = tk.Frame(self.root, bg=BG)
        modebar.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(modebar, text="MODE:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))
        self.btn_mode = {}
        for key, teks in [("kartu", "🂡  Kartu"),
                          ("penyu", "🐢  Penyu TurtleID2022"),
                          ("muka", "🙂  Muka Manusia"),
                          ("banding", "⚖️  Bandingkan 2 Foto")]:
            b = Klik(modebar, teks, lambda k=key: self.set_mode(k))
            b.pack(side="left", padx=3)
            self.btn_mode[key] = b

        # --- pemilih sumber kamera
        self.kambar = kambar = tk.Frame(self.root, bg=BG)
        kambar.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(kambar, text="KAMERA:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))
        self.btn_kamera = {}
        for key, teks, jenis, nilai in [
            ("mac0", "📷  Webcam Mac", "mac", 0),
            ("mac1", "📷  Kamera Mac #1", "mac", 1),
            ("ip", "🌐  Kamera IP / HTTP…", "ip", None),
        ]:
            b = Klik(kambar, teks,
                     lambda j=jenis, n=nilai: self.set_sumber(j, n))
            b.pack(side="left", padx=3)
            self.btn_kamera[key] = b
        self.btn_kamera["mac0"].set_aktif(True)

        # --- tombol khusus mode banding
        self.bandbar = bandbar = tk.Frame(self.root, bg=BG)
        tk.Label(bandbar, text="BANDINGKAN:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))
        Klik(bandbar, "📷  Kamera vs Yang Tersimpan", self.banding_kamera,
             warna="#2d4a2d").pack(side="left", padx=3)
        Klik(bandbar, "📂  Foto vs Yang Tersimpan", self.banding_file,
             warna="#2d3d4a").pack(side="left", padx=3)
        Klik(bandbar, "🆚  Pilih 2 File Sendiri",
             self.pilih_dua_file).pack(side="left", padx=3)
        tk.Label(bandbar, text="  |  uji sistem:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9)).pack(side="left")
        Klik(bandbar, "SAMA", lambda: self.ambil_pasangan(True)).pack(side="left", padx=2)
        Klik(bandbar, "BEDA", lambda: self.ambil_pasangan(False)).pack(side="left", padx=2)

        # --- pemilih dataset penyu (hasil riset: crop kepala >> foto utuh)
        self.dsbar = dsbar = tk.Frame(self.root, bg=BG)
        tk.Label(dsbar, text="DATASET PENYU:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))
        self.btn_ds = {}
        for key, teks in [("utuh", "🌊  Foto utuh — by_individual (61%)"),
                          ("kepala", "🐢  Crop kepala — SeaTurtleIDHeads (79.5%)")]:
            b = Klik(dsbar, teks, lambda k=key: self.set_dataset(k))
            b.pack(side="left", padx=3)
            self.btn_ds[key] = b
        self.btn_ds["utuh"].set_aktif(True)
        Klik(dsbar, "🗂  Kelola Tersimpan", self.kelola_tersimpan).pack(
            side="left", padx=(14, 0))

        # --- pemilih deskriptor (hanya berlaku untuk mode penyu)
        self.descbar = descbar = tk.Frame(self.root, bg=BG)
        descbar.pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(descbar, text="DESKRIPTOR PENYU:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))
        self.btn_desc = {}
        for key, teks in [("baseline", "Baseline piksel  (6%)"),
                          ("T", "MegaDescriptor-T  (26.5%)"),
                          ("L", "MegaDescriptor-L  (28%)"),
                          ("arcface", "⭐ ArcFace fine-tuned")]:
            b = Klik(descbar, teks, lambda k=key: self.set_deskriptor(k))
            b.pack(side="left", padx=3)
            self.btn_desc[key] = b

        atas = tk.Frame(self.root, bg=BG)
        atas.pack(fill="x", padx=12, pady=(6, 6))
        self.btn_live = Klik(atas, "▶  Live Kamera", self.toggle_live)
        self.btn_live.pack(side="left")
        Klik(atas, "Buka Gambar", self.buka_file).pack(side="left", padx=6)
        Klik(atas, "Contoh Berikutnya ⏭", self.contoh_berikut).pack(side="left")
        Klik(atas, "Simpan Tahap Ini", self.simpan).pack(side="left", padx=6)
        # Teksnya diperbarui set_mode() — subjeknya beda tiap mode.
        self.btn_daftar = Klik(atas, "➕  Daftarkan", self.daftarkan)
        self.btn_daftar.pack(side="left")

        self.lbl_hasil = tk.Label(atas, text="", bg=BG, fg=OK,
                                  font=("Helvetica", 16, "bold"))
        self.lbl_hasil.pack(side="right")

        tk.Label(self.root, text="PENCET TAHAPNYA:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12)
        # Tombol tahap dibungkus jadi beberapa baris; 13 tombol tidak muat satu baris.
        self.bar = tk.Frame(self.root, bg=BG)
        self.bar.pack(fill="x", padx=12, pady=(2, 8))

        badan = tk.Frame(self.root, bg=BG)
        badan.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        # Panel kanan di-pack DULUAN supaya lebarnya terjamin. Kalau kanvas
        # yang duluan, gambar besar akan mendorong panel keluar jendela.
        kanan = tk.Frame(badan, bg=PANEL, width=340)
        kanan.pack(side="right", fill="y", padx=(10, 0))
        kanan.pack_propagate(False)

        kiri = tk.Frame(badan, bg="#111118")
        kiri.pack(side="left", fill="both", expand=True)
        kiri.pack_propagate(False)      # jangan ikut membesar mengikuti gambar
        self.kanvas = tk.Label(kiri, bg="#111118")
        self.kanvas.pack(fill="both", expand=True)

        self.lbl_judul = tk.Label(kanan, text="", bg=PANEL, fg=ACCENT,
                                  font=("Helvetica", 13, "bold"),
                                  wraplength=315, justify="left")
        self.lbl_judul.pack(anchor="w", padx=12, pady=(12, 2))
        self.lbl_note = tk.Label(kanan, text="", bg=PANEL, fg=WARN,
                                 font=("Courier", 10), wraplength=315, justify="left")
        self.lbl_note.pack(anchor="w", padx=12)
        self.lbl_detail = tk.Label(kanan, text="", bg=PANEL, fg=FG,
                                   font=("Helvetica", 10), wraplength=315,
                                   justify="left")
        self.lbl_detail.pack(anchor="w", padx=12, pady=(8, 12))

        ttk.Separator(kanan, orient="horizontal").pack(fill="x", padx=12)
        tk.Label(kanan, text="PARAMETER  (geser, video langsung berubah)",
                 bg=PANEL, fg=MUTED, font=("Helvetica", 9, "bold"),
                 wraplength=315, justify="left").pack(anchor="w", padx=12, pady=(10, 4))

        self.sliders = {}
        for key, label, lo, hi, res in [
            ("scale", "Skala", 0.2, 1.0, 0.05),
            ("blur", "Blur (ganjil)", 1, 21, 2),
            ("canny_lo", "Canny bawah", 0, 300, 1),
            ("canny_hi", "Canny atas", 0, 400, 1),
            ("thresh", "Threshold", 0, 255, 1),
            ("area_div", "Area / (kecil=besar)", 10, 600, 10),
        ]:
            self._slider(kanan, key, label, lo, hi, res)

        ttk.Separator(kanan, orient="horizontal").pack(fill="x", padx=12, pady=(8, 0))
        self.lbl_rank_judul = tk.Label(kanan, text="KECOCOKAN TERATAS", bg=PANEL,
                                       fg=MUTED, font=("Helvetica", 9, "bold"),
                                       wraplength=315, justify="left")
        self.lbl_rank_judul.pack(anchor="w", padx=12, pady=(8, 4))
        self.lbl_rank = tk.Label(kanan, text="", bg=PANEL, fg=FG,
                                 font=("Courier", 10), justify="left")
        self.lbl_rank.pack(anchor="w", padx=12)

        self.lbl_status = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                                   font=("Helvetica", 10), anchor="w",
                                   wraplength=1300, justify="left")
        self.lbl_status.pack(fill="x", padx=12, pady=(0, 8))

        self.root.bind("<Left>", lambda e: self._geser(-1))
        self.root.bind("<Right>", lambda e: self._geser(1))
        self.root.bind("<space>", lambda e: self.toggle_live())
        self.kanvas.bind("<Configure>", lambda e: self._tampil())
        self.root.protocol("WM_DELETE_WINDOW", self._tutup)

    def _slider(self, parent, key, label, lo, hi, res):
        baris = tk.Frame(parent, bg=PANEL)
        baris.pack(fill="x", padx=12, pady=1)
        tk.Label(baris, text=label, bg=PANEL, fg=FG, font=("Helvetica", 9),
                 width=17, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=getattr(self.params, key))
        # Daftarkan var SEBELUM Scale dibuat: tk.Scale memanggil command-nya
        # sekali saat inisialisasi, dan callback itu membaca self.sliders[key].
        self.sliders[key] = var
        tk.Scale(baris, from_=lo, to=hi, resolution=res, orient="horizontal",
                 variable=var, bg=PANEL, fg=FG, troughcolor="#1a1a26",
                 highlightthickness=0, showvalue=True, length=150,
                 command=lambda _v, k=key: self._ubah(k)).pack(side="right")

    def _ubah(self, key):
        val = self.sliders[key].get()
        setattr(self.params, key, val if key == "scale" else int(val))
        if not self.live:
            self.proses()        # saat live, frame berikutnya sudah memprosesnya

    # ------------------------------------------------------------- proses
    def proses(self):
        if self.mode == "banding":
            if self.banding_gambar is None:
                self.stages, self.hasil = [], None
                return
            h = self.banding_hasil
            n = h["sisik"]["jumlah"] if h["sisik"] else 0
            self.stages = [pipeline.Stage(
                "banding", "Perbandingan Dua Foto", self.banding_gambar,
                f"jarak {h['jarak']:.4f} · {n} pasang sisik",
                "Dua bukti sekaligus. GLOBAL: jarak MegaDescriptor, satu angka "
                "untuk seluruh gambar. LOKAL: garis berwarna menghubungkan pola "
                "sisik yang cocok. Kalau global bilang mirip tapi tidak ada "
                "pasangan sisik, kemiripannya datang dari latar atau ciri "
                "spesies — bukan dari identitas individu.")]
            self.hasil = None
            self._bangun_tombol()
            self._tampil()
            self.lbl_hasil.config(text=h["putusan"], fg={
                "INDIVIDU SAMA": OK, "RAGU": WARN,
                "INDIVIDU BERBEDA": ACCENT}[h["putusan"]])
            if self.banding_peringkat:
                self.lbl_rank_judul.config(text="PERINGKAT TERSIMPAN")
                self.lbl_rank.config(fg=FG, text="\n".join(
                    f"{i}. {n:<10} {j:.4f}\n   {s} sisik"
                    for i, (n, j, s) in enumerate(self.banding_peringkat, 1)))
                return
            self.lbl_rank_judul.config(text="BUKTI")
            self.lbl_rank.config(fg=FG, text=(
                f"GLOBAL  {h['jarak']:.4f}\n"
                f"ambang  {h['ambang']:.3f}\n\n"
                f"LOKAL   {n} pasang\n\n"
                f"jarak ini:\n"
                f"{h['z_sama']:+.2f} SD dari SAMA\n"
                f"{h['z_beda']:+.2f} SD dari BEDA\n\n"
                f"Kedua distribusi\nbertumpang tindih —\n"
                f"itulah masalahnya."))
            return

        if self.mode == "muka":
            self.stages, self.hasil = face_mode.jalankan(
                self.frame, self.faceid_muka, self.params)
        elif self.mode == "penyu":
            self.stages, self.hasil = turtle_mode.jalankan(
                self.frame, self.galeri_penyu or {}, self.params, faceid=self.faceid)
        else:
            self.stages, self.hasil = pipeline.jalankan(
                self.frame, self.templates, self.params)
        self._bangun_tombol()
        self._tampil()

        if self.mode in ("penyu", "muka"):
            # Yang penting di sini hasil Face ID, bukan kecocokan dataset.
            judul, teks, warna = self._teks_faceid()
            self.lbl_rank_judul.config(text=judul)
            self.lbl_rank.config(text=teks, fg=warna)
            f = self.hasil.get("faceid") if self.hasil else None
            if f and f["status"] == "dikenal":
                self.lbl_hasil.config(text=f"✓ {f['nama']}  ({f['jarak']:.3f})", fg=OK)
            elif f and f["status"] == "tidak dikenal":
                self.lbl_hasil.config(text=f"? TIDAK DIKENAL  ({f['jarak']:.3f})",
                                      fg=WARN)
            else:
                subjek = "muka" if self.mode == "muka" else "penyu"
                self.lbl_hasil.config(text=f"belum ada {subjek} terdaftar", fg=MUTED)
            return

        self.lbl_rank_judul.config(text="KECOCOKAN TERATAS")
        if self.hasil:
            nama, skor = self.hasil["nama"], self.hasil["skor"]
            self.lbl_hasil.config(text=f"{nama}   ({skor:.4f})",
                                  fg=OK if self.hasil["diterima"] else WARN)
            self.lbl_rank.config(fg=FG, text="\n".join(
                f"{i+1}. {n:<12} {s:.4f}"
                for i, (n, s) in enumerate(self.hasil["peringkat"][:6])))
        else:
            self.lbl_hasil.config(text="tidak ada kartu", fg=MUTED)
            self.lbl_rank.config(text="—", fg=FG)

    def _stage_aktif(self):
        for st in self.stages:
            if st.key == self.pilih_key:
                return st
        return self.stages[0]

    def _bangun_tombol(self):
        """Tombol dibangun ulang hanya kalau daftar tahap berubah — kalau tiap
        frame dibangun ulang, tombolnya berkedip dan klik jadi meleset."""
        keys = [st.key for st in self.stages]
        if keys == list(self.tombol.keys()):
            self._warnai()
            return
        for w in self.bar.winfo_children():
            w.destroy()
        self.tombol = {}
        baris = None
        for i, st in enumerate(self.stages):
            if i % PER_BARIS == 0:
                baris = tk.Frame(self.bar, bg=BG)
                baris.pack(anchor="w", pady=1)
            b = Klik(baris, LABEL.get(st.key, st.key),
                     lambda k=st.key: self._pilih(k))
            b.pack(side="left", padx=2)
            self.tombol[st.key] = b
        if self.pilih_key not in self.tombol:
            self.pilih_key = self.stages[0].key
        self._warnai()

    def _warnai(self):
        for k, b in self.tombol.items():
            b.set_aktif(k == self.pilih_key)

    def _pilih(self, key):
        self.pilih_key = key
        self._warnai()
        self._tampil()

    def _geser(self, d):
        if not self.stages:
            return
        keys = [s.key for s in self.stages]
        i = keys.index(self.pilih_key) if self.pilih_key in keys else 0
        self._pilih(keys[(i + d) % len(keys)])

    def _tampil(self):
        if not self.stages:
            return
        st = self._stage_aktif()
        img = st.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cw = max(self.kanvas.winfo_width(), 50)
        ch = max(self.kanvas.winfo_height(), 50)
        h, w = rgb.shape[:2]
        f = min(cw / w, ch / h)
        if f > 0:
            rgb = cv2.resize(rgb, (max(1, int(w * f)), max(1, int(h * f))),
                             interpolation=cv2.INTER_NEAREST if f > 1 else cv2.INTER_AREA)

        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.kanvas.config(image=self._photo)

        fps = f"   •   {self._fps:.0f} FPS" if self.live else ""
        self.lbl_judul.config(text=st.title + fps)
        self.lbl_note.config(text=st.note)
        self.lbl_detail.config(text=st.detail)

    def _status(self, teks, warna=MUTED):
        self.lbl_status.config(text=teks, fg=warna)

    def _tutup(self):
        self.henti_live()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
