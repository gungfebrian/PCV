# PCV

Visualizer pipeline Computer Vision dengan tiga mode: pengenalan kartu remi,
re-identification penyu (TurtleID2022), dan Face ID wajah manusia.

## Cara menjalankan

Semua paket ada di `.venv/`, **bukan** di Python global.

```bash
.venv/bin/python visualizer.py
```

atau `source .venv/bin/activate` dulu, lalu `python visualizer.py`.

## File utama

| File | Guna |
|---|---|
| **`visualizer.py`** | **Mulai dari sini.** Tiga mode dalam satu jendela |
| `pipeline.py` | Pipeline kartu + kelas `Stage` dan `Params` bersama |
| `turtle_mode.py` | Pipeline penyu (deskriptor bisa ditukar) |
| `face_mode.py` | Pipeline wajah manusia (YuNet + SFace) |
| `faceid_penyu.py` | Galeri Face ID: daftar, kenali, ambang, kalibrasi |
| `megadescriptor.py` | Pembungkus MegaDescriptor asli (timm + torch) |
| `has.py` | Game tebak kartu dengan kamera (pygame) |
| `game_gui_final.py` | Game tebak kartu GUI murni, tanpa kamera |

Sekali jalan untuk menyiapkan aset:

```bash
.venv/bin/python unduh_template.py       # 52 template kartu HD
.venv/bin/python unduh_model_wajah.py    # YuNet + SFace (~39 MB)
```

File mati (`a.py`, `arob.py`, `test.py`, `game_ui.py`, `game.html`) sudah
dihapus. `Card_Recognizer-master/` dibiarkan sebagai referensi walau tidak
bisa jalan di Python 3.14.

## Kontrol visualizer

- **MODE** — Kartu / Penyu TurtleID2022 / Muka Manusia, berganti di jendela yang sama
- **KAMERA** — Webcam Mac / Kamera Mac #1 / Kamera IP-HTTP (DroidCam, IP Webcam)
- **DESKRIPTOR PENYU** — Baseline / MegaDescriptor-T / MegaDescriptor-L (hanya muncul di mode penyu)
- **Tombol tahap** — Gray, Blur, Edges, Kontur, CLAHE, Warp, … pilih yang tampil besar
- **Slider** — efeknya langsung terlihat, termasuk saat video live jalan
- **➕ Daftarkan** — simpan subjek pada frame ini ke galeri Face ID
- `←` `→` pindah tahap, `Spasi` jeda/lanjut

Sumber kamera dipilih lewat tombol **KAMERA**. Untuk kamera IP, URL awalnya
diambil dari `CAM_URL` dan bisa diubah saat aplikasi jalan. Kamera IP dicek
dengan soket berbatas waktu dulu — kalau tidak menjawab, statusnya dilaporkan
alih-alih menggantungkan UI 30 detik.

## Kerangka bersama

Ketiga mode memakai kerangka yang sama — kerangka yang juga dipakai
MegaDescriptor dan sistem re-ID mana pun:

```
           DETECT              ALIGN                DESCRIBE           MATCH
Kartu      kontur 4 sudut      warpPerspective      threshold Otsu     SAD
Penyu      kontur terbesar     crop + resize 224    MegaDescriptor     cosine
Wajah      YuNet               alignCrop (mata)     SFace 128-dim      cosine
```

## Hasil pengukuran

Re-ID penyu, 20 individu TurtleID2022, 200 foto **held-out** (tidak ada di galeri):

| Deskriptor | Top-1 | Top-5 |
|---|---|---|
| Tebak acak | 5.0% | — |
| Baseline piksel | 6.0% | 30.5% |
| MegaDescriptor-T-224 | **26.5%** | **56.0%** |
| MegaDescriptor-L-384 | **28.0%** | **59.5%** |

Model kecil dengan pra-proses benar (26.5%) hampir menyamai model besar
(28.0%) yang parameternya 7x lipat — pra-proses lebih menentukan daripada
ukuran model.

### Pengaruh pra-proses (MegaDescriptor-T, protokol sama)

| Pra-proses | Top-1 | Top-5 |
|---|---|---|
| **Foto utuh, tanpa crop & tanpa mask** | **21.0%** | **51.5%** |
| Crop ROI kontur | 17.0% | 39.0% |
| Foto utuh + masking GrabCut | 13.5% | 41.5% |
| Crop ROI + masking GrabCut | 9.5% | 38.0% |

**Tidak melakukan apa-apa mengungguli semuanya.** Masking heuristik memangkas
akurasi lebih dari setengah. Masking hanya menolong kalau mask-nya benar —
GrabCut hanya menempatkan 18.7% keypoint di objek, padahal luas objeknya 24.2%.

Karena itu `MASKING` dan crop ROI keduanya bukan jalan maju. Yang dibutuhkan
adalah detektor kepala penyu terlatih, yang sekaligus menyelesaikan DETECT,
ALIGN, dan menyediakan mask yang benar.

## Batasan yang diketahui

1. **DETECT penyu salah sasaran.** Kontur terbesar sering menemukan riak pasir,
   bukan penyunya. Karena itu `CROP_ROI` dimatikan — memakai crop tersebut
   justru menurunkan Top-1 dari 26.5% ke 17.0%. Kotaknya tetap digambar di
   tahap "Wilayah Objek" supaya terlihat seberapa sering meleset. Perbaikan
   sebenarnya butuh detektor kepala penyu terlatih.
2. **Ambang penyu belum bermakna.** `kalibrasi_ambang()` pada deskriptor
   baseline hanya mencapai akurasi seimbang 51.5% (acak = 50%), artinya tidak
   ada ambang yang benar-benar memisahkan. Wajib dikalibrasi ulang setelah
   pindah ke MegaDescriptor.
3. **Wajah kecil ditolak.** Wajah < 110 px atau skor < 0.85 sengaja tidak
   dikenali — pada pengujian, wajah 83x139 px salah dikenali sebagai orang lain.

## Catatan lingkungan

- Python 3.14 (Homebrew). TensorFlow tidak punya build untuk 3.14, jadi
  `Card_Recognizer-master/` tidak bisa dijalankan tanpa venv 3.11/3.12.
- `pygame` tidak punya wheel 3.14; yang terpasang `pygame-ce` (impor tetap `pygame`).
- tkinter butuh formula brew `python-tk@3.14`.
- MegaDescriptor memakai MPS (GPU Apple Silicon) kalau tersedia.
- Kamera perlu izin di System Settings → Privacy & Security → Camera.
